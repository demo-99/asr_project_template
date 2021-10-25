import torch
import torch.nn.functional as F
from torch import nn


class MaskCNN(nn.Module):
    def __init__(self, sequential: nn.Sequential) -> None:
        super().__init__()
        self.sequential = sequential

    def forward(self, inputs, seq_length):
        output = None

        for module in self.sequential:
            output = module(inputs)
            mask = torch.BoolTensor(output.size()).fill_(0)

            if output.is_cuda:
                mask = mask.cuda()

            seq_length = self._get_sequence_length(module, seq_length)

            for i in range(mask.size(0)):
                if (mask[i].size(2) - seq_length) > 0:
                    mask[i].narrow(dim=2, start=seq_length, length=mask[i].size(2) - seq_length).fill_(1)

            output = output.masked_fill(mask, 0)

        return output

    def _get_sequence_length(self, module, seq_length):
        if isinstance(module, nn.Conv2d):
            numerator = seq_length + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_length = numerator / module.stride[1]
            seq_length = seq_length + 1

        elif isinstance(module, nn.MaxPool2d):
            seq_length *= 2

        return seq_length


class BNReluRNN(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_state_dim: int = 512,
            bidirectional: bool = True,
            dropout_p: float = 0.1,
    ):
        super(BNReluRNN, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_state_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )

    def forward(self, inputs):
        outputs = F.relu(self.batch_norm(inputs.permute(0, 2, 1)))
        outputs, hidden_states = self.rnn(outputs.permute(0, 2, 1))
        return outputs


class DeepSpeech2(nn.Module):
    def __init__(
            self,
            n_feats: int,
            n_class: int,
            num_rnn_layers: int = 5,
            rnn_hidden_dim: int = 512,
            dropout_p: float = 0.1,
            bidirectional: bool = True,
            device: torch.device = 'cuda',
    ):
        super(DeepSpeech2, self).__init__()
        self.device = device
        self.conv = MaskCNN(
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
            )
        )
        self.layers = nn.ModuleList()
        rnn_output_size = rnn_hidden_dim << 1 if bidirectional else rnn_hidden_dim

        for idx in range(num_rnn_layers):
            self.layers.append(
                BNReluRNN(
                    input_size=((n_feats + 1) // 2 + 1) // 2 * 32 if idx == 0 else rnn_output_size,
                    hidden_state_dim=rnn_hidden_dim,
                    bidirectional=bidirectional,
                    dropout_p=dropout_p,
                )
            )

        self.fc = nn.Sequential(
            nn.LayerNorm(rnn_output_size),
            nn.Linear(rnn_output_size, n_class, bias=False),
        )

    def forward(self, spectrogram, *args, **kwargs):
        inputs = spectrogram.unsqueeze(1).permute(0, 1, 3, 2)
        outputs = self.conv(inputs, inputs.size(-1))
        batch_size, num_channels, hidden_dim, seq_length = outputs.size()
        outputs = outputs.view(batch_size, num_channels * hidden_dim, seq_length).permute(2, 0, 1).contiguous()

        for layer in self.layers:
            outputs = layer(outputs)

        outputs = outputs.permute(1, 0, 2)
        return self.fc(outputs)

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
