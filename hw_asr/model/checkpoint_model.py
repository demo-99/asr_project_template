import torch
from torch import nn

from hw_asr.base import BaseModel


class CheckpointModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.fc_hidden = fc_hidden
        self.lstm = nn.LSTM(n_feats, fc_hidden, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(fc_hidden, n_class)

    def init_hidden(self):
        return (torch.randn(1, 4, self.fc_hidden, device='cuda'),
                torch.randn(1, 4, self.fc_hidden, device='cuda'))

    def forward(self, spectrogram, *args, **kwargs):
        self.hidden = self.init_hidden()
        logits, self.hidden = self.lstm(spectrogram, self.hidden)
        logits = self.fc(logits)
        return logits

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
