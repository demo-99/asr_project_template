import torch
from torch import nn

from hw_asr.base import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size=11, stride=1, dilation=1, padding=0,
                 dropout=0.2, tcs=False):
        super().__init__()
        layers = []
        if tcs:
            layers.append(nn.Conv1d(in_feats, in_feats, kernel_size, stride=stride,
                                    dilation=dilation, padding=padding, groups=in_feats))
        layers.append(nn.Conv1d(in_feats, out_feats, kernel_size, stride=stride, dilation=dilation, padding=padding))
        self.conv_block = nn.Sequential(
            *layers,
            nn.BatchNorm1d(out_feats, eps=1e-3, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.conv_block(x)


class TCSConvBlock(nn.Module):
    def __init__(self, in_feats, out_feats, repeat=3,
                 kernel_size=11, stride=1, residual=True,
                 dilation=1, dropout=0.2, tcs=False):
        super().__init__()
        padding_val = (dilation * kernel_size) // 2 - 1 if dilation > 1 else kernel_size // 2
        layers = [ConvBlock(in_feats, out_feats, kernel_size=kernel_size, stride=stride, dilation=dilation,
                            padding=padding_val, dropout=dropout, tcs=tcs)]
        for _ in range(repeat-1):
            layers.append(ConvBlock(out_feats, out_feats, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                    padding=padding_val, dropout=dropout, tcs=tcs))
        self.net = nn.Sequential(*layers)
        self.residual = residual
        if self.residual:
            self.residual_layer = ConvBlock(
                in_feats, out_feats, kernel_size=1, dropout=dropout)

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            out += self.residual_layer(x)
        return out


class QuartzNet(BaseModel):
    def __init__(self, n_feats, n_class, model_config, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)

        self.stride = 1

        layers = []
        in_feats = n_feats
        for cfg in model_config:
            self.stride *= cfg['stride']

            tcs = cfg.get('tcs', False)
            residual = cfg.get('residual', True)
            layers.append(
                TCSConvBlock(
                    in_feats,
                    cfg['hidden'],
                    repeat=cfg['repeat'],
                    kernel_size=cfg['kernel'],
                    stride=cfg['stride'],
                    dilation=cfg['dilation'],
                    dropout=cfg.get('dropout', 0.0),
                    residual=residual,
                    tcs=tcs
                )
            )
            in_feats = cfg['hidden']

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Conv1d(1024, n_class, kernel_size=1, bias=True)

    def forward(self, spectrogram, *args, **kwargs):
        spectrogram = spectrogram.permute(0, 2, 1)
        out = self.encoder(spectrogram)
        print(self.fc(out).size())
        return self.fc(out).permute(0, 2, 1)

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
