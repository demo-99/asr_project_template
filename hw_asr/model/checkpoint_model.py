from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class CheckpointModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = nn.Sequential(
            nn.Conv1d(n_feats, fc_hidden, kernel_size=80),
            nn.BatchNorm1d(fc_hidden),
            nn.MaxPool1d(4),
            nn.Conv1d(fc_hidden, fc_hidden, kernel_size=3),
            nn.BatchNorm1d(fc_hidden),
            nn.MaxPool1d(4),
            nn.Conv1d(fc_hidden, 2 * fc_hidden, kernel_size=3),
            nn.BatchNorm1d(2 * fc_hidden),
            nn.MaxPool1d(4),
            nn.Conv1d(2 * fc_hidden, 2 * fc_hidden, kernel_size=3),
            nn.BatchNorm1d(2 * fc_hidden),
            nn.MaxPool1d(4),
            nn.Linear(2 * fc_hidden, n_class),
        )

    def forward(self, spectrogram, *args, **kwargs):
        return {"logits": self.net(spectrogram)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
