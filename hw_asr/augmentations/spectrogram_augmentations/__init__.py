import torch
import torchaudio

from ..random_apply import RandomApply
from hw_asr.augmentations.base import AugmentationBase


__all__ = [
    "RandomApply"
]


class FrequencyMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.FrequencyMasking(*args, **kwargs)

    def __call__(self, data: torch.Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
