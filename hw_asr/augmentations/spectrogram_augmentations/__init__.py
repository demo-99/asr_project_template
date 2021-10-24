import torch
import audiomentations

from random import random


from hw_asr.augmentations.base import AugmentationBase


class SpecFrequencyMask(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = audiomentations.SpecFrequencyMask(*args, **kwargs)

    def __call__(self, data: torch.Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
