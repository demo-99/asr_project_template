import torch
import torch_audiomentations

from hw_asr.augmentations.base import AugmentationBase
from random import random


class HahnWindow(AugmentationBase):
    def __init__(self, p: float = 0.2, *args, **kwargs):
        self.p = p

    def __call__(self, data: torch.Tensor):
        if random() < self.p:
            return torch.hann_window(data.size(-1)) * data
        else:
            return data
