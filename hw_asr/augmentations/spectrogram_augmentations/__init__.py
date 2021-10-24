import torch

from random import random


from hw_asr.augmentations.base import AugmentationBase


class RandomFourierTransform(AugmentationBase):
    def __init__(self, n_fft: int = 1024, p: float = 0.2, *args, **kwargs):
        self.n_fft = n_fft
        self.p = p

    def __call__(self, data: torch.Tensor):
        if random() < self.p:
            return torch.fft.rfft(data, n=self.n_fft).abs().pow(2)
        else:
            return data
