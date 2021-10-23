import torch

from hw_asr.augmentations.base import AugmentationBase


class RandomFourierTransform(AugmentationBase):
    def __init__(self, n_fft=1024, *args, **kwargs):
        self.n_fft = n_fft

    def __call__(self, data: torch.Tensor):
        return torch.fft.rfft(data, n=self.n_fft).abs().pow(2)
