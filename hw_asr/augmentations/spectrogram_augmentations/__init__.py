import torch

from hw_asr.augmentations.base import AugmentationBase


class RandomFourierTransform(AugmentationBase):
    def __call__(self, data: torch.Tensor):
        return torch.fft.rfft(data).abs().pow(2)
