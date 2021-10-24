from torch import Tensor
import torch_audiomentations

from hw_asr.augmentations.base import AugmentationBase


class PeakNormalization(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.PeakNormalization(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data.numpy())
