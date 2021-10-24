from torch import Tensor
import torch_audiomentations

from hw_asr.augmentations.base import AugmentationBase


class PeakNormalization(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.PeakNormalization(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
