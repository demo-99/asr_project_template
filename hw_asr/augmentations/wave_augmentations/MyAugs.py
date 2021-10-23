import torch_audiomentations
import torch

from hw_asr.augmentations.base import AugmentationBase


class HahnWindow(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: torch.Tensor):
        return torch.hann_window(data.size(-1)) * data
