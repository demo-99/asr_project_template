import logging
from typing import List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {
        'audio_path': [],
        'duration': [],
        'text': [],
    }

    def alignment(d, key, pad_sz):
        if key not in result_batch:
            pad_sz = d[key].size(-1)
            result_batch[key] = d[key]
            result_batch[key + '_length'] = torch.Tensor([d[key].size(-1)]).type(torch.int64)
        else:
            pad_sz = max(d[key].size(-1), pad_sz)
            result_batch[key + '_length'] = torch.cat(
                (
                    result_batch[key + '_length'],
                    torch.Tensor([d[key].size(-1)]).type(torch.int64),
                ),
                dim=0,
            )
            result_batch[key] = torch.cat(
                (
                    F.pad(result_batch[key], (0, pad_sz-result_batch[key].size(-1))),
                    F.pad(d[key], (0, pad_sz-d[key].size(-1))),
                ),
                dim=0,
            )

        return pad_sz

    spectrogram_pad = -1
    text_encoded_pad = -1
    audio_pad = -1
    for data in dataset_items:
        spectrogram_pad = alignment(data, 'spectrogram', spectrogram_pad)
        text_encoded_pad = alignment(data, 'text_encoded', text_encoded_pad)
        audio_pad = alignment(data, 'audio', audio_pad)
        result_batch['audio_path'].append(data['audio_path'])
        result_batch['duration'].append(data['duration'])
        result_batch['text'].append(data['text'])

    result_batch['spectrogram'] = result_batch['spectrogram'].permute(0, 2, 1)

    return result_batch
