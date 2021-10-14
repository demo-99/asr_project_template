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
        'text': [],
    }

    def alignment(d, key):
        pad_sz = -1
        if key not in result_batch:
            pad_sz = d[key].size(-1)
            result_batch[key] = d[key]
            if key == 'text_encoded':
                result_batch['text_encoded_length'] = torch.Tensor([d[key].size(-1)])
        else:
            pad_sz = max(d[key].size(-1), pad_sz)
            result_batch[key] = torch.cat(
                (
                    F.pad(result_batch[key], (0, pad_sz-result_batch[key].size(-1))),
                    F.pad(d[key], (0, pad_sz-d[key].size(-1))),
                ),
                dim=0,
            )
            if key == 'text_encoded':
                result_batch['text_encoded_length'] = torch.cat(
                    (
                        result_batch['text_encoded_length'],
                        torch.Tensor([d[key].size(-1)]),
                    ),
                    dim=0,
                )

    for data in dataset_items:
        alignment(data, 'spectrogram')
        alignment(data, 'text_encoded')
        result_batch['text'].append(data['text'])

    result_batch['spectrogram'] = result_batch['spectrogram'].permute(0, 2, 1)

    return result_batch
