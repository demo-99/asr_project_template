from math import log
from typing import List, Tuple

import numpy as np
import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        res = ""
        for i, ind in enumerate(inds):
            ind = ind.item() if torch.is_tensor(ind) else ind
            if i > 0 and ind == inds[i-1] or ind == 0:
                continue
            else:
                res += self.ind2char[ind]
        return res

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = []
        # best = probs[0].argsort()[-beam_size:]
        # for i in range(1, probs.shape[0]):
        #     new_hypos = [] * beam_size
        #     for j, b in enumerate(hypos):
        #         for k in probs[i]:
        #             new_hypos[j].append(k * b)
        #     hypos = np
        # best = prob
        return sorted(hypos, key=lambda x: x[1], reverse=True)
