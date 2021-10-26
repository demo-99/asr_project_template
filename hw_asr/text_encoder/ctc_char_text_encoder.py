from typing import List, Tuple
from ctcdecode import CTCBeamDecoder

import numpy as np
import torch

from hw_asr.language_models.gram import pretrained_language_model
from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str], beam_width: int = 100, alpha: int = 0.5, beta: int = 1.0,
                 model_path: str = '3-gram.pruned.1e-7.arpa.gz'):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.beam_search = CTCBeamDecoder(
            [self.EMPTY_TOK] + alphabet,
            model_path=pretrained_language_model(model_path),
            alpha=alpha,
            beta=beta,
            beam_width=beam_width,
            log_probs_input=True
        )

    def ctc_decode(self, inds: List[int]) -> str:
        res = ""
        for i, ind in enumerate(inds):
            ind = ind.item() if torch.is_tensor(ind) else ind
            if i > 0 and ind == inds[i-1] or ind == 0:
                continue
            else:
                res += self.ind2char[ind]
        return res

    def ctc_beam_search(self, probs: torch.tensor, log_probs_length=None, beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        # hypos = []
        # best = probs[0].argsort()[-beam_size:]
        # for i in range(1, probs.shape[0]):
        #     new_hypos = [] * beam_size
        #     for j, b in enumerate(hypos):
        #         for k in probs[i]:
        #             new_hypos[j].append(k * b)
        #     hypos = np
        # best = prob
        beam_results, beam_scores, timesteps, out_lens = self.beam_search.decode(probs)
        res = ''.join([self.ind2char[int(i)] for i in beam_results[0][0][:out_lens[0][0]]])
        return res
