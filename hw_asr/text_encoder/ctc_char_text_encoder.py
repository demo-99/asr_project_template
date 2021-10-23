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
        blank_idx = len(chars)

        # initialise beam state
        last = BeamList()
        labeling = ()
        last.entries[labeling] = BeamEntry()
        last.entries[labeling].pr_blank = log(1)
        last.entries[labeling].pr_total = log(1)

        # go over all time-steps
        for t in range(char_length):
            curr = BeamList()

            # get beam-labelings of best beams
            best_labelings = last.sort_labelings()[:beam_size]

            # go over best beams
            for labeling in best_labelings:

                # probability of paths ending with a non-blank
                pr_non_blank = log(0)
                # in case of non-empty beam
                if labeling:
                    pr_non_blank = last.entries[labeling].pr_non_blank + log(mat[t, labeling[-1]])

                pr_blank = last.entries[labeling].pr_total + log(mat[t, blank_idx])

                curr.entries[labeling].labeling = labeling
                curr.entries[labeling].pr_non_blank = np.logaddexp(curr.entries[labeling].pr_non_blank, pr_non_blank)
                curr.entries[labeling].pr_blank = np.logaddexp(curr.entries[labeling].pr_blank, pr_blank)
                curr.entries[labeling].pr_total = np.logaddexp(curr.entries[labeling].pr_total,
                                                               np.logaddexp(pr_blank, pr_non_blank))
                curr.entries[labeling].pr_text = last.entries[labeling].pr_text
                curr.entries[
                    labeling].lm_applied = True

                for c in range(voc_size - 1):
                    new_labeling = labeling + (c,)
                    if labeling and labeling[-1] == c:
                        pr_non_blank = last.entries[labeling].pr_blank + log(mat[t, c])
                    else:
                        pr_non_blank = last.entries[labeling].pr_total + log(mat[t, c])
                    curr.entries[new_labeling].labeling = new_labeling
                    curr.entries[new_labeling].pr_non_blank = np.logaddexp(curr.entries[new_labeling].pr_non_blank,
                                                                           pr_non_blank)
                    curr.entries[new_labeling].pr_total = np.logaddexp(curr.entries[new_labeling].pr_total,
                                                                       pr_non_blank)
                    apply_lm(curr.entries[labeling], curr.entries[new_labeling], chars, lm)
            last = curr

        last.normalize()
        best_labeling = last.sort_labelings()[0]
        res = ''.join([chars[label] for label in best_labeling])
        return res
        return sorted(hypos, key=lambda x: x[1], reverse=True)
