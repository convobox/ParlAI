"""
RankerAgent based on PolyencoderAgent.
"""

import torch
import parlai.utils.logging as logging
from parlai.agents.transformer.polyencoder import PolyencoderAgent
from parlai.utils.torch import padded_3d


class RankerAgent(PolyencoderAgent):
    '''
    A polyencoder that read candidates dynamically.
    '''
    def set_fixed_candidates(self,
                             shared=None,
                             cands=[],
                             cands_score_ratios=[]):
        """
        make fixed candidates easy to change.
        """
        if len(cands) == 0:
            return
        vecs = self._make_candidate_vecs(cands)
        self.fixed_candidates = cands
        self.num_fixed_candidates = len(self.fixed_candidates)
        self.fixed_candidate_vecs = vecs
        if self.use_cuda:
            self.fixed_candidate_vecs = self.fixed_candidate_vecs.cuda()
        if self.encode_candidate_vecs:
            encs = self._make_candidate_encs(self.fixed_candidate_vecs)
            self.fixed_candidate_encs = encs
            if self.use_cuda:
                self.fixed_candidate_encs = self.fixed_candidate_encs.cuda()
            if self.fp16:
                self.fixed_candidate_encs = self.fixed_candidate_encs.half()
            else:
                self.fixed_candidate_encs = self.fixed_candidate_encs.float()
        self.cands_score_ratios = cands_score_ratios

    def _make_candidate_vecs(self, cands):
        """
        Prebuild cached vectors for fixed candidates.
        """
        cand_batches = [cands[i:i + 512] for i in range(0, len(cands), 512)]
        cand_vecs = []
        for batch in cand_batches:
            cand_vecs.extend(self.vectorize_fixed_candidates(batch))
        return padded_3d([cand_vecs],
                         pad_idx=self.NULL_IDX,
                         dtype=cand_vecs[0].dtype).squeeze(0)

    def _make_candidate_encs(self, vecs):
        """
        Make candidate encs.

        """
        cand_encs = []
        bsz = self.opt.get('encode_candidate_vecs_batchsize', 256)
        vec_batches = [vecs[i:i + bsz] for i in range(0, len(vecs), bsz)]
        self.model.eval()
        with torch.no_grad():
            for vec_batch in vec_batches:
                cand_encs.append(self.encode_candidates(vec_batch).cpu())
        rep = torch.cat(cand_encs, 0).to(vec_batch.device)
        return rep.transpose(0, 1).contiguous()

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        scores = super().score_candidates(batch, cand_vecs, cand_encs)
        if len(self.cands_score_ratios) == scores.shape[1]:
            scores_ratios = torch.Tensor([self.cands_score_ratios
                                          ]).to(scores.device)
            scores *= scores_ratios
        return scores