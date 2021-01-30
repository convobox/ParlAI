"""
Multi Response PolyencoderAgent based on PolyencoderAgent.
"""

import random
from itertools import islice
from parlai.core.opt import Opt
from parlai.core.metrics import AverageMetric
from parlai.agents.transformer.polyencoder import PolyencoderAgent
from parlai.core.torch_agent import Output
from parlai.core.params import ParlaiParser


class MrpolyencoderAgent(PolyencoderAgent):
    @staticmethod
    def add_cmdline_args(argparser: ParlaiParser):
        """
        Add command-line arguments specifically for this agent.
        """
        PolyencoderAgent.add_cmdline_args(argparser)
        argparser.add_argument(
            '--response-num',
            type=int,
            default=1,
            help='response number of when act',
        )
        argparser.add_argument(
            '--response-sep',
            type=str,
            default='__ressep__',
            help='response resperator',
        )

    def __init__(self, opt: Opt, shared=None):
        self._response_seperator = opt.get('response_sep', '')
        self._response_num = opt.get('response_num', 1)
        if self._response_num > 1 and len(self._response_seperator) == 0:
            raise ValueError(
                'Response seperator empty while response number over 1.')
        super().__init__(opt, shared)

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        batchsize = (batch.text_vec.size(0)
                     if batch.text_vec is not None else batch.image.size(0))
        self.model.eval()

        cands, cand_vecs, label_inds = self._build_candidates(
            batch, source=self.eval_candidates, mode='eval')

        cand_encs = None
        if self.encode_candidate_vecs and self.eval_candidates in [
                'fixed', 'vocab'
        ]:
            # if we cached candidate encodings for a fixed list of candidates,
            # pass those into the score_candidates function
            if self.fixed_candidate_encs is None:
                self.fixed_candidate_encs = self._make_candidate_encs(
                    cand_vecs).detach()
            if self.eval_candidates == 'fixed':
                cand_encs = self.fixed_candidate_encs
            elif self.eval_candidates == 'vocab':
                cand_encs = self.vocab_candidate_encs

        scores = self.score_candidates(batch, cand_vecs, cand_encs=cand_encs)
        if self.rank_top_k > 0:
            sorted_scores, ranks = scores.topk(min(self.rank_top_k,
                                                   scores.size(1)),
                                               1,
                                               largest=True)
        else:
            sorted_scores, ranks = scores.sort(1, descending=True)

        if self.opt.get('return_cand_scores', False):
            sorted_scores = sorted_scores.cpu()
        else:
            sorted_scores = None

        # Update metrics
        if label_inds is not None:
            loss = self.criterion(scores, label_inds)
            self.record_local_metric('loss', AverageMetric.many(loss))
            ranks_m = []
            mrrs_m = []
            for b in range(batchsize):
                rank = (ranks[b] == label_inds[b]).nonzero()
                rank = rank.item() if len(rank) == 1 else scores.size(1)
                ranks_m.append(1 + rank)
                mrrs_m.append(1.0 / (1 + rank))
            self.record_local_metric('rank', AverageMetric.many(ranks_m))
            self.record_local_metric('mrr', AverageMetric.many(mrrs_m))

        ranks = ranks.cpu()
        max_preds = self.opt['cap_num_predictions']
        cand_preds = []
        for i, ordering in enumerate(ranks):
            if cand_vecs.dim() == 2:
                cand_list = cands
            elif cand_vecs.dim() == 3:
                cand_list = cands[i]
            # using a generator instead of a list comprehension allows
            # to cap the number of elements.
            cand_preds_generator = (cand_list[rank] for rank in ordering
                                    if rank < len(cand_list))
            cand_preds.append(list(islice(cand_preds_generator, max_preds)))

        if (self.opt.get('repeat_blocking_heuristic', True)
                and self.eval_candidates == 'fixed'):
            cand_preds = self.block_repeats(cand_preds)

        if self.opt.get('inference', 'max') == 'topk':
            # Top-k inference.
            preds = []
            for i in range(batchsize):
                preds.append(random.choice(cand_preds[i][0:self.opt['topk']]))
        else:
            preds = [cand_preds[i][0] for i in range(batchsize)]

        # multi-response
        res_num = self._response_num
        res_sep = self._response_seperator
        if res_num > 1:
            for i, cand_pred in enumerate(cand_preds):
                preds[i] = res_sep.join(cand_pred[0:res_num])

        return Output(preds, cand_preds, sorted_scores=sorted_scores)