#!/usr/bin/env python3
"""
Add weighted decoding and rerank_beam to TransformerGeneratorAgent
"""
import os
import pickle
import math
import typing as tp
import torch
import torch.nn.functional as F
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.torch_agent import Batch
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.utils.torch import neginf
from parlai.utils.io import PathManager


class WdgeneratorAgent(TransformerGeneratorAgent):
    @staticmethod
    def add_cmdline_args(argparser: ParlaiParser):
        """
        Add command-line arguments specifically for this agent.
        """
        TransformerGeneratorAgent.add_cmdline_args(argparser)
        argparser.add_argument(
            '--nidf',
            type='int',
            default=4,
            help=
            'nidf of weighted decoding , see https://arxiv.org/abs/1902.08654')

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self._make_nidf_feat_vec()

    def _make_nidf_feat_vec(self):
        """
        Construct the NIDF feature vector from self.dict.
        """
        print("Constructing NIDF feature vector...")
        model_file = self.opt['model_file']
        word2count_fp = model_file + '.word2count.pkl'
        with PathManager.open(word2count_fp, "rb") as f:
            data = pickle.load(f)
        word2count = data['word2count']
        min_c = min(word2count.values())  # max count
        max_c = max(word2count.values())  # min count
        word2nidf = {
            w: (math.log(max_c) - math.log(c)) /
            (math.log(max_c) - math.log(min_c))
            for w, c in word2count.items()
        }

        self._nidf_feats = torch.zeros((len(self.dict)))
        num_oovs = 0
        for idx in range(len(self.dict)):
            word = self.dict[idx]
            if word in word2nidf:
                # Leave emoji (these appear in Twitter dataset) as NIDF=0
                # (so we don't encourage emoji when we set WD weight high for NIDF)
                if word[0] == '@' and word[-1] == '@':
                    continue
                nidf = word2nidf[word]  # between 0 and 1
                self._nidf_feats[idx] = nidf
            else:
                # print("WARNING: word %s has no NIDF; marking it as NIDF=0" % word)
                num_oovs += 1  # If we don't have NIDF for this word, set as 0
        self._nidf_feats *= self.opt['nidf']
        print(
            'Done constructing NIDF feature vector; of %i words in dict there '
            'were %i words with unknown NIDF; they were marked as NIDF=0.' %
            (len(self.dict), num_oovs))

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: tp.Optional[torch.LongTensor] = None,
    ):
        """
        Generate an output with beam search.

        Depending on the options, this may perform greedy/topk/nucleus generation.

        :param Batch batch:
            Batch structure with input and labels
        :param int beam_size:
            Size of each beam during the search
        :param int max_ts:
            the maximum length of the decoded sequence
        :param prefix_tokens:
            if given, a tensor of tokens that must begin the decoded sequence.

        :return:
            tuple (beam_pred_scores, beams)

            - beam_preds_scores: list of (prediction, score) pairs for each sample in
              Batch
            - beams :list of Beam instances defined in Beam class, can be used for any
              following postprocessing, e.g. dot logging.
        """
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        encoder_states = model.encoder(*self._encoder_input(batch))
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = (
            len(batch.text_lengths) if batch.text_lengths is not None else len(
                batch.image)  # type: ignore
        )
        if batch.text_vec is not None:
            batchsize = batch.text_vec.size(0)
            beams = [
                self._treesearch_factory(dev).set_context(
                    self._get_context(batch, batch_idx)).set_block_list(
                        self.beam_block_list) for batch_idx in range(batchsize)
            ]
        else:
            beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(
            1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            score, incr_state = model.decoder(decoder_input, encoder_states,
                                              incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, -1:, :]
            score = model.output(score)
            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = F.log_softmax(score, dim=-1,
                                  dtype=torch.float32)  # type: ignore
            if prefix_tokens is not None and _ts < prefix_tokens.size(1):
                # generate prefix_tokens for every timestep that they exist
                # achieve by setting score of all other tokens to be -inf
                prefix_toks = prefix_tokens[:, _ts].unsqueeze(-1).repeat(
                    1, beam_size)
                prefix_score = score.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.NULL_IDX)
                score[prefix_mask] = neginf(score.dtype)
                score[prefix_mask] = score[prefix_mask].scatter_(
                    -1,
                    prefix_toks[prefix_mask].unsqueeze(-1),
                    prefix_score[prefix_mask],
                )
            for i, b in enumerate(beams):
                if not b.is_done():
                    score_in = score[i]
                    score_in += self._nidf_feats.to(dev)
                    b.advance(score_in)
            incr_state_inds = torch.cat([
                beam_size * i + b.get_backtrack_from_current_step()
                for i, b in enumerate(beams)
            ])
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds)
            selection = torch.cat([
                b.get_output_from_current_step() for b in beams
            ]).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(
                decoder_input, selection, incr_state_inds)

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]
        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(  # type: ignore
                batch, n_best_beam_preds_scores)

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [
            n_best_list[0] for n_best_list in n_best_beam_preds_scores
        ]
        return beam_preds_scores, beams