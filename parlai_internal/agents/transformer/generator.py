"""
Transformer Generator agent based on 
"""
import typing as tp
import torch
import torch.nn.functional as F
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.torch_agent import Batch
from parlai.utils.torch import neginf


class GeneratorAgent(TransformerGeneratorAgent):
    @staticmethod
    def add_cmdline_args(argparser: ParlaiParser):
        """
        Add command-line arguments specifically for this agent.
        """
        TransformerGeneratorAgent.add_cmdline_args(argparser)
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
        self._pad_tokens = [
            '__fp16_pad_0__',
            '__fp16_pad_1__',
            '__fp16_pad_2__',
            '__fp16_pad_3__',
        ]
        self._response_seperator = opt.get('response_sep', '')
        self._response_num = opt.get('response_num', 1)
        if self._response_num > 1 and len(self._response_seperator) == 0:
            raise ValueError(
                'Response seperator empty while response number over 1.')
        super().__init__(opt, shared)

    def build_dictionary(self):
        """
        Return the constructed dictionary, which will be set to self.dict.

        If you need to add additional tokens to the dictionary, this is likely the right
        place to do it.
        """
        d = self.dictionary_class()(self.opt)
        self.special_toks = self._get_special_tokens()
        if self.special_toks:
            d.add_additional_special_tokens(self.special_toks)

        if self.opt.get('person_tokens'):
            # try to keep dict length unchanged
            if len(self._pad_tokens) >= 2:
                for _ in range(2):
                    pop_pad_token = self._pad_tokens.pop(-1)
                    del d.freq[pop_pad_token]
                    idx = d.tok2ind.pop(pop_pad_token)
                    del d.ind2tok[idx]
            else:
                raise ValueError(
                    'Not enough padding tokens to add person tokens')
            d[self.P1_TOKEN] = 999_999_999
            d[self.P2_TOKEN] = 999_999_998
        rsep = self._response_seperator
        if rsep:
            if rsep in d.freq:
                raise ValueError(
                    'Response seperator conflict with existing dict')
            if len(self._pad_tokens) >= 1:
                pop_pad_token = self._pad_tokens.pop(-1)
                del d.freq[pop_pad_token]
                idx = d.tok2ind.pop(pop_pad_token)
                del d.ind2tok[idx]
                d.add_token(rsep)
            else:
                raise ValueError(
                    'Not enough padding tokens to add response seperator.')
        return d

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
                    b.advance(score[i])
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
        res_num = self.opt.get('response_num', 1)
        if res_num <= 1:
            beam_preds_scores = [
                n_best_list[0] for n_best_list in n_best_beam_preds_scores
            ]
        else:
            beam_preds_scores = []
            rsep_ind = self.dict.tok2ind[self._response_seperator]
            for n_best_list in n_best_beam_preds_scores:
                pred_list = []
                for index, pred_score in enumerate(n_best_list[0:res_num]):
                    if index == 0:
                        pred_list.append(pred_score[0])
                    else:
                        pred_list.append(
                            torch.Tensor([rsep_ind]).to(pred_score[0].device))
                        pred_list.append(pred_score[0])
                pred = torch.cat(pred_list)
                score = n_best_list[0][1]
                beam_preds_scores.append((pred, score))
        return beam_preds_scores, beams

    def _v2t(self, vec):
        """
        Convert token indices to string of tokens.
        """
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i != self.END_IDX and i != self.START_IDX:
                new_vec.append(i)
        return self.dict.vec2txt(new_vec)