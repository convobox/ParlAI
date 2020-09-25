#!/usr/bin/env python3

import typing as tp
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.opt import Opt
from parlai.core.message import Message


class RetnrefAgent(TransformerGeneratorAgent):
    @staticmethod
    def add_cmdline_args(argparser):
        TransformerGeneratorAgent.add_cmdline_args(argparser)
        parser = argparser.add_argument_group(
            'RetNRef Transformer Generator Agent Args')
        parser.add_argument(
            '--use-knowledge',
            type='bool',
            default=False,
            help='Use knowledge to generate response.',
        )
        parser.add_argument(
            '--add-knowledge-to-history',
            type='bool',
            default=False,
            help='using knowledge as temp history if True.',
        )
        parser.add_argument(
            '--chosen-sentence',
            type='bool',
            default=True,
            help='instead of using all knowledge, use gold'
            'label, i.e. the chosen sentence',
        )

    def observe(self, observation: Message) -> Message:
        use_knowledge = self.opt.get('use_knowledge', False)
        add_knowledge_to_history = self.opt.get('add_knowledge_to_history',
                                                False)
        if use_knowledge and add_knowledge_to_history:
            if self.opt.get('chosen_sentence', True):
                add_text = observation.get('checked_sentence', None)
            else:
                add_text = observation.get('knowledge', None)
            if isinstance(add_text, str) and add_text != '':
                observation.force_set(
                    'text',
                    observation['text'] + self.history.delimiter + add_text)
        return super().observe(observation)

    def get_temp_history(self, observation: Message) -> tp.Optional[str]:
        use_knowledge = self.opt.get('use_knowledge', False)
        add_knowledge_to_history = self.opt.get('add_knowledge_to_history',
                                                False)
        if use_knowledge and not add_knowledge_to_history:
            if self.opt.get('chosen_sentence', True):
                return observation.get('checked_sentence', None)
            return observation.get('knowledge', None)
        return None