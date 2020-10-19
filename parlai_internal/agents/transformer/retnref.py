#!/usr/bin/env python3
"""
RetnrefAgent based on TransformerGeneratorAgent.
"""
import typing as tp
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.message import Message
from parlai.core.params import ParlaiParser

class RetnrefAgent(TransformerGeneratorAgent):
    """
    Retrieve&Refine agent. Retrieval part chooses knowledge retrieval while
    refine part generally takes TransformerGeneratorAgent.
    """
    @staticmethod
    def add_cmdline_args(argparser: ParlaiParser):
        """
        Add command-line arguments specifically for this agent.
        """
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
        """
        Before general observe, if use_knowledge and add knowledge to history,
        knowledge will be added to agent's along with text.
        """
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
        """
        If use_knowledge and not add knowledge to history, knowledge will be
        set as temp history in agent's history and vectorize function will run
        accordingly.
        """
        use_knowledge = self.opt.get('use_knowledge', False)
        add_knowledge_to_history = self.opt.get('add_knowledge_to_history',
                                                False)
        if use_knowledge and not add_knowledge_to_history:
            if self.opt.get('chosen_sentence', True):
                return observation.get('checked_sentence', None)
            return observation.get('knowledge', None)
        return None
