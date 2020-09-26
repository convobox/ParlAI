#!/usr/bin/env python3
"""
Interactive world for Retrieve&Refine with knowledge retriever and Blended skill talk.
"""
from copy import deepcopy
import typing as tp
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.message import Message
from parlai.core.worlds import validate
from parlai.tasks.blended_skill_talk.worlds import InteractiveWorld as InteractiveBaseWorld
from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import (
    KnowledgeRetrieverAgent, )


class InteractiveWorld(InteractiveBaseWorld):
    """
    InteractiveWorld combined with Blended skill talk InteractiveWorld
    and WoW knowledge retrieval InteractiveWorld.
    """
    @staticmethod
    def add_cmdline_args(argparser: ParlaiParser) -> None:
        """
        Add command-line arguments specifically for this task world.
        """
        InteractiveBaseWorld.add_cmdline_args(argparser)
        parser = argparser.add_argument_group('RetNRef Interactive World Args')
        parser.add_argument(
            '--print-checked-sentence',
            type='bool',
            default=True,
            help='Print sentence that the model checks.',
        )
        parser.add_argument(
            '--add-token-knowledge',
            type='bool',
            default=False,
            help='Add knowledge token to retrieved knowledge',
        )

    def __init__(self, opt: Opt, agents: tp.List[tp.Any], shared=None) -> None:
        super().__init__(opt, agents, shared)
        self._set_up_knowledge_agent(opt.get('add_token_knowledge', False))
        self.print_checked_sentence = opt['print_checked_sentence']

    def _set_up_knowledge_agent(self,
                                add_token_knowledge: bool = False) -> None:
        """
        Set up knowledge agent for knowledge retrieval generated from WoW project.
        """
        parser = ParlaiParser(False, False)
        KnowledgeRetrieverAgent.add_cmdline_args(parser)
        parser.set_params(
            model='projects:wizard_of_wikipedia:knowledge_retriever',
            add_token_knowledge=add_token_knowledge,
        )
        knowledge_opt = parser.parse_args([])
        self.knowledge_agent = KnowledgeRetrieverAgent(knowledge_opt)

    def _add_knowledge_to_act(self, act: Message) -> Message:
        """
        After human agent act, if use_knowledge is True, add knowledge to act.
        Knowledge agent first observes human agent's act, then acts itself.
        Key 'knowledge' represents full knowledge consisting of multi knowledge sentences.
        Key 'checked_sentence' represents gold result among full knowledge.
        """
        if self.opt.get('use_knowledge', False):
            self.knowledge_agent.observe(act, actor_id='apprentice')
            knowledge_act = self.knowledge_agent.act()
            act['knowledge'] = knowledge_act['text']
            act['checked_sentence'] = knowledge_act['checked_sentence']
            if self.print_checked_sentence:
                print('[ Using chosen sentence from Wikpedia ]: {}'.format(
                    knowledge_act['checked_sentence']))
            act['title'] = knowledge_act['title']
        return act

    def parley(self) -> None:
        # random initialize human and model persona
        if self.turn_cnt == 0:
            self.p1, self.p2 = self.get_contexts()

        if self.turn_cnt == 0 and self.p1 != '':
            # add the context on to the first message to human
            context_act = Message({
                'id': 'context',
                'text': self.p1,
                'episode_done': False
            })
            # human agent observes his/her persona
            self.agents[0].observe(validate(context_act))
        try:
            # human agent act first
            act = deepcopy(self.agents[0].act())
        except StopIteration:
            self.reset()
            self.finalize_episode()
            self.turn_cnt = 0
            return
        self.acts[0] = act
        if self.turn_cnt == 0 and self.p2 != '':
            # add the context on to the first message to agent 1
            context_act = Message({
                'id': 'context',
                'text': self.p2,
                'episode_done': False
            })
            # model observe its persona
            self.agents[1].observe(validate(context_act))

        # add knowledge to the model observation
        act = self._add_knowledge_to_act(act)

        # model observe human act and knowledge
        self.agents[1].observe(validate(act))
        # model agent act
        self.acts[1] = self.agents[1].act()

        # add the mdoel reply to the knowledge retriever's dialogue history
        self.knowledge_agent.observe(validate(self.acts[1]))

        # human agent observes model act
        self.agents[0].observe(validate(self.acts[1]))
        self.update_counters()
        self.turn_cnt += 1

        if act['episode_done']:
            self.finalize_episode()
            self.turn_cnt = 0
