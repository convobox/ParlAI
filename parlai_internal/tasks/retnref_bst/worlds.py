#!/usr/bin/env python3

from parlai.tasks.blended_skill_talk.worlds import InteractiveWorld as InteractiveBaseWorld

class InteractiveWorld(InteractiveBaseWorld):
    @staticmethod
    def add_cmdline_args(argparser):
        InteractiveBaseWorld.add_cmdline_args(argparser)
        
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
