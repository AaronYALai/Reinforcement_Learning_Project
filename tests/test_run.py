# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-08 13:31:46
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-10-08 14:35:31

from unittest import TestCase
from gomoku_game import try_display
from train_agent import train_agents
from agents_play import agent_play
from human_play import combat

import builtins
import mock


class test_running(TestCase):

    def test_game(self):
        try_display()

    def test_train_agent_and_play(self):
        train_agents(width=11, agent1_save='./test_agent1.pickle',
                     agent2_save='./test_agent2.pickle')

        # re-train
        train_agents(False, './test_agent1.pickle',
                     './test_agent2.pickle', width=11)

        agent_play('./test_agent1.pickle', './test_agent2.pickle', 11)

    def test_human(self):
        replies = (x for x in ['1,1', '4,4', 'end'])
        with mock.patch.object(builtins, 'input', lambda x: next(replies)):
            combat('./agent1.pickle', 11, 0)
