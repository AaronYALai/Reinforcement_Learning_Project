# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-08 13:31:46
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-10-08 15:29:51

from unittest import TestCase
from gomoku_game import try_display
from train_agent import train_agents
from agents_play import agent_play
from human_play import combat

import builtins
import mock
import os


class test_running(TestCase):
    ag1 = './test_agent1.pickle'
    ag2 = './test_agent2.pickle'

    def test_game(self):
        try_display()

    def test_train_agent_and_play(self):
        train_agents(width=9, agent1_save=self.ag1, agent2_save=self.ag2,
                     n_layer1=1, n_layer2=1, neurons1=128, neurons2=128)

        # re-train
        train_agents(False, self.ag1, self.ag2, width=9)

        agent_play(self.ag1, self.ag2, 9)

        os.remove(self.ag1)
        os.remove(self.ag2)

    def test_human(self):
        replies = (x for x in ['1,1', '4,4', 'end'])
        with mock.patch.object(builtins, 'input', lambda x: next(replies)):
            combat('./agent1.pickle', 11, 0)
