# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-08 01:55:41
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-10-08 12:39:13

from gomoku_game import initGame, makeMove, getReward, drawGrid, displayGrid
from agent_utils import load_agent

import numpy as np


def agent_play(agent1_name, agent2_name, width, win_reward=500,
               lose_reward=-1000, even_reward=-100,
               keepgoing_reward=-10):
    """Load two agents and let them play against each other"""
    agent1 = load_agent(agent1_name)
    agent2 = load_agent(agent2_name)
    play_game(agent1, agent2, width, win_reward, lose_reward,
              even_reward, keepgoing_reward)


def play_game(agent1, agent2, width, win_reward, lose_reward,
              even_reward, keepgoing_reward):
    """agents will take the move with the highest Q value"""
    state, available = initGame(width)
    agents = [agent1, agent2]

    # while game still in progress
    stop = False
    step = 0

    for k in range(width**2 + 5):

        for actor in range(2):
            step += 1
            qval = agents[actor].predict(state.reshape(1, 2 * width**2))

            # policy: choose the move with the max Q value
            action = (np.argmax(qval + available.reshape(1, width**2)))
            action = int(action / width), (action % width)
            print('Move #: %s; Actor %s, taking action: %s' %
                  (step, actor, action))

            state, available = makeMove(state, available, action, actor)
            displayGrid(drawGrid(state))
            reward = getReward(state, actor, win_reward, lose_reward,
                               even_reward, keepgoing_reward)

            if reward[actor] != -10:
                print("Reward: %s" % (reward,))
                stop = True
                break

        if (step > 350):
            print("Game lost; too many moves.")
            break

        if stop:
            break


def main():
    agent_play('agent1.pickle', 'agent2.pickle', 11)


if __name__ == '__main__':
    main()
