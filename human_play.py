# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-08 12:28:55
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-10-08 13:23:04

from gomoku_game import initGame, makeMove, getReward, drawGrid, displayGrid
from agent_utils import load_agent

import numpy as np


def combat_with_human(agent, width, turn, win_reward, lose_reward,
                      even_reward, keepgoing_reward):
    """Play a game with the agent"""
    state, available = initGame(width)

    # decide who moves first
    if turn == 0:
        human_actor, actor = 0, 1
    else:
        human_actor, actor = 1, 0

    step = 0
    terminate = False
    print('Game start!')

    for k in range(width**2 + 5):
        step += 1

        if turn == 0:
            # human plays
            flag = True

            while flag:
                action = input(">>> Input (ex. 1, 1): ")

                if action == 'end':
                    flag = False
                    terminate = True
                    print('Terminating the game...')
                    break

                # determine if the input is legal
                try:
                    action = tuple(map(int, action.split(',')))
                    flag = available[action] != 0
                except:
                    flag = True

                if flag:
                    print("can't put there")

            # human terminates the game
            if terminate:
                break

            state, available = makeMove(state, available, action, human_actor)
            reward = getReward(state, human_actor, win_reward, lose_reward,
                               even_reward, keepgoing_reward)[human_actor]
            turn = 1

        else:
            # agent plays
            qval = agent.predict(state.reshape(1, 2 * width**2))
            action = (np.argmax(qval + available.reshape(1, width**2)))
            action = int(action / width), (action % width)
            print('AI taking action: %s' % (action,))

            state, available = makeMove(state, available, action, actor)
            reward = getReward(state, actor, win_reward, lose_reward,
                               even_reward, keepgoing_reward)[actor]
            turn = 0

        # show the chessboard
        displayGrid(drawGrid(state))

        # check if the game proceeds
        if reward != keepgoing_reward:
            if reward > 0 and turn == 1:
                print('Human Wins!')
            else:
                print('AI Wins!')
            break

        if (step > width**2 - 9):
            print("Game lost; too many moves.")
            break


def combat(agent_name, width, turn, win_reward=500, lose_reward=-1000,
           even_reward=-100, keepgoing_reward=-10):
    """load the agent and play with human"""
    agent = load_agent(agent_name)
    combat_with_human(agent, width, turn, win_reward, lose_reward,
                      even_reward, keepgoing_reward)


def main():
    combat('agent1.pickle', 11, 0)


if __name__ == '__main__':
    main()
