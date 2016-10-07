# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-07 15:03:47
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-10-07 15:58:20

import numpy as np


def initGame(width=19):
    """Initialize width x width new game"""
    state = np.zeros((width, width, 2))
    available = np.zeros((width, width))

    return state, available


def makeMove(state, available, action, actor):
    """specify the actor and the location of the new stone"""
    available_ret = np.zeros(available.shape)
    available_ret[:] = available[:]

    if available_ret[action] == 0:
        state[action][actor] = 1
        available_ret[action] = float("-inf")
        return state, available_ret
    else:
        return None, available_ret


def winGame(sub_state):
    """check if the game winning criteria is met"""
    for i in range(sub_state.shape[0] - 4):
        for j in range(sub_state.shape[1] - 4):

            horizontal = sub_state[i][j: j+5]
            if (horizontal == 1).all():
                return True

            vertical = [sub_state[i+k, j] for k in range(5)]
            if (np.array(vertical) == 1).all():
                return True

            diagonal = [sub_state[(i+k, j+k)] for k in range(5)]
            if (np.array(diagonal) == 1).all():
                return True

    return False


def fullGrid(state):
    """check if the chessboard is full"""
    return not ((state[:, :, 0] + state[:, :, 1]) == 0).any()


def getReward(state, whose_turn, win_reward=500, lose_reward=-1000,
              even_reward=-100, keepgoing_reward=-10):
    """calculate the reward given to whom just moved"""
    reward = [0, 0]

    if winGame(state[:, :, whose_turn]):
        reward[whose_turn] = win_reward
        reward[1 - whose_turn] = lose_reward

    elif fullGrid(state):
        reward = [even_reward, even_reward]

    else:
        reward[whose_turn] = keepgoing_reward

    return reward


def drawGrid(state):
    """visualize the chessboard"""
    grid = np.zeros(state.shape[:2], dtype='<U2')
    grid[:] = ' '

    for i in range(state.shape[0]):
        for j in range(state.shape[1]):

            if (state[(i, j)] > 0).any():

                if (state[(i, j)] == 1).all():
                    raise

                elif state[(i, j)][0] == 1:
                    grid[(i, j)] = 'O'

                else:
                    grid[(i, j)] = 'X'

    return grid


def displayGrid(grid):
    """print out the chessboard"""
    wid = grid.shape[0]
    show_num = 9 if wid > 9 else wid

    # chessboard
    line = '\n' + '- + ' * (wid - 1) + '- {}\n'
    line = line.join([' | '.join(grid[i]) for i in range(wid)])

    # mark the number of its lines
    bottom = ('\n' + '  {} ' * show_num)
    bottom = bottom.format(*[i+1 for i in range(show_num)])

    if show_num == 9:
        part = (' {} '*(wid - show_num))
        part = part.format(*[i+1 for i in range(show_num, wid)])
        bottom += part

    print(line.format(*[i+1 for i in range(wid)]) + bottom)
