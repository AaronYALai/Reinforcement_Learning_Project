# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-07 15:03:47
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-10-07 15:26:05

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
    for i in range(15):
        for j in range(15):

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


def getReward(state, whose_turn):
    reward = [0, 0]
    if winGame(state[:, :, whose_turn]):
        reward[whose_turn] = 500
        reward[1 - whose_turn] = -1000
    elif fullGrid(state):
        reward = [-100, -100]
    else:
        reward[whose_turn] = -10
    return reward


def drawGrid(state):
    grid = np.zeros((19, 19), dtype='<U2')
    grid[:] = ' '
    for i in range(19):
        for j in range(19):
            if (state[(i, j)] > 0).any():
                if (state[(i, j)] == 1).all():
                    raise
                elif state[(i, j)][0] == 1:
                    grid[(i, j)] = 'O'
                else:
                    grid[(i, j)] = 'X'
    return grid


def displayGrid(grid):
    line = '\n' + '- + '*18 + '- {}\n'
    line = line.join([' | '.join(grid[i]) for i in range(19)])
    bottom = ('\n' + '  {} '*9).format(*[i+1 for i in range(9)])
    bottom += (' {} '*9).format(*[i+1 for i in range(9, 19)])
    print(line.format(*[i+1 for i in range(19)])+bottom)
