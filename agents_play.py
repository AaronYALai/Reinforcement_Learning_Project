# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-08 01:55:41
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-10-08 02:36:02

from gomoku_game import initGame, makeMove, getReward, drawGrid, displayGrid


def agent_play(agent1_name, agent2_name):
    agent1 = load_agent(agent1_name)
    agent2 = load_agent(agent2_name)
    testAlgo(agent1, agent2)


def testAlgo(agent1, agent2, width):
    state, available = initGame(width)
    agents = [agent1, agent2]

    # while game still in progress
    stop = False
    step = 0

    for k in range(width**2 + 5):

        for actor in range(2):
            step += 1
            qval = agent1.predict(state.reshape(1, 2 * width**2))

            # policy: choose the move with the max Q value
            action = (np.argmax(qval + available.reshape(1, width**2)))
            action = int(action / width), (action % width)
            print('Move #: %s; Actor %s, taking action: %s' % (step, actor, action))

            state, available = makeMove(state, available, action, actor)
            displayGrid(drawGrid(state))
            reward = getReward(state, actor)
            if reward[actor] != -10:
                print("Reward: %s" % (reward,))
                stop = True
                break

        if (step > 350):
            print("Game lost; too many moves.")
            break

        if stop:
            break


