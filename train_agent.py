# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-08-02 21:42:08
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-10-08 13:37:13

import random
import numpy as np
from agent_utils import initAgent, trans_act, compute_Q, compute_label, \
                        check_exp, save_agent, load_agent
from gomoku_game import initGame, makeMove, getReward


# epsilon greedy, experience replay
def training(epoch, agent1, agent2, gamma=0.5, gamma2=0.95, epsilon=0.8,
             eps_threshold=0.05, buffersize=100, batchsize=40, print_every=20,
             verb=[0, 0], width=19, win_reward=500, lose_reward=-1000,
             even_reward=-100, keepgoing_reward=-10):
    """
    Train 2 deep Q-network agents with epsilon greedy implementation
    Use experience replay to avoid catastrophic forgetting
    Policy: balancing suppressing opponent(gamma) and optimize return(gamma2)
    """
    agents = [agent1, agent2]
    agent_exps = [[], []]
    running = [0, 0]

    for i in range(epoch):
        # initialize new game
        state, available = initGame(width)
        print("Game #: %s" % (i,))

        # start playing
        count = 0
        stop = False
        y_pre = np.zeros((width**2,))
        X_riv = state.reshape(2 * width**2,)

        # play the game
        while not stop:

            for actor, agent in enumerate(agents):
                count += 1

                # choose an action to take
                qval = agent.predict(state.reshape(1, 2 * width**2))

                # use epsilon greedy action selection
                if random.random() < epsilon:
                    x = np.random.randint(width)
                    y = np.random.randint(width)
                    action = (x, y)

                    while available[action] != 0:
                        x = np.random.randint(width)
                        y = np.random.randint(width)
                        action = (x, y)

                else:
                    index = np.argmax(qval + available.reshape((1, width**2)))
                    action = trans_act(index, width)

                # take the action and compute the reward of it
                new_state, new_available = makeMove(state, available,
                                                    action, actor)

                reward = getReward(new_state, actor, win_reward, lose_reward,
                                   even_reward, keepgoing_reward)

                # compute the target output value y of the agents
                maxQ, max_furtherQ = compute_Q(agents, actor, state,
                                               new_state, new_available, width)

                y, y_riv = compute_label(maxQ, max_furtherQ, actor, action,
                                         reward, gamma, gamma2, qval, y_pre,
                                         keepgoing_reward, width)

                X = state.reshape(2 * width**2,)

                # update with experience reply
                X_train, y_train = check_exp(agent_exps, actor, buffersize,
                                             X, y, running, batchsize)

                if len(X_train) != 0:
                    agent.fit(X_train, y_train, batch_size=batchsize,
                              nb_epoch=1, verbose=verb[0])

                # update the rival if necessary, i.e. game terminate
                if y_riv is not None:
                    exp = check_exp(agent_exps, 1 - actor, buffersize,
                                    X_riv, y_riv, running, batchsize)
                    Xriv_train, yriv_train = exp

                    if len(X_train) != 0:
                        agents[1 - actor].fit(Xriv_train, yriv_train,
                                              batch_size=batchsize,
                                              nb_epoch=1, verbose=verb[1])

                X_riv, y_pre = state.reshape(2 * width**2,), y
                state, available = new_state, new_available

                # check if the game terminate
                if reward[actor] != keepgoing_reward:
                    stop = True
                    break

                if count % print_every == 0:
                    print('\t step:', count)

        # decrease epsilon (prob of random action) every epoch
        if epsilon > eps_threshold:
            epsilon -= 2 / epoch

    return agent1, agent2


def train_agents(new=True, agent1_name=None, agent2_name=None, epoch=10,
                 agent1_save='agent1.pickle', agent2_save='agent2.pickle',
                 n_layer1=2, n_layer2=2, neurons1=1024, neurons2=1024,
                 gamma=0.5, gamma2=0.95, epsilon=0.8,
                 eps_threshold=0.05, buffersize=100, batchsize=40,
                 print_every=20, verb=[0, 0], width=19, win_reward=500,
                 lose_reward=-1000, even_reward=-100, keepgoing_reward=-10,
                 lr=1e-4, moment=0.9):
    """Create new agents or load existing agents then do training"""
    if new:
        agent1 = initAgent(neurons1, n_layer1, lr, moment, width)
        agent2 = initAgent(neurons2, n_layer2, lr, moment, width)
    else:
        agent1 = load_agent(agent1_name)
        agent2 = load_agent(agent2_name)

    agent1, agent2 = training(epoch, agent1, agent2, gamma, gamma2, epsilon,
                              eps_threshold, buffersize, batchsize,
                              print_every, verb, width, win_reward,
                              lose_reward, even_reward, keepgoing_reward)

    save_agent(agent1, agent1_save)
    save_agent(agent2, agent2_save)


def main():
    train_agents(False, 'agent1.pickle', 'agent2.pickle', width=11)


if __name__ == '__main__':
    main()
