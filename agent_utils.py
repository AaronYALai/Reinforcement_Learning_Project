# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-07 16:21:50
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-10-07 17:30:59

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD
from keras.layers.advanced_activations import LeakyReLU

import pickle


def initAgent(neurons=512, alpha=0.1, layers=1, lr=1e-3,
              moment=0.9, width=19):
    """Initialize agent: specify num of neurons and hidden layers"""
    model = Sequential()
    model.add(Dense(2 * width**2, init='lecun_uniform',
              input_shape=(2 * width**2,)))
    model.add(LeakyReLU(alpha=alpha))

    for i in range(layers):
        model.add(Dense(neurons, init='lecun_uniform'))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(0.2))

    model.add(Dense(width**2, init='lecun_uniform'))
    # use linear output layer to generate real-valued outputs
    model.add(Activation('linear'))

    # opt = RMSprop(lr=lr)
    opt = SGD(lr=lr, momentum=moment, decay=1e-18, nesterov=False)
    model.compile(loss='mse', optimizer=opt)

    return model


def trans_act(num, width):
    """transform the integer back to the position on chessboard"""
    return int(num / width), (num % width)


def compute_Q(agents, actor, pre_state, new_state, new_available, width=19):
    """
    Compute Q values of each possible move.
    The policy: 1st - to suppress the opponent; 2nd - optimize self expected return
    """
    # suppress rival - choose the move that minimizes rival's max possible Q values
    rival_Q = agents[1 - actor].predict(new_state.reshape(1, 2 * width**2))
    newQ = new_available.reshape((1, width**2)) - rival_Q
    maxQ = np.max(newQ)

    # rival's reaction: assume the rival would choose the move with max Q values
    rival_action = np.argmax(rival_Q + new_available.reshape((1, width**2)))
    rival_action = int(rival_action / width), (rival_action % width)
    further_state, further_avai = makeMove(new_state, new_available, rival_action, 1 - actor)

    if further_state is None:
        raise

    # the agent's decision: the move with max "further" Q values
    further_Q = agents[actor].predict(further_state.reshape(1, 2 * width**2))
    max_furtherQ = np.max(further_Q + further_avai.reshape((1, width**2)))

    return maxQ, max_furtherQ


def compute_label(maxQ, max_furtherQ, actor, action, reward,
                  gamma, gamma2, qval, y_pre, keepgoing_reward, width=19):
    """compute the target output y for two agents' the deep Q-network"""
    if reward[actor] == keepgoing_reward:
        update = reward[actor] + (gamma * maxQ) + (gamma2 * max_furtherQ)
        y_riv = None
    else:
        y_riv = np.array(y_pre)
        y_riv[action[0] * width + action[1]] = reward[1 - actor]
        update = reward[actor]

    # calculate y
    y = np.array(qval)
    y[0][action[0] * width + action[1]] = update
    y = y.reshape(width**2,)

    return y, y_riv


def check_exp(agent_exps, actor, buffer, X, y, running, batchsize):
    X_train = []
    y_train = []
    if len(agent_exps[actor]) < buffer:
        agent_exps[actor].append((X, y))
    else:
        if running[actor] < (buffer - 1):
            running[actor] += 1
        else:
            running[actor] = 0
        agent_exps[actor][running[actor]] = (X, y)
        #randomly sample our experience replay memory
        minibatch = random.sample(agent_exps[actor], batchsize)
        
        for memX, memY in minibatch:
            X_train.append(memX)
            y_train.append(memY)
        X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train
