Reinforcement Learning Project
========

[![Build Status](https://travis-ci.org/AaronYALai/Reinforcement_Learning_Project.svg?branch=master)](https://travis-ci.org/AaronYALai/Reinforcement_Learning_Project)
[![Coverage Status](https://coveralls.io/repos/github/AaronYALai/Reinforcement_Learning_Project/badge.svg?branch=master)](https://coveralls.io/github/AaronYALai/Reinforcement_Learning_Project?branch=master)

About
--------
Using **Q-learning**, a model-free reinforcement learning technique ([wiki](https://en.wikipedia.org/wiki/Q-learning)), to find optimal action-selection policy to play Gomoku (or Five-in-a-Row). Build two Gomoku agents playing against each other.

Some terms:
- Markov Decision Process(MDP): when making a decision to maximize future rewards, the information of the current state is just enough.
- Q function: given an action and the state, output the "value" of this state-action pair.
- Policy: a strategy to choose an action given the state, available actions and values of all state-action pairs.

**Deep Q-learning**: Train a deep neural network as our action-value Q function.


Details
--------
Policy used: epsilon greedy action selection policy

Greedy action:

- Since two agents facing against each other, the "greedy action" is the one balancing between
    - **suppressing the opponent's max Q value at the next step**
    - **promoting self's max Q value the next time you play** (Assume the opponent just reacts with the move with max Q value).

Experience replay:

- Update the neural network with **experience replay** which stores (X, y) in a fixed-length list (memory) and random sample a batch of (X, y) to update the network periodically.

When Playing:

- The agent will just choose the move with the largest state-action value (Q value).

Usage
--------
Clone the repo and use the [virtualenv](http://www.virtualenv.org/):

    git clone https://github.com/AaronYALai/Reinforcement_Learning_Project

    cd Reinforcement_Learning_Project

    virtualenv venv

    source venv/bin/activate

Install all dependencies and train agents:

    pip install -r requirements.txt

    python train_agent.py

    python agents_play.py

    python human_play.py


#### Reference: [**Q-learning with Neural Network**](http://outlace.com/Reinforcement-Learning-Part-3/)

