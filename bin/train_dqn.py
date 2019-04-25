#!/usr/bin/env python

import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
import random
import math
import csv
from numpy.linalg import inv

from envs.ea import de_R1, de_R2, de_R3, de_validate

import gym
from gym import spaces
from gym.utils import seeding

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from callbacks import Callback
from rl.agents.dqn import DQNAgent

from rl.memory import SequentialMemory
from rl.util import *

import argparse

class ModelCheckpoint(Callback):
    def __init__(self, filepath, interval, verbose=1):
        super(ModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.total_episodes = 0
        self.best_reward = -np.inf
        self.total_reward = 0

    def on_episode_end(self, episode, logs):
        """ Save weights at interval steps during training if
        reward improves """
        self.total_episodes += 1
        self.total_reward += logs['episode_reward']
        if self.total_episodes % self.interval != 0:
            # Nothing to do.
            return
            
        if self.total_reward > self.best_reward:
            if self.verbose > 0:
                print('Episode {}: Reward improved '
                        'from {} to {}'.format(self.total_episodes,
                        self.best_reward, self.total_reward))
            self.model.save_weights(self.filepath, overwrite=True)
            self.best_reward = self.total_reward
            self.total_reward = 0

class PolicyDebug(object):
    """Abstract base class for all implemented policies.

        Each policy helps with selection of action to take on an environment.

        Do not use this abstract base class directly but instead use one of the concrete policies implemented.
        To implement your own policy, you have to implement the following methods:

        - `select_action`

        # Arguments
        agent (rl.core.Agent): Agent used
    """
    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        """Return configuration of the policy

            # Returns
            Configuration as dict
        """
        return {}

class BoltzmannQPolicy(PolicyDebug):
    """Implement the Boltzmann Q Policy

        Boltzmann Q Policy builds a probability law on q values and returns
        an action selected randomly according to this law.
    """
    def __init__(self, tau=1., clip=(-500., 500.)):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action

            # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

            # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        """Return configurations of BoltzmannQPolicy

            # Returns
            Dict of config
        """
        config = super(BoltzmannQPolicy, self).get_config()
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config


class EpsGreedyQPolicy(PolicyDebug):
    """Implement the epsilon greedy policy

        Eps Greedy policy either:

        - takes a random action with probability epsilon
        - takes current best action with prob (1 - epsilon)
        """
    def __init__(self, eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        """Return the selected action

            # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

            # Returns
            Selection action
            """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions-1)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy

            # Returns
            Dict of config
            """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


                    ############################################### Parameters ##############################################################

parser = argparse.ArgumentParser(description = "de and ddqn parameters")
parser.add_argument('--instance', type=int, help='Validation instance selected')
parser.add_argument('--FF', type=float, default=0.5,  help='scaling factor for DE')
parser.add_argument('--NP', type=int, default=100,  help='population size for DE')
parser.add_argument('--CR', type=float, default=0.5,  help='crossover rate for DE')
parser.add_argument('--FE', type=int, default=100,  help='function evaluations')
parser.add_argument('--max_gen', type=int, default=10,  help='maximum nuber of genertion')
parser.add_argument('--W', type=int, default=50,  help='window size')
parser.add_argument('--unit', type=int, default=100,  help='number of nodes in each hidden layer')
parser.add_argument('--batchsize', type=int, default=4,  help='batch size of neural network')
parser.add_argument('--gamma', type=float, default=0.99,  help='discount factor')
parser.add_argument('--C', type=int, default=1000,  help='target network synchronised')
parser.add_argument('--memory', type=int, default=100000,  help='memory size')
parser.add_argument('--warmup', type=int, default=10000,  help='number of evaluations used for warm up')
parser.add_argument('--LR', type=float, default=0.0001,  help='Adam learning rate')
parser.add_argument('--training_steps', type=int, default=100000000,  help='Steps required for each training')

args = parser.parse_args()
instance = args.instance
FF = args.FF#; print(FF)
NP = args.NP
CR = args.CR
FE = args.FE
max_gen = args.max_gen
W = args.W
unit = args.unit
batchsize = args.batchsize
gamma = args.gamma
C = args.C
memory = args.memory
warmup = args.warmup
LR = args.LR
training_steps = args.training_steps

ENV_NAME = 'ea'

                            ################################################# Training phase ##############################################################
env_train = de_R2.DEEnv(FF, NP, CR, FE, max_gen, W) # Can be changed to create an object of de-R1 or de-R3 for reward defintions R1 and R3 resp.

nb_actions = env_train.action_space.n

# Build a sequential model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env_train.observation_space.shape))
model.add(Dense(unit, activation = 'relu'))
model.add(Dense(unit, activation = 'relu'))
model.add(Dense(unit, activation = 'relu'))
model.add(Dense(unit, activation = 'relu'))
model.add(Dense(nb_actions, activation = 'linear'))
print("Model Summary: ",model.summary())

memory = SequentialMemory(limit=memory, window_length=1)

# Boltzmann Q Policy
policy = EpsGreedyQPolicy()

# DQN Agent: Finally, we configure and compile our agent. You can use every built-in Keras optimizer and even the metrics!
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=warmup, target_model_update=C, policy=policy, enable_double_dqn=True, batch_size=batchsize, gamma=gamma) # nb_steps_warmup >= nb_steps 2000
# DQN stores the experience in the memory buffer for the first nb_steps_warmup. This is done to get the required size of batch during experience replay.
# When number of steps exceeds nb_steps_warmup then the neural network would learn and update the weight.

# Neural Compilation
dqn.compile(Adam(lr=LR), metrics=['mae'])

callbacks = [ModelCheckpoint('dqn_ea_weights.h5f', 32)]
print("above fit")
# Fit the model: training for nb_steps = number of generations
dqn.fit(env_train, callbacks = callbacks, nb_steps=training_steps, visualize=False, verbose=0, nb_max_episode_steps = None)

                            ##################################################### Validation phase ##############################################################

env_validate = de_validate.DEEnv(instance, FF, NP, CR, FE, max_gen, W) # Can be changed to create an object of de-R1 or de-R3 for reward defintions R1 and R3 resp.

nb_actions = env_validate.action_space.n

# Build a sequential model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env_validate.observation_space.shape))
model.add(Dense(unit, activation = 'relu'))
model.add(Dense(unit, activation = 'relu'))
model.add(Dense(unit, activation = 'relu'))
model.add(Dense(unit, activation = 'relu'))
model.add(Dense(nb_actions, activation = 'linear'))

dqn.test(env_validate, nb_episodes=1, visualize = False)
print(env_validate.best_so_far)
#return env_validate.best_so_far



















