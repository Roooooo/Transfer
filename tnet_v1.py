from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
import numpy as np
import tensorflow as tf
import gym
import pickle

import time
import math

def func(s, a):
    x,y = s
    ny = (a-1) * 0.001 + math.cos(3*x) * 0.0025 + y
    ny = np.clip(ny, -0.07, 0.07)
    nx = x + ny
    nx = np.clip(nx, -1.2, 0.6)
    return [nx, ny]

class Net:
    def __init__(layer_size, learning_rate = 0.09):
        self.lr = learning_rate
        self.layer_size = layer_size

        self.sample, self.label = [] , []

        self._build_net()

    def _build_net():
        pass

    def store_data(state, action, newState):
        self.sample.append(np.concatenate((state, action)))
        self.label.append(newState)

    def train():
        pass

    def predict(state, action):
        pass

NUM_SAMPLES = 50000
NUM_TEST_SAMPLES = 5000
UP_BOUND = [0.6, 0.07]
LOW_BOUND = [-1.2, -0.07]
EPS = 1e-6

tnet = Net()

for _ in xrange(NUM_SAMPLES):
    s = np.random.random(2) * (UP_BOUND - LOW_BOUND) + LOW_BOUND
    a = np.random.randint(3)
    store_data(s,a,func(s,a))

tnet.train()

acc = 0
for _ in xrange(NUM_TEST_SAMPLES):
    s = np.random.random(2) * (UP_BOUND - LOW_BOUND) + LOW_BOUND
    a = np.random.randint(3)
    ns = tnet.predict(s,a)
    if (ns - s) / (UP_BOUND - LOW_BOUND) < EPS:
        acc += 1

print('acc rate:' , acc / NUM_TEST_SAMPLES)