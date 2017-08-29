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
import math
import time
from network import MLP

np.random.seed(123)

State_range = [1.8, 0.14]
State_min = [-1.2, -0.07]
State_max = [-0.6, 0.07]

def origin_transition(inputs):
    s,v,a = inputs
    ns = s
    nv = (a - 1) * 0.001 + math.cos(3 * s) * (-0.003)
    ns += nv

    return np.array([ns, nv])

def data_gen(data_len = 10000):
    X = np.random.rand(data_len, 2) * State_range + State_min
    X = np.column_stack((X, np.random.random_integers(0,2,[data_len,1])))
    Y = np.array([origin_transition(x) for x in X])
    return X, Y

def calc_MSE(Y, label):
    MSE = np.square((Y - label)/State_range)
    print((Y - label) / State_range)
    print('mse: ',np.mean(MSE, axis = 0))
    MSE = np.mean(np.mean(MSE, axis = 0))
    return MSE

t0 = time.time()

nn = MLP(
    n_features = 3,
    n_hidden = 128,
    n_labels = 2,
    #learning_rate = 0.005,
)

X, Y = data_gen(10000)
for (x,y) in zip(X,Y): 
    nn.store_data(x,y)
nn.learn(batch_size = 50, iters = 10000)

X, label = data_gen(5000)

Y = nn.predict(X)
print('error:', calc_MSE(Y, label))

print('Process time:', time.time() - t0)