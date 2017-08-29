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

np.random.seed(1)

env = GymEnv("MountainCarModified-v0")

baseline = LinearFeatureBaseline(env_spec=env.spec)
policy = CategoricalMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(32,)
)   
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100000,
    n_itr=1,
    discount=0.99,
    step_size=0.01,
)
algo.train()

ret = 0
done = False
obs = env.reset()
cnt = 0
while not done:
    action, _ = policy.get_action(obs)
    obs, r, done, _ = env.step(action)
    ret += r
    cnt += 1
    if cnt % 50 == 0:
        print('obs:', obs)
print(ret)