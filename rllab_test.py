from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
import numpy as np
import tensorflow as tf
import gym
from RL_brain import PolicyGradient

env = GymEnv("MountainCar-v0")

policy = CategoricalMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32,32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    whole_paths=True,
    max_path_length=100000,
    n_itr=50,
    discount=0.99,
    step_size=0.01,
)

algo.train()
exit()
obs = env.reset()
tot = 0
for iter in range(2000):
    
    action, _ = policy.get_action(obs)
    next_obs, reward, done, info = env.step(action) 
    print(obs, action,reward,next_obs)
    tot += reward
    if done: break
    
    obs = next_obs

print(tot)