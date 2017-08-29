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

MAX_EPISODE = 1
MAX_TEST_TIMES = 5
MAX_TRAJ_LENGTH = 500
TARGET_REWARD = -50

np.random.seed(1)
sim_env = GymEnv("MountainCarModified-v0")
real_env = GymEnv("MountainCar-v0")


# sim_env.reset()
# while True:
#     p,v,a = input('p,v,a').split()
#     sim_env.env.state = (float(p),float(v))
#     s,r,d,_  = sim_env.step(int(a))
#     print(s)
# obs = sim_env.reset()
# done = False
# while not done:
#     print(obs)
#     if(obs[0] <= -1.2): break
#     obs, r, done, _ = sim_env.step(2)

def train_act_policy():
    baseline = LinearFeatureBaseline(env_spec=sim_env.spec)
    policy = CategoricalMLPPolicy(
        env_spec=sim_env.spec,
        hidden_sizes=(32,)
    )   
    algo = TRPO(
        env=sim_env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100000,
        n_itr=1,
        discount=0.99,
        step_size=0.01,
    )
    algo.train()
    return policy

def train_exp_policy():
    baseline = LinearFeatureBaseline(env_spec=sim_env.spec)
    policy = CategoricalMLPPolicy(
        env_spec=sim_env.spec,
        hidden_sizes=(32,)
    )   
    algo = TRPO(
        env=sim_env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100000,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
    )
    algo.train()
    return policy

def test_act_policy(policy):
    ret = 0
    trajs = []
    for i in range(MAX_TEST_TIMES):
        step_cnt = 0
        done = False
        obs = real_env.reset()
        trajs.append([[obs],[]])
        while not done:
            action, _ = policy.get_action(obs)
            obs, r, done, _ = real_env.step(action)
            step_cnt += 1
            trajs[-1][0].append(obs)
            trajs[-1][1].append(np.array([action]))
        ret += step_cnt / float(MAX_TEST_TIMES)
    return -ret, trajs

def modify_trans_model(trajs):
    X, Y = [], []
    for traj in trajs:
        for i in range(0,len(traj[1])):
            X.append(np.concatenate((traj[0][i], traj[1][i])))
            Y.append(traj[0][i+1])
    for (x,y) in zip(X,Y): sim_env.env.trans_net.store_data(x,y)
    print(sim_env.env.trans_net.ep_obs.shape)
    print(sim_env.env.trans_net.ep_ls.shape)
    sim_env.env.trans_net.learn(iters = 10000)

def sample(policy):
    trajs = []

    for i in range(MAX_SAMPLE_TIMES):
        step_cnt = 0
        done = False
        obs = real_env.reset()
        trajs.append([[obs],[]])
        while not done:
            action, _ = policy.get_action(obs)
            obs, r, done, _ = real_env.step(action)
            step_cnt += 1
            trajs[-1][0].append(obs)
            trajs[-1][1].append(action)
            if(step_cnt >= MAX_TRAJ_LENGTH): break

    return trajs

for episode in range(MAX_EPISODE):
    
    print('episode', episode,':')

    act_policy = train_act_policy()

    ret, trajs = test_act_policy(act_policy)

    if ret > TARGET_REWARD:
        break
    
    modify_trans_model(trajs)

    exp_policy = train_exp_policy()

    exp_trajs = sample(exp_policy)

    modify_trans_model(exp_trajs)
# Train best policy
# Test best policy
# Train explore policy
# Sample
# Modify model