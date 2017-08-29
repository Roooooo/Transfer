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

sim_env = GymEnv("MountainCarModified-v0")
real_env = GymEnv("MountainCar-v1")

REWARD_THRESHOLD = 0
MAX_PATH_LENGTH = 50000
MAX_SAMPLE_PATH_LENGTH = 500
NUM_SAMPLES = 250 + 50
TNET_TRAIN_EPOCH = 60

def test_in_real_env(policy, render = False):
    obs = real_env.reset()
    traj = []
    traj.append(obs)
    tot_reward = 0
    while True:
        if render : 
            real_env.render()
        action, _ = policy.get_action(obs)

        next_obs, reward, done, _ = real_env.step(action)
        traj.append(action)
        traj.append(next_obs)
        tot_reward += reward
        if done: return tot_reward, traj

        obs = next_obs

def sample_in_real_env(policy):
    traj = []
    obs = real_env.reset()
    traj.append(obs)
    for i in range(MAX_SAMPLE_PATH_LENGTH):
        action, _ = policy.get_action(obs)
        obs_, reward, done, _ = real_env.step(action)
        traj.append(action)
        traj.append(obs_)
        if done : break
        obs = obs_
    
    return traj   


def init_tnet():
    baseline = LinearFeatureBaseline(env_spec=real_env.spec)
    policy = CategoricalMLPPolicy(
        env_spec=real_env.spec,
        hidden_sizes=(32, 32)
    )
    algo = TRPO(
        env=real_env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100000,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
    )

    algo.train()
    for episode in range(1):
        traj_set = []
        for iter in range(NUM_SAMPLES):
            obs = real_env.reset()
            traj = []
            for i in range(200):
                action,_ = policy.get_action(obs)
                obs_, _, done, _ = real_env.step(action)
                sample = [obs, action, obs_]
                traj.append(sample)
                if done : break
                obs = obs_
            traj_set.append(traj)

        test_traj_set = traj_set[0:50]
        traj_set = traj_set[50:]

        sim_env.reset()
        inp , outp = [] , []
        for traj in traj_set:
            for sample in traj:
                inp.append(np.concatenate((sample[0], np.array([sample[1]]))))
                outp.append(sample[2])
                # print(sample, obs_, outp[-1])
            
        # print(traj_set[0][0:5])
        for iter in range(160):
            loss = sim_env.train(inp, outp)
            if iter % 10 == 0:
                print('episode :', episode, 'iter :', iter, '  loss :', loss)
        
        for traj in test_traj_set:
            real_obs = real_env.reset()
            obs = sim_env.reset()
            tot = [0,0]
            mean =[0,0]
            cnt = 0
            for sample in traj:
                real_obs = sample[0]
                sim_env.set_state(real_obs)
                real_env.set_state(real_obs)
                real_obs_,_,done,_ = real_env.step(sample[1])
                obs_, _,_,_ = sim_env.step(sample[1])
                
                mean+= real_obs_ - obs_
                tot += np.square(real_obs_ - obs_) 
                cnt += 1
            print("After, delta obs:", np.sqrt(tot / cnt), ' mean :', mean/cnt)

    obs = sim_env.reset()
    for iter in range(50):
        action,_ = policy.get_action(obs)
        obs_, _ , done , _ = sim_env.step(action)
        print(obs, action, obs_)
        if done: break
        obs = obs_
    
def calc_dis(traj1, traj2):
    cnt = 0
    mean = [0,0]
    while 2*cnt < len(traj1) and 2*cnt < len(traj2):
        mean += np.square(traj1[cnt*2] - traj2[cnt*2])
        cnt += 1
    return np.sqrt(mean / cnt)

file = open('outp.txt','w+')
begintime = time.asctime( time.localtime(time.time()) )

print('begin :', begintime)

sim_env._build_net()

baseline = LinearFeatureBaseline(env_spec=sim_env.spec)
# init_tnet()
for episode in range(5):
    # Train policy
    policy = CategoricalMLPPolicy(
        env_spec=sim_env.spec,
        hidden_sizes=(32,)
    )

    algo = TRPO(
        env=sim_env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=MAX_PATH_LENGTH,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
    )

    exp_algo = TRPO(
        env=sim_env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=MAX_PATH_LENGTH,
        n_itr=30,
        discount=0.99,
        step_size=0.01,
        sampler_args={
            "calc_dist":True
        }
    )

    algo.train()
    real_env.reset()
    # Test policy
    running_reward = 0

    traj_set = []
    for iter in range(15):
        reward , traj = test_in_real_env(policy)
        running_reward += reward
        traj_set.append(traj)
        sim_env.add_traj(traj[0:MAX_SAMPLE_PATH_LENGTH])

    # mean = [0,0]
    # for traj in traj_set:
    #     for traj_ in traj_set:
    #         mean += calc_dis(traj, traj_)
    # mean /= pow(len(traj_set),2)
    # print(mean)

    running_reward = running_reward / 15
    
    print('episode:' , episode, 'running_reward:', running_reward)    
    file.write('episode:' + str(episode) + 'running_reward:' + str(running_reward))
    if running_reward >= REWARD_THRESHOLD: break

    # Train exp_policy

    exp_algo.train()

    # Sampling
    traj_set = []set_state
    for iter in range(NUM_SAMPLES):
        traj_set.append(sample_in_real_env(policy))
        if iter % 50 == 0:
            print('LENGTH :', len(traj_set[-1]))
        sim_env.add_traj(traj_set[-1])

    X, Y = [], []
    sim_env.reset()
    
    for traj in traj_set:
        for iter in range(0,len(traj)-1,2):
            X.append(np.concatenate((traj[iter],np.array([traj[iter+1]]))))
            Y.append((traj[iter+2]))

    print("_____________episode " , episode)

    # Train residue
    for iter in range(TNET_TRAIN_EPOCH):
        loss = sim_env.train(X,Y)
        if iter % 10 == 0:
            print('episode : ', episode, ' iter : ', iter, ' loss : ',loss)
    
    for traj in traj_set:
        real_obs = real_env.reset()
        obs = sim_env.reset()
        tot = [0,0]
        mean =[0,0]
        cnt = 0
        for iter in range(0,len(traj)-1, 2):
            real_obs = traj[iter] 
            sim_env.set_state(real_obs)
            real_env.set_state(real_obs)
            real_obs_,_,done,_ = real_env.step(traj[iter+1])
            obs_, _,_,_ = sim_env.step(traj[iter+1])
            
            mean+= real_obs_ - obs_
            tot += np.square(real_obs_ - obs_) 
            cnt += 1
        print("After, delta obs:", np.sqrt(tot / cnt), ' mean :', mean/cnt)

    sim_env.env.env.traj_set = []

endtime = time.asctime( time.localtime(time.time()) )
print('begin : ', begintime, ' end :', endtime)
