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
from RL_brain import PolicyGradient
from tensorflow.examples.tutorials.mnist import input_data

env = GymEnv('MountainCarTNet-v0')
env.reset()
env._build_net()
real_env = GymEnv('MountainCar-v1')
real_env.reset()

env.reset()
mean = [0,0]
sqrmean = [0,0]
for iter in range(1000):
    x = np.random.random() * 1.8 - 1.2
    y = np.random.random() * 0.14 - 0.07
    a = np.random.randint(3)
    real_env.set_state(np.array([x,y]))
    o,r,d,i = real_env.step(a)
    na = (a-1)/2 + 1
    ny = (na-1)*0.001 + math.cos(3*x)*(-0.0025) + y
    ny = np.clip(ny, -0.07, 0.07)
    nx = x + ny
    nx = np.clip(nx, -1.2, 0.6)
    mean += (o-np.array([x,y]))/np.array([1.8,0.14])
    sqrmean += np.square((o-np.array([x,y]))/np.array([1.8,0.14]))
print('mean :' , mean / 1000, 'sqr mean:', np.sqrt(sqrmean / 1000))

if __name__ == '__main__':
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    
    for episode in range(5):
        policy = CategoricalMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(16, 16)
        )

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=100000,
            n_itr=50,
            discount=0.99,
            step_size=0.01,
            # sampler_args={
            #     "calc_dist":True
            # }
        )
        algo.train()

        traj_set = []
        env.add_traj(-1)
        for iter in range(200):
            obs = real_env.reset()
            tmp = []
            traj = []
            tmp.append(obs)
            for i in range(500):
                action,_ = policy.get_action(obs)
                obs_, _, done, _ = real_env.step(action)
                sample = [obs, action, obs_]
                traj.append(sample)
                tmp.append(action)
                tmp.append(obs_)
                if done : break
                obs = obs_
            if iter % 30 == 0:
                env.add_traj(tmp)
            traj_set.append(traj)

        test_traj_set = traj_set[0:50]
        traj_set = traj_set[50:]

        for traj in test_traj_set:
            real_obs = real_env.reset()
            obs = env.reset()
            tot = [0,0]
            mean =[0,0]
            cnt = 0
            for sample in traj:
                real_obs = sample[0]
                env.set_state(real_obs)
                real_env.set_state(real_obs)
                real_obs_,_,done,_ = real_env.step(sample[1])
                obs_, _,_,_ = env.step(sample[1])
                
                mean+= real_obs_ - obs_
                tot += np.square(real_obs_ - obs_) 
                cnt += 1
            print("Before , delta obs:", np.sqrt(tot / cnt), ' mean :', mean/cnt)

        env.reset()
        inp , outp = [] , []
        for traj in traj_set:
            for sample in traj:
                inp.append(np.concatenate((sample[0], np.array([sample[1]]))))
                outp.append(sample[2])

        for iter in range(60):
            loss = env.train(inp, outp)
            if iter % 10 == 0:
                print('episode :', episode, 'iter :', iter, '  loss :', loss)
        
        for traj in test_traj_set:
            real_obs = real_env.reset()
            obs = env.reset()
            tot = [0,0]
            mean =[0,0]
            cnt = 0
            for sample in traj:
                real_obs = sample[0]
                env.set_state(real_obs)
                real_env.set_state(real_obs)
                real_obs_,_,done,_ = real_env.step(sample[1])
                obs_, _,_,_ = env.step(sample[1])
                
                mean+= real_obs_ - obs_
                tot += np.square(real_obs_ - obs_) 
                cnt += 1
            print("After, delta obs:", np.sqrt(tot / cnt), ' mean :', mean/cnt)