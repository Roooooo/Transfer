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

class Net:
    def __init__(self,n_hidden = 50,learning_rate = 0.01):
        self.n_input = 2 + 3
        self.n_hidden = n_hidden
        self.n_output = 2
        self.lr = learning_rate
        self.alpha = 10
        self.beta = 0.5
        self._build_net()
        
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        self.sim_s = tf.placeholder(tf.float32, [None, 1], name="input_state")
        self.sim_v = tf.placeholder(tf.float32, [None, 1], name="input_velocity")
        self.sim_act = tf.placeholder(tf.float32, [None, 1], name="acts")
        self.real_obs = tf.placeholder(tf.float32, [None , self.n_output], name="labels")

        self.sim_v += (self.sim_act-1)*0.001 + tf.cos(3*self.sim_s)*(-0.0025)
        self.sim_v = tf.clip_by_value(self.sim_v, -0.07, 0.07)
        self.sim_s += self.sim_v
        self.sim_s = tf.clip_by_value(self.sim_s, -1.2, 0.6)

        self.layer_inp = tf.concat((self.sim_s, self.sim_v, self.sim_act),1)

        layer_input = tf.layers.dense(
            inputs = self.layer_inp,
            units = self.n_hidden,
            activation = tf.nn.tanh,
            kernel_initializer = tf.random_normal_initializer(mean=0,stddev=0.2),
            bias_initializer = tf.constant_initializer(0.1),
        )

        layer_hidden = tf.layers.dense(
            inputs = layer_input,
            units = self.n_hidden,
            activation = tf.nn.tanh,
            kernel_initializer = tf.random_normal_initializer(mean=0,stddev=0.2),
            bias_initializer = tf.constant_initializer(0.1),
        )

        layer_out = tf.layers.dense( 
            inputs = layer_hidden,
            units = self.n_output,
            activation = None,
            kernel_initializer = tf.random_normal_initializer(mean=0,stddev=0.2),
            bias_initializer = tf.constant_initializer(0.1),
        )

        self.prediction = layer_out

 #       self.loss = tf.reduce_mean(tf.square(self.prediction - self.real_obs)) * self.alpha + tf.reduce_mean(tf.square(self.prediction))
        self.delta = (self.prediction - self.real_obs) / np.array([1.8, 0.14])

        self.loss = tf.reduce_mean(tf.square(self.delta)) * self.alpha + \
                    tf.abs(tf.reduce_mean(self.delta)) * self.beta + \
                    tf.reduce_mean(tf.square(self.prediction))

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, inp, outp):
        batch_size = 250
        id = [x for x in range(len(inp))]
        np.random.shuffle(id)
        inp = [inp[x] for x in id]
        outp = [outp[x] for x in id]

        s = [x[0] for x in inp]
        v = [x[1] for x in inp]
        act = [x[2] for x in inp]

        for i in range(int(len(inp)/batch_size)+1):
            if (i-1) * batch_size >= len(inp) : break
            batch_s = s[(i-1)*batch_size:i*batch_size-1]
            batch_v = v[(i-1)*batch_size:i*batch_size-1]
            batch_act = act[(i-1)*batch_size:i*batch_size-1]
            batch_outp = outp[(i-1)*batch_size:i*batch_size-1]
            self.sess.run(self.train_op,
                feed_dict={
                    self.sim_s: np.vstack(batch_s),
                    self.sim_v: np.vstack(batch_v),
                    self.sim_act: np.vstack(batch_act),
                    self.real_obs: batch_outp,
                }
            )
        loss = self.sess.run(self.loss,
            feed_dict = {
                self.sim_s: np.vstack(s),
                self.sim_v: np.vstack(v),
                self.sim_act: np.vstack(act),
                self.real_obs: outp,
            }
        )
        return loss
    
    def predict(self, obs, action):
        ret = self.sess.run(self.prediction,
            feed_dict={
                self.sim_s: np.vstack([obs[0]]),
                self.sim_v: np.vstack([obs[1]]),
                self.sim_act: np.vstack([action]),
            }
        )
        
        return ret[0]

if __name__ == '__main__':
    env = GymEnv('MountainCarTNet-v0')
    env.reset()
    env._build_net()
    real_env = GymEnv('MountainCar-v1')
    real_env.reset()
    baseline = LinearFeatureBaseline(env_spec=real_env.spec)
    nn = Net()  
    policy = CategoricalMLPPolicy(
        env_spec=real_env.spec,
        hidden_sizes=(16, 16)
    )
    algo = TRPO(
        env=real_env,
        policy=policy,
        baseline=baseline,
        batch_size=400,
        max_path_length=100000,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
    )
    algo.train()
    traj_set = []
    for iter in range(150):
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

    print(len(traj_set[0]))


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

    inp , outp = [] , []
    print(traj_set[0][0:5])
    for traj in traj_set:
        for sample in traj:
            inp.append(np.concatenate((sample[0], np.array([sample[1]]))))
            outp.append(sample[2])

    for iter in range(60):
        loss = nn.train(inp, outp)
        if iter % 10 == 0:
            print('iter :', iter, '  loss :', loss)
    
    for traj in test_traj_set:
        real_obs = real_env.reset()
        tot = [0,0]
        mean =[0,0]
        cnt = 0
        for sample in traj:
            real_obs = sample[0]
            real_env.set_state(real_obs)
            real_obs_,_,_,_ = real_env.step(sample[1])
            obs_ = nn.predict(real_obs, sample[1])
            
            mean+= real_obs_ - obs_
            tot += np.square(real_obs_ - obs_) 
            cnt += 1
        print("After, delta obs:", np.sqrt(tot / cnt), ' mean :', mean/cnt)