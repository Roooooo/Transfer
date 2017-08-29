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

sim_env = GymEnv('MountainCarTNet-v0')
real_env = GymEnv('MountainCar-v1')


class Net:
    def __init__(self,n_hidden = 100,learning_rate = 0.01):
        self.n_input = 2 + 3
        self.n_hidden = n_hidden
        self.n_output = 2
        self.lr = learning_rate
        self.alpha = 1
        self.beta = 0
        self._build_net()
        
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        self.inp = tf.placeholder(tf.float32, [None, 3], name = "inp")
        self.real_obs = tf.placeholder(tf.float32, [None , self.n_output], name="labels")

        layer_input = tf.layers.dense(
            inputs = self.inp,
            units = self.n_hidden,
            activation = tf.nn.sigmoid,
            kernel_initializer = tf.random_normal_initializer(mean=0,stddev=0.2),
            bias_initializer = tf.constant_initializer(0.1),
        )

        layer_hidden = tf.layers.dense(
            inputs = layer_input,
            units = self.n_hidden,
            activation = tf.nn.sigmoid,
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

        self.delta = (self.prediction - self.real_obs) / np.array([1.8, 0.14])

        self.preloss = tf.reduce_mean(tf.square(self.delta)) 

        self.loss = tf.reduce_mean(tf.square(self.delta)) * \
                    tf.abs(tf.reduce_mean(self.delta)) * self.beta

        self.pretrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.preloss)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def pretrain(self, inp, outp):
        batch_size = 20
        id = [x for x in range(len(inp))]
        np.random.shuffle(id)
        inp = [inp[x] for x in id]
        outp = [outp[x] for x in id]
        for i in range(int(len(inp)/batch_size)+1):
            if (i-1) * batch_size >= len(inp) : break
            batch_inp = inp[(i-1)*batch_size:i*batch_size-1]
            batch_outp = outp[(i-1)*batch_size:i*batch_size-1]
            self.sess.run(
                self.pretrain_op,
                feed_dict={
                    self.inp:batch_inp,
                    self.real_obs: batch_outp,
                }
            )
        loss = self.sess.run(
            self.preloss,
            feed_dict={
                self.inp:inp,
                self.real_obs:outp,
            }
        )
        return loss
        

    def train(self, inp, outp):
        batch_size = 5
        id = [x for x in range(len(inp))]
        np.random.shuffle(id)
        inp = [inp[x] for x in id]
        outp = [outp[x] for x in id]
        for i in range(int(len(inp)/batch_size)+1):
            if (i-1) * batch_size >= len(inp) : break
            batch_inp = inp[(i-1)*batch_size:i*batch_size-1]
            batch_outp = outp[(i-1)*batch_size:i*batch_size-1]
            self.sess.run(self.train_op,
                feed_dict={
                    self.inp:batch_inp,
                    self.real_obs: batch_outp,
                }
            )
        loss = self.sess.run(self.loss,
            feed_dict = {
                self.inp:inp,
                self.real_obs: outp,
            }
        )
        return loss
    
    def predict(self, obs, action):
        ret = self.sess.run(self.prediction,
            feed_dict={
                self.inp: np.atleast_2d(np.concatenate((obs,np.array([action])))),
            }
        )
        
        return ret[0]

def calc(x,y,a):
    ny = (a-1)*0.001 + math.cos(3*x)*(-0.0025) + y
    ny = np.clip(ny, -0.07, 0.07)
    nx = x + ny
    nx = np.clip(nx, -1.2, 0.6)
    return [nx, ny]

inp = []
outp = []
for x in np.arange(-1.2,0.6,0.01):
    for y in np.arange(-0.07,0.07,0.001):
        for a in range(3):
            inp.append(np.array([x,y,a]))
            ny = (a-1)*0.001 + math.cos(3*x)*(-0.0025) + y
            ny = np.clip(ny, -0.07, 0.07)
            nx = x + ny
            nx = np.clip(nx, -1.2, 0.6)
            outp.append(np.array([nx, ny]))


# pretrain
nn = Net(n_hidden = 25)
print(nn.n_hidden)
for iter in range(1000):
    loss = nn.pretrain(inp, outp)
    if iter%10 == 0: 
        print('iter :', iter, ' loss :', loss)
        tot = [0,0]
        for iter_ in range(200):
            x = np.random.random() * 1.8 - 1.2
            y = np.random.random() * 0.14 - 0.07
            a = np.random.randint(3)
            tot += np.square((nn.predict([x,y],a) - calc(x,y,a))/np.array([1.8, 0.14]))
        tot /= 200
        print(np.sqrt(tot))
        if loss < 1e-4: break

tot = [0,0]
for iter in range(200):
    x = np.random.random() * 1.8 - 1.2
    y = np.random.random() * 0.14 - 0.07
    a = np.random.randint(3)
    if iter < 5:
        print('pred  : ', nn.predict([x,y],a), 'true :', calc(x,y,a))
    tot += np.square((nn.predict([x,y],a) - calc(x,y,a))/np.array([1.8, 0.14]))
tot /= 200
print(np.sqrt(tot))

for iter in range(20):
    x = np.random.random() * 1.8 - 1.2
    y = np.random.random() * 0.14 - 0.07
    a = np.random.randint(3)
    tot += np.square(nn.predict([x,y],a) - calc(x,y,(a-1)/2+1))
tot /= 20
print('episode ', -1)
print('total loss :',np.sqrt(np.sqrt(tot)/np.array([1.8, 0.14])))

# train
for episode in range(20):
    real_env.reset()
    inp = []
    outp = []
    for iter in range(5000):
        x = np.random.random() * 1.8 - 1.2
        y = np.random.random() * 0.14 - 0.07
        a = np.random.randint(3)
        real_env.set_state(np.array([x,y]))
        obs, _ ,_ ,_ = real_env.step(a)
        inp.append(np.array([x,y,a]))
        outp.append(obs)

    for iter in range(75):
        loss = nn.train(inp, outp)
        if iter % 10 == 0:
            print('iter :', iter, ' loss :', loss)

    tot = [0,0]
    for iter in range(20):
        x = np.random.random() * 1.8 - 1.2
        y = np.random.random() * 0.14 - 0.07
        a = np.random.randint(3)
        tot += np.square(nn.predict([x,y],a) - calc(x,y,(a-1)/2+1))
    tot /= 20
    print('episode ', episode)
    print('total loss :',np.sqrt(np.sqrt(tot)/np.array([1.8, 0.14])))
