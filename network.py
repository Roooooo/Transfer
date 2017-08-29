"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
class MLP():
    def __init__(
        self,
        n_features,
        n_labels,
        n_hidden=32,
        learning_rate=0.01,
        activation=tf.nn.tanh,
    ):
        self.n_labels = n_labels
        self.n_features = n_features
        self.lr = learning_rate

        self.n_hidden_units = n_hidden

        self.activation = activation

        self._build_net()

        self.ep_obs, self.ep_ls = [], []

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def addlayer(self, inputs, n_in, n_out, activation=None):
        W = tf.Variable(tf.random_normal([n_in,n_out]))
        b = tf.Variable(tf.zeros([1,n_out])+0.1)
        out = tf.matmul(inputs, W) + b

        if activation != None:
            out = activation(out)
        return out

    def _build_net(self):
        self.obs = tf.placeholder(tf.float32, [None,self.n_features], name="observations")
        self.labels = tf.placeholder(tf.float32, [None,self.n_labels], name="labels")

        layer = self.addlayer(self.obs, self.n_features, self.n_hidden_units, activation=tf.nn.tanh)
        layer = self.addlayer(layer, self.n_hidden_units, self.n_hidden_units, activation=tf.nn.tanh)
        layer = self.addlayer(layer, self.n_hidden_units, self.n_labels)

        self.prediction = layer
        loss = tf.reduce_mean(tf.reduce_sum(tf.square((self.prediction - self.labels)/[18, 1.4]), reduction_indices=[1]))
        self.loss = loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def learn(self, batch_size = 1,iters=1000):
        batch_num = int((len(self.ep_obs)+batch_size-1)/batch_size)
        for iter in range(iters):
            batch_index = [i for i in range(batch_num)]
            np.random.shuffle(batch_index)
            for i in range(batch_num):
                self.sess.run(self.train_op,
                    feed_dict = {
                        self.obs: np.vstack(self.ep_obs[batch_index[i] * batch_size:(batch_index[i]+1)*batch_size]),
                        self.labels: np.vstack(self.ep_ls[batch_index[i] * batch_size:(batch_index[i]+1)*batch_size]),
                    }
                )
            if iter % 100 == 0:
                print('iters:', iter, 'loss:', self.sess.run(self.loss, feed_dict={
                    self.obs: self.ep_obs,
                    self.labels: self.ep_ls,
                }))
        self.ep_obs , self.ep_as = [] , []

    def store_data(self, X, Y):
        self.ep_obs.append(X)
        self.ep_ls.append(Y)

    def predict(self, observation):
        return self.sess.run(self.prediction, feed_dict={
            self.obs: np.vstack(observation)
        })

def origin_transition(s, v, a):
    ns = s
    nv = (a - 1) * 0.001 + math.cos(3 * s) * (-0.003)
    ns += nv

    return np.array([ns, nv])

def clip(s,v):
    ns = np.clip(s,-1.2, 0.6)
    nv = np.clip(v,-0.07, 0.07)
    if(ns == -1.2 and nv < 0): nv = 0
    return np.array([ns,nv])

if __name__ == '__main__':
    nn = MLP(
        n_features = 3,
        n_hidden = 64,
        n_labels = 2,
    )
    X = np.array([np.linspace(x,y,500) for x,y in zip([-1.5,-0.14],[1,0.14])]).T
    X = np.array([np.append(x, z) for z in range(0,3) for x in X])
    Y = np.array([origin_transition(x[0], x[1], x[2]) for x in X])
    for (x,y) in zip(X,Y): 
        nn.store_data(x,y)
    nn.learn(batch_size = 100, iters = 10000)
    
    p = np.linspace(-1.2,0.6,100)
    v = np.linspace(-0.14,0.14,100)
    X,Y = np.meshgrid(p,v)
    testX = np.array([[x,y] for x,y in zip(np.ravel(X),np.ravel(Y))])
    
    tot_err = np.array([0 ,0], dtype = np.float32)

    for a in range(0,3):
        testY = np.array([origin_transition(x[0], x[1], a) for x in testX])
        predY = nn.predict([np.append(x,a) for x in testX])
        error = (testY - predY) / [1.8, 0.14]
        tot_err += np.mean(error, axis = 0) / 3

    while True:
        a,b,c = input("a,b,c:").split()
        a = float(a)
        b = float(b)
        c = float(c)
        arr = origin_transition(a,b,c)
        s,v = clip(arr[0], arr[1])
        print(s,v)
        print(nn.predict([[a,b,c]]))

