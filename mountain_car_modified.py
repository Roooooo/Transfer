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
            for i in range(int(batch_num * 8 / 10)):
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
        self.ep_obs , self.ep_ls = [] , []

    def store_data(self, X, Y):
        self.ep_obs.append(X)
        self.ep_ls.append(Y)

    def predict(self, observation):
        return self.sess.run(self.prediction, feed_dict={
            self.obs: np.vstack(observation)
        })


class MountainCarEnv_Modified(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        
        self.viewer = None

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.trans_net = MLP(
            n_features = 3, 
            n_hidden = 256, 
            n_labels = 2,
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high)

        self._seed()

        self._step_cnt = 0
        self.trajs_set = []

        self.max_init_iters = 5
        self._init_net()
        self.reset()

    def save_traj(self, traj):
        self.trajs_set.append(traj)

    def calc_reward(self, state):
        ret = [0,0]
        cnt = 0
        for traj in trajs_set:
            traj_state = traj[0][self._step_cnt]
            ret += -np.exp(-np.sum(np.square((state - traj_state) / [1.8, 0.14])))
            cnt += 1
        ret /= cnt
        return ret

    def origin_transition(self, s, v, a):
        ns = s
        nv = (a - 1) * 0.001 + math.cos(3 * s) * (-0.003)
        nv = np.clip(nv, -self.max_speed, self.max_speed)
        ns += nv
        ns = np.clip(ns, self.min_position, self.max_position)

        if(ns == self.min_position and nv < 0): nv = 0
        return np.array([ns, nv])

    def _init_net(self):
        sample_num = 50000
        for iter in range(self.max_init_iters):
            X = np.random.rand(sample_num, 2) * [1.8, 0.14] + [self.min_position, -self.max_speed]
            X = np.column_stack((X, np.random.random_integers(0,2,[sample_num,1])))
            # X = np.array([np.linspace(x,y,500) for x,y in zip(self.low, self.high)]).T
            # X = np.array([np.append(x, z) for z in range(-1,2) for x in X])
            Y = np.array([self.origin_transition(x[0], x[1], x[2]) for x in X])
            for (x,y) in zip(X,Y): 
                self.trans_net.store_data(x,y)
            print('Init transition network.')
            self.trans_net.learn(batch_size = 50, iters = 2000)

    def _seed(self, seed=None): 
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _state_modify(self, position, velocity, action):
        position, velocity = self.trans_net.predict(np.array([[position, velocity, action]]))[0]
        position = np.clip(position, self.min_position, self.max_position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        return position, velocity

    def _step(self, action, confidence_reward = False):
        self._step_cnt += 1
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state

        position, velocity = self._state_modify(position, velocity, action)

        done = bool(position >= self.goal_position)
        reward = -1.0

        self.state = (position, velocity)

        if confidence_reward:
            reward += self.calc_reward(self.state)

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self._step_cnt = 0
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
