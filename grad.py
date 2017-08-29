import numpy as np
import tensorflow as tf
import gym

env = gym.make('MountainCar-v0')

env.seed(1)

class NN:
    def __init__(
        self,
        n_hid,
        n_actions = 3,
        learning_rate = 0.1,
    ):
        self.n_input = env.observation_space.shape[0] + n_actions
        self.n_hidden_units = n_hid
        self.n_output = env.observation_space.shape[0]
        self.n_actions = n_actions

        self._build_net()

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        self.in_obs = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]], name="in_obs")
        self.in_act = tf.placeholder(tf.float32, [None, self.n_actions], name="in_act") 
        self.in_var = tf.concat([self.in_obs, self.in_act],1) 
        self.label = tf.placeholder(tf.float32, [None, self.n_output], name="label")

        W1 = tf.Variable(tf.random_normal([self.n_input, self.n_hidden_units]))
        B1 = tf.Variable(tf.zeros([1,self.n_hidden_units])+0.1)
        W2 = tf.Variable(tf.random_normal([self.n_hidden_units, self.n_output]))
        B2 = tf.Variable(tf.zeros([1,self.n_output])+0.1)

        layer_in = tf.nn.tanh(tf.add(tf.matmul(self.in_var, W1), B1))
        layer_out = tf.add(tf.matmul(layer_in, W2), B2)

        # layer_in = tf.layers.dense(
        #     inputs = self.in,
        #     units = self.n_hidden_units,
        #     activation = tf.nn.tanh,
        #     kernel_initializer = tf.random_normal_initializer(mean=0, stddec=0.2),
        #     bias_initializer = tf.constant_initializer(0.1),
        # )

        # layer_out = tf.layers.dense(
        #     inputs = layer_in,
        #     units = self.n_output,
        #     activation = None,
        #     kernel_initializer = tf.random_normal_initializer(mean=0, stddec=0.2),
        #     bias_initializer = tf.constant_initializer(0.1),
        # )

        self.prediction = layer_out
        self.loss = tf.reduce_mean(tf.square(self.prediction - self.label), 1)
        self.gradient = tf.gradients(self.loss, tf.trainable_variables())
    
    def generate_trajectory(self, traj):
        sim_traj = []
        sim_traj.append(traj[0])
        for i in range(1,len(traj),2):
            next_obs = self.sess.run(
                self.prediction,
                feed_dict = {
                    self.in_obs: np.vstack(sim_traj[-1]),
                    self.in_act: np.vstack(traj[i]),
                }
            )
            sim_traj.append(traj[i])
            sim_traj.append(next_obs)
        return sim_traj

    def train(self, traj):
        sim_traj = self.train(traj)
        for i in range(1,len(traj),2):
            gradient = self.sess.run(
                self.gradient,
                feed_dict = {
                    self.in_obs: np.vstack(traj[i-1]),
                    self.in_act: np.vstack(traj[i]),
                    self.label: np.vstack(sim_traj[i+1]),
                }
            )

if __name__ == '__main__':
    net = NN(n_hid = 10)
    print 'test'