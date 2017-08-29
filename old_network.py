import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt

# reproducible
np.random.seed(1)
tf.set_random_seed(1)

class MLP:
    def __init__(
        self,
        n_features,
        n_labels,
        n_hidden_units=30,
        n_hidden_units_2=30,
        learning_rate=0.01,
        activation=tf.nn.tanh,
        optimizer=tf.train.GradientDescentOptimizer
    ):
        self.n_labels = n_labels
        self.n_features = n_features
        self.lr = learning_rate

        self.n_hidden_units = n_hidden_units
        self.n_hidden_units_2 = n_hidden_units_2

        self.optimizer = optimizer

        self.activation = activation

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def addlayer(self, inputs, n_in, n_out, activation=None):
        W = tf.random_normal([n_in,n_out])
        b = tf.zeros([1,n_out])+0.1
        out = tf.matmul(inputs, W) + b

        if activation != None:
            out = activation(out)
        return out

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.labels = tf.placeholder(tf.float32, [None, self.n_labels], name="actions_num")
            #self.value = tf.placeholder(tf.float32, [None, ], name="actions_value")

        layer_1 = tf.layers.dense(
            inputs = self.obs,
            units = self.n_hidden_units,
            activation=self.activation,
            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.2),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer1',
        )

        # layer_2 = tf.layers.dense(
        #     inputs = layer_1,
        #     units = self.n_hidden_units_2,
        #     activation=self.activation,
        #     kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
        #     bias_initializer=tf.constant_initializer(0.1),
        #     name='layer2',
        # )

        # layer = self.addlayer(self.obs, self.n_features, self.n_hidden_units, self.activation)
        # layer = self.addlayer(layer, self.n_hidden_units, self.n_labels)

        layer_3 = tf.layers.dense(
            inputs = layer_1,
            units = self.n_labels,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer3',
        )

        self.prediction = layer_3

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.prediction - self.labels), reduction_indices=[1]))
            # neg_log_prob = tf.nn.sparse_softmax_entropy_with_logits(logits=layer_2, labels=self.n_labels)
            self.loss = loss
            # loss = tf.reduce_mean(neg_log_prob * self.value)

        with tf.name_scope('train'):
            self.train_op = self.optimizer(self.lr).minimize(loss)
    
    def learn(self, batch_size = 1,iters=1000):
        batch_num = (len(self.ep_obs)+batch_size-1)/batch_size
        for iter in xrange(iters):
            batch_index = range(batch_num) 
            np.random.shuffle(batch_index)
            for i in range(batch_num):
                self.sess.run(self.train_op,
                    feed_dict = {
                        self.obs: np.vstack(self.ep_obs[batch_index[i] * batch_size:(batch_index[i]+1)*batch_size]),
                        self.labels: np.vstack(self.ep_as[batch_index[i] * batch_size:(batch_index[i]+1)*batch_size]),
                    }
                )
            if iter % 100 == 0:
                print 'iters:', iter, 'loss:', self.sess.run(self.loss, feed_dict={
                    self.obs: self.ep_obs,
                    self.labels: self.ep_as,
                })
                # tmp = self.sess.run(self.prediction, feed_dict={
                #     self.obs: self.ep_obs
                # })
                # self.visualize(self.ep_obs , tmp)

        self.ep_obs , self.ep_as = [] , []
    
    def predict(self, observation):
        return self.sess.run(self.prediction, feed_dict={
            self.obs: np.vstack(observation)
        })


    def plot_decision_boundary(self, pred_func, X, y):
        # Set min and max values and give it some padding
        x_axis = [X[i][0] for i in range(len(X))]
        y_axis = [y[i] for i in range(len(X))]
        x_min, x_max = min(x_axis) - .5, max(x_axis) + .5
        y_min, y_max = min(y_axis) - .5, max(y_axis) + .5
        h = 0.01
        # Generate a grid of points with distance h between them
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole gid
        # Z = np.array(pred_func(np.c_[xx.ravel(), yy.ravel()]))
        # Z = Z.reshape(xx.shape) 
        # Plot the contour and training examples
        # plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(x_axis, y_axis, cmap=plt.cm.Spectral)
        plt.show()

    def visualize(self,X ,y):
        # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
        # plt.show()
        self.plot_decision_boundary(lambda x:self.predict(x), X, y)
        plt.title("Logistic Regression")

if __name__ == '__main__':
    net = MLP(2,1,n_hidden_units=5,n_hidden_units_2=5)

    X, Y = datasets.make_moons(200, noise=0.20)
    # plt.scatter(X[:,0], X[:,1], s=40, c=Y, cmap=plt.cm.Spectral)
    # plt.show()
    # X = [
    #     [0,0,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,1],
    #     [0,1,0,0],[0,1,1,0],[0,1,0,1],[0,1,1,1],
    #     [1,0,0,0],[1,0,1,0],[1,0,0,1],[1,0,1,1],
    #     [1,1,0,0],[1,1,1,0],[1,1,0,1],[1,1,1,1]
    # ]
    # Y = [
    #     [0,1],[0,0],[0,1],[0,1],
    #     [1,0],[0,1],[1,1],[0,0],
    #     [0,1],[0,0],[1,0],[0,1],
    #     [1,1],[1,0],[0,0],[1,1]
    # ]
    net.ep_obs = X[:len(X)/2]
    net.ep_as = [[tmp] for tmp in Y[:len(X)/2]]
        
    net.learn(iters = 10000)
    ans = net.predict(X[len(X)/2:])
    ans = [1 if a > 0.5 else 0 for a in ans]
    print [ans , Y[len(X)/2:]]
    cnt = 0
    for i, j in zip(ans, Y[len(X)/2:]):
        if i != j: cnt+=1
    print cnt / len(Y[len(X)/2:])
    