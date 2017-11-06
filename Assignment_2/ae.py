import tensorflow as tf
import numpy as np

from misc.utils import save_images, save_plot
from tensorflow.examples.tutorials.mnist import input_data

corruption_level = 0.1
training_epochs = 25
learning_rate = 0.1
batch_size = 128


def init_weights(n_visible, n_hidden, name):
    init_w_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
    init_w = tf.random_uniform(shape=[n_visible, n_hidden],
                               minval=-init_w_max,
                               maxval=init_w_max,
                               dtype=tf.float32)
    return tf.Variable(init_w, name=name)


def init_bias(n_hidden, name):
    return tf.Variable(tf.zeros([n_hidden]), name=name)


def get_batch(X, X_, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X_[a]


class StackedAutoEncoder:

    def __init__(self, dims, epoch=25, lr=0.1, batch_size=128, corruption_level=0.1, config=1):
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.dims = dims
        self.depth = len(dims)
        self.corruption_level = corruption_level
        self.weights_e, self.biases_e = [], []
        self.weights_d, self.biases_d = [], []
        self.loss_val = []
        self.config = config

    def add_noise(self, x):
        return np.random.binomial(n=1, p=1-self.corruption_level, size=x.shape)*x

    def fit(self, x):
        for i in range(self.depth):
            print('Layer %d' % (i + 1))
            x = self.run(data_x=self.add_noise(np.copy(x)), data_x_=x,
                         hidden_dim=self.dims[i], num=i+1)

    def transform(self, data):
        tf.reset_default_graph()
        sess = tf.Session()
        corrupted_data = self.add_noise(data)
        x = tf.constant(np.copy(corrupted_data), dtype=tf.float32)
        for w, b in zip(self.weights_e, self.biases_e):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = tf.nn.sigmoid(layer)
        for w, b in zip(self.weights_d[::-1], self.biases_d[::-1]):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = tf.nn.sigmoid(layer)
        return corrupted_data, x.eval(session=sess)

    def run(self, data_x, data_x_, hidden_dim, num):
        tf.reset_default_graph()
        input_dim = len(data_x[0])
        sess = tf.Session()
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_')
        encode = {'weights': init_weights(input_dim, hidden_dim, 'w_%d' % num),
                  'biases': init_bias(hidden_dim, 'b_%d' % num)}
        decode = {'weights': tf.transpose(encode['weights']),
                  'biases': init_bias(input_dim, 'b_inv_%d' % num)}
        encoded = tf.nn.sigmoid(tf.matmul(x, encode['weights']) + encode['biases'])
        decoded = tf.nn.sigmoid(tf.matmul(encoded, decode['weights']) + decode['biases'])

        loss = -tf.reduce_mean(tf.reduce_sum(x_*tf.log(decoded) + (1-x_)*tf.log(1-decoded), axis=1))
        train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)

        sess.run(tf.global_variables_initializer())
        num_iters = input_dim//self.batch_size
        train_errors = []
        for epoch in range(self.epoch):
            for i in range(num_iters):
                b_x, b_x_ = get_batch(data_x, data_x_, self.batch_size)
                sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            l = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
            train_errors.append(l)
            print('epoch {0}: loss = {1}'.format(epoch, l))
        save_plot(train_errors, "ae_%d_layer_%d" % (self.config, num))
        self.loss_val.append(l)
        self.weights_e.append(sess.run(encode['weights']))
        self.biases_e.append(sess.run(encode['biases']))
        self.weights_d.append(sess.run(decode['weights']))
        self.biases_d.append(sess.run(decode['biases']))
        return sess.run(encoded, feed_dict={x: data_x_})


def main():
    # reading MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
        mnist.test.images, mnist.test.labels

    model = StackedAutoEncoder(dims=[900, 625, 400])
    model.fit(trX)
    corrupted, clean = model.transform(teX)
    save_images(np.reshape(clean[:100], [100, 28, 28]), [10, 10], 'plots/clean.png')
    save_images(np.reshape(corrupted[:100], [100, 28, 28]), [10, 10], 'plots/corrupted.png')


if __name__ == "__main__":
    main()
