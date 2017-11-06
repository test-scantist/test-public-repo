import tensorflow as tf
import numpy as np
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

    def __init__(self, dims, epoch=25, lr=0.1, batch_size=128, corruption_level=0.1):
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.dims = dims
        self.depth = len(dims)
        self.corruption_level = corruption_level
        self.weights, self.biases = [], []
        self.loss_val = []

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
        x = tf.constant(data, dtype=tf.float32)
        for w, b in zip(self.weights, self.biases):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = tf.nn.sigmoid(layer)
        return x.eval(session=sess)

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
        for epoch in range(self.epoch):
            for i in range(num_iters):
                b_x, b_x_ = get_batch(data_x, data_x_, self.batch_size)
                sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            l = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
            print('epoch {0}: loss = {1}'.format(epoch, l))
        self.loss_val.append(l)
        self.weights.append(sess.run(encode['weights']))
        self.biases.append(sess.run(encode['biases']))
        return sess.run(encoded, feed_dict={x: data_x_})


def main():
    # reading MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
        mnist.test.images, mnist.test.labels

    model = StackedAutoEncoder(dims=[900, 625, 400])
    model.fit(trX)
    test_X_ = model.transform(teX)


if __name__ == "__main__":
    main()
