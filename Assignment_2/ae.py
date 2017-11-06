import tensorflow as tf
import numpy as np

from misc.utils import save_images, save_plot
from tensorflow.examples.tutorials.mnist import input_data


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
        counter = 1
        for w, b in zip(self.weights_e, self.biases_e):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = tf.nn.sigmoid(layer)
        out = x.eval(session=sess)
        save_images(np.reshape(out[:100], [100] + [np.sqrt(out.shape[1])]*2), [10, 10],
                    'activations_%d_layer_%d.png' % (self.config, counter))
        counter += 1
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
            print('epoch %d: loss = %.2d' % (epoch, l))
        self.loss_val.append(l)
        self.weights_e.append(sess.run(encode['weights']))
        self.biases_e.append(sess.run(encode['biases']))
        self.weights_d.append(sess.run(decode['weights']))
        self.biases_d.append(sess.run(decode['biases']))
        save_plot(train_errors, "ae_%d_layer_%d" % (self.config, num))
        save_images(np.reshape(np.transpose(self.weights_e[-1])[:100], [100] +
                    [np.sqrt(input_dim)]*2), [10, 10],
                    'weights_%d_layer_%d.png' % (self.config, num))
        return sess.run(encoded, feed_dict={x: data_x_})

    def classify(self, mnist):
        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
        for w, b in zip(self.weights_e, self.biases_e):
            weight = tf.Variable(w, dtype=tf.float32)
            bias = tf.Variable(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = tf.nn.sigmoid(layer)
        w_out = init_weights(self.dims[-1], 10, 'w_out')
        b_out = init_bias(10, 'b_out')
        out = tf.matmul(x, w_out) + b_out
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)

        sess.run(tf.global_variables_initializer())
        num_iters = mnist.train.images.shape[0]//self.batch_size
        train_errors = []
        test_accuracy = []
        for epoch in range(self.epoch):
            for i in range(num_iters):
                batch = mnist.train.next_batch(self.batch_size)
                sess.run(train_op, feed_dict={x: batch[0], y_: batch[1]})
            l = sess.run(loss, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            train_errors.append(l)
            test_accuracy.append(acc)
            print('epoch %d: loss = %.2d, accuracy = %.2d' % (epoch, l, acc))
        save_plot(train_errors, "ae_classify_%d_train_error" % (self.config),
                  ylabel="Categorical Cross Entropy Loss")
        save_plot(test_accuracy, "ae_classify_%d_test_acc" % (self.config), label="test_accuracy",
                  y_label="Accuracy")


def main():
    # reading MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, teX = mnist.train.images, mnist.test.images

    model = StackedAutoEncoder(dims=[900, 625, 400])
    model.fit(trX)
    corrupted, clean = model.transform(teX)
    save_images(np.reshape(clean[:100], [100, 28, 28]), [10, 10], 'clean.png')
    save_images(np.reshape(corrupted[:100], [100, 28, 28]), [10, 10], 'corrupted.png')
    model.classify(mnist)


if __name__ == "__main__":
    main()
