import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from misc.utils import save_images, save_plot

import tensorflow as tf

FLAGS = None
tf.logging.set_verbosity(tf.logging.INFO)
learning_rate = 0.05
batch_size = 128
decay_param = 0.0001
momentum_param = 0.1
num_epochs = 100


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 15 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([9, 9, 1, 15])
        b_conv1 = bias_variable([15])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 15 feature maps to 20.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 20])
        b_conv2 = bias_variable([20])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 3x3x20 feature maps -- maps this to 100 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([3 * 3 * 20, 100])
        b_fc1 = bias_variable([100])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 3 * 3 * 20])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Map the 100 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([100, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(decay_param))


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    config = 1

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv = deepnn(x)
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    if config == 1:
        with tf.name_scope('Optimizer'):
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    elif config == 2:
        with tf.name_scope('Optimizer'):
            train_step = tf.train.MomentumOptimizer(learning_rate, momentum_param).minimize(
                cross_entropy)
    elif config == 3:
        with tf.name_scope('Optimizer'):
            train_step = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=1e-4, momentum=0.9,
                                                   epsilon=1e-6,).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_iters = mnist.train.images.shape[0]//batch_size
        train_errors = []
        test_accuracy = []
        for epoch in range(num_epochs):
            for i in range(num_iters):
                batch = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
            l = sess.run(cross_entropy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            train_errors.append(l)
            test_accuracy.append(acc)
            print('epoch %d: loss = %.2f, accuracy = %.2f' % (epoch, l, acc))
        save_plot(train_errors, "cnn_classify_%d_train_error" % (config),
                  ylabel="Categorical Cross Entropy Loss")
        save_plot(test_accuracy, "cnn_classify_%d_test_acc" % (config), label="test_accuracy",
                  ylabel="Accuracy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
