import argparse
import json
import numpy as np
import tensorflow as tf

NUM_EPOCHS = 1000


def mlp_3(inputs, num_hidden_units):
    with tf.variable_scope('Hidden'):
        hidden = tf.contrib.layers.fully_connected(
                inputs, num_hidden_units, activation_fn=tf.nn.sigmoid,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope('Output'):
        output = tf.contrib.layers.fully_connected(
                hidden, 1, activation_fn=None,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer())
    return output


def mlp_4(inputs, num_hidden_units, weight_decay):
    with tf.variable_scope('Hidden_1'):
        hidden_1 = tf.contrib.layers.fully_connected(
                inputs, num_hidden_units, activation_fn=tf.nn.sigmoid,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope('Hidden_2'):
        hidden_2 = tf.contrib.layers.fully_connected(
                hidden_1, num_hidden_units, activation_fn=tf.nn.sigmoid,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope('Output'):
        output = tf.contrib.layers.fully_connected(
                hidden_2, 1, activation_fn=None,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer())
    return output


def mlp_5(inputs, num_hidden_units, weight_decay):
    with tf.variable_scope('Hidden_1'):
        hidden_1 = tf.contrib.layers.fully_connected(
                inputs, num_hidden_units, activation_fn=tf.nn.sigmoid,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope('Hidden_2'):
        hidden_2 = tf.contrib.layers.fully_connected(
                hidden_1, 20, activation_fn=tf.nn.sigmoid,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope('Hidden_3'):
        hidden_2 = tf.contrib.layers.fully_connected(
                hidden_1, 20, activation_fn=tf.nn.sigmoid,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope('Output'):
        output = tf.contrib.layers.fully_connected(
                hidden_2, 1, activation_fn=None,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer())
    return output


def ops(train_inputs, train_outputs, test_inputs, test_outputs, num_hidden_units, global_step,
        learning_rate, mlp_layers=3, ):
    if mlp_layers == 3:
        mlp = tf.make_template('3-MLP', mlp_3, num_hidden_units=num_hidden_units)
    elif mlp_layers == 4:
        mlp = tf.make_template('4-MLP', mlp_4, num_hidden_units=num_hidden_units)
    else:
        mlp = tf.make_template('5-MLP', mlp_5, num_hidden_units=num_hidden_units)
    with tf.name_scope('train_model'):
        mlp_train = mlp(train_inputs)
    with tf.name_scope('test_model'):
        mlp_test = mlp(test_inputs)
    loss = tf.reduce_mean(tf.square(train_outputs - mlp_train), name='loss')
    vars_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Hidden_1')
    vars_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Hidden_2') +\
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Hidden_3') +\
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Output')
    op_1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, name="optimizer_1",
                                                                     global_step=global_step,
                                                                     var_list=vars_1)
    op_2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, name="optimizer_2",
                                                                     global_step=global_step,
                                                                     var_list=vars_2)
    op = tf.group(op_1, op_2)
    with tf.variable_scope('train_accuracy'):
        train_accuracy = tf.reduce_mean(train_outputs - mlp_train)
    with tf.variable_scope('test_accuracy'):
        test_accuracy = tf.reduce_mean(test_outputs - mlp_test)
    train_op = {"Optimizer": op, "Accuracy": train_accuracy, "Loss": loss}
    test_op = {"Accuracy": test_accuracy}
    return train_op, test_op


def test(sess, test_op, batch_size, num_samples):
    test_acc, counter = 0, 0.0
    for i in range(int(np.ceil(num_samples/batch_size))):
        acc = sess.run(test_op)
        test_acc += acc['Accuracy']
        counter += 1
    return test_acc/counter


def train(filename, num_samples, test_filename, test_num_samples, num_hidden_units, batch_size,
          learning_rate, log_file, mlp_layers=3):
    tf.reset_default_graph()
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    train_feature_batch, train_label_batch = "some reader for train"
    test_feature_batch, test_label_batch = "some reader for test"
    global_step = tf.Variable(0, trainable=False, name="global_step")
    train_step, test_step = ops(train_feature_batch, train_label_batch, test_feature_batch,
                                test_label_batch, num_hidden_units, global_step, learning_rate,
                                mlp_layers=mlp_layers)
    train_writer = tf.summary.FileWriter("./tb_logs")
    train_writer.add_graph(tf.get_default_graph())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    logs = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_acc': [], 'best_acc': 0}
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, counter = 0, 0, 0.0
        for i in range(int(np.floor(num_samples/batch_size))):
            step, g_s = sess.run([train_step, global_step])
            train_loss += np.mean(step['Loss'])
            train_acc += step['Accuracy']
            counter += 1
            train_writer.add_summary(step['Summary'], g_s)
        test_acc = test(sess, test_step, batch_size, test_num_samples)
        if test_acc > logs['best_acc']:
            logs['best_acc'] = test_acc
        # Add logs
        logs['epoch'].append(epoch)
        logs['train_loss'].append(train_loss/counter)
        logs['train_acc'].append(train_acc/counter)
        logs['test_acc'].append(test_acc)
    with open('logs/'+log_file, 'w') as outfile:
        json.dump(logs, outfile, sort_keys=True, indent=4)
    coord.request_stop()
    coord.join(threads)
    print('Done training -- epoch limit reached')
    sess.close()


def parameter_search(filename, num_samples, test_filename, test_num_samples):
    learning_rate = [1e-3, 0.5*1e-3, 1e-4, 0.5*1e-4, 1e-5]
    num_hidden_units_params = [20, 30, 40, 50, 60]
    # need to call train
    return


def run_mlp(filename, num_samples, test_filename, test_num_samples):
    # need to call train for 4 and 5 layer mlp
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='./dataset/sat_train.csv', help='CSV w train data')
    parser.add_argument('--train_samples', default="4435", help='No. of samples in train set')
    parser.add_argument('--test_data', default='./dataset/sat_test.csv', help='CSV w test data')
    parser.add_argument('--test_samples', default="2000", help='No. of samples in test set')
    parser.add_argument('--mlp_layers', default=3, help='Number of layers in the mlp')

    args = parser.parse_args()
    # call parameter search or run_mlp


if __name__ == "__main__":
    main()
