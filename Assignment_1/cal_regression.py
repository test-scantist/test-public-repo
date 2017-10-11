import argparse
import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import KFold

NUM_EPOCHS = 1000
BATCH_SIZE = 32


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


def mlp_4(inputs, num_hidden_units):
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
    with tf.variable_scope('Output'):
        output = tf.contrib.layers.fully_connected(
                hidden_2, 1, activation_fn=None,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer())
    return output


def mlp_5(inputs, num_hidden_units):
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
        hidden_3 = tf.contrib.layers.fully_connected(
                hidden_2, 20, activation_fn=tf.nn.sigmoid,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.variable_scope('Output'):
        output = tf.contrib.layers.fully_connected(
                hidden_3, 1, activation_fn=None,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer())
    return output


def ops(inputs, outputs, num_hidden_units, lr_1, lr_2, mlp_layers=3):
    if mlp_layers == 3:
        mlp = mlp_3(inputs, num_hidden_units)
    elif mlp_layers == 4:
        mlp = mlp_4(inputs, num_hidden_units)
    else:
        mlp = mlp_5(inputs, num_hidden_units)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(outputs - mlp), name='loss')
    vars_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Hidden') +\
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Hidden_1')
    vars_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Hidden_2') +\
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Hidden_3') +\
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Output')
    op_1 = tf.train.GradientDescentOptimizer(lr_1).minimize(loss, name="optimizer_1",
                                                                     var_list=vars_1)
    op_2 = tf.train.GradientDescentOptimizer(lr_2).minimize(loss, name="optimizer_2",
                                                                     var_list=vars_2)
    with tf.name_scope('optim'):
        op = tf.group(op_1, op_2)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.abs((outputs - mlp)))
    train_test_op = {"Optimizer": op, "Accuracy": accuracy, "Loss": loss}
    return train_test_op


def shuffle_data(x_data, y_data):
    data = np.hstack([x_data, y_data])
    np.random.shuffle(data)
    return data[:, :8], data[:, 8:]


def train_test_cv(filename, test_filename, num_hidden_units, learning_rate, log_file, mlp_layers=3):
    train_data = np.load(filename)
    train_x, train_y = train_data[:, :8], train_data[:, 8:]
    test_data = np.load(test_filename)
    test_x, test_y = test_data[:, :8], test_data[:, 8:]
    test_iters = test_x.shape[0]/BATCH_SIZE
    kf = KFold(n_splits=5)
    split_counter = 0
    for train_index, val_index in kf.split(train_x):
        split_counter += 1
        print 'Evaluating %s Split %d' % (log_file, split_counter)
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=(None, 8))
        y = tf.placeholder(tf.float32, shape=(None))
        x_train, x_val = train_x[train_index], train_x[val_index]
        y_train, y_val = train_y[train_index], train_y[val_index]
        train_iters = len(train_index)/BATCH_SIZE
        val_iters = len(val_index)/BATCH_SIZE
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=config) as sess:
            train_test_step = ops(x, y, num_hidden_units, learning_rate, learning_rate, mlp_layers)
            train_writer = tf.summary.FileWriter("./tb_logs/"+log_file)
            train_writer.add_graph(tf.get_default_graph())
            sess.run(tf.global_variables_initializer())
            logs = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'test_acc': 0}
            for epoch in range(NUM_EPOCHS):
                x_train, y_train = shuffle_data(x_train, y_train)
                train_loss, train_acc, train_counter = 0, 0, 0
                for i in range(train_iters):
                    step = sess.run(train_test_step, feed_dict={
                        x: x_train[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE],
                        y: y_train[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]})
                    train_loss += step['Loss']
                    train_acc += step['Accuracy']
                    train_counter += 1
                val_loss, val_acc, val_counter = 0, 0, 0
                for i in range(val_iters):
                    va_lo, va_ac  = sess.run([train_test_step['Loss'], train_test_step['Accuracy']], feed_dict={
                        x: x_val[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE],
                        y: y_val[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]})
                    val_acc += va_ac
		    val_loss += va_lo
                    val_counter += 1
                logs['epoch'].append(epoch)
                logs['train_loss'].append(train_loss/train_counter)
                logs['train_acc'].append(train_acc/train_counter)
                logs['val_loss'].append(val_loss/val_counter)
                logs['val_acc'].append(val_acc/val_counter)
            test_acc, test_counter = 0, 0
            for i in range(test_iters):
                step = sess.run(train_test_step['Accuracy'], feed_dict={
                    x: test_x[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE],
                    y: test_y[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]})
                test_acc += step
                test_counter += 1
            logs['test_acc'] = test_acc/test_counter
            with open('logs_2/'+log_file+"_split_"+str(split_counter)+'.json', 'w') as outfile:
                json.dump(logs, outfile, sort_keys=True, indent=4)
            print('Done training -- epoch limit reached')
            sess.close()


def train_test(filename, test_filename, num_hidden_units, lr_1, lr_2, log_file, mlp_layers=3):
    print 'Evaluating %s Full' % (log_file)
    train_data = np.load(filename)
    train_x, train_y = train_data[:, :8], train_data[:, 8:]
    test_data = np.load(test_filename)
    test_x, test_y = test_data[:, :8], test_data[:, 8:]
    train_iters = train_x.shape[0]/BATCH_SIZE
    test_iters = test_x.shape[0]/BATCH_SIZE
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=(None, 8))
    y = tf.placeholder(tf.float32, shape=(None))
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        train_test_step = ops(x, y, num_hidden_units, lr_1, lr_2, mlp_layers)
        train_writer = tf.summary.FileWriter("./tb_logs/"+log_file)
        train_writer.add_graph(tf.get_default_graph())
        sess.run(tf.global_variables_initializer())
        logs = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        for epoch in range(NUM_EPOCHS):
            x_train, y_train = shuffle_data(train_x, train_y)
            train_loss, train_acc, train_counter = 0, 0, 0
	    for i in range(train_iters):
		step = sess.run(train_test_step, feed_dict={
			x: x_train[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE],
			y: y_train[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]})
		train_loss += step['Loss']
		train_acc += step['Accuracy']
		train_counter += 1
            test_loss, test_acc, test_counter = 0, 0, 0
	    for i in range(test_iters):
		te_lo, te_ac = sess.run([train_test_step['Loss'], train_test_step['Accuracy']], feed_dict={
			x: test_x[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE],
			y: test_y[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]})
		test_loss += te_lo
		test_acc += te_ac
		test_counter += 1
            logs['epoch'].append(epoch)
            logs['train_loss'].append(train_loss/train_counter)
            logs['train_acc'].append(train_acc/train_counter)
            logs['test_loss'].append(test_loss/test_counter)
            logs['test_acc'].append(test_acc/test_counter)
        with open('logs_2/'+log_file+'_full.json', 'w') as outfile:
            json.dump(logs, outfile, sort_keys=True, indent=4)
        print('Done training -- epoch limit reached')
        sess.close()

def parameter_search(filename, test_filename):
    learning_rate = [1e-3, 0.5*1e-3, 1e-4, 0.5*1e-4, 1e-5]
    num_hidden_units_params = [20, 40, 50, 60]
    num_units = 30
    for lr in learning_rate:
        log_file = '3_layer_mlp_%d_%s' % (num_units, lr)
        train_test_cv(filename, test_filename, num_units, lr, log_file)


def run_mlp(filename, test_filename):
    num_units = 60
    lr = 1e-5
    log_file = '5_layer_mlp_%d_%s' % (num_units, lr)
    train_test(filename, test_filename, num_units, lr, 1e-4, log_file, mlp_layers=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='./dataset/cal_housing_train.npy')
    parser.add_argument('--test_data', default='./dataset/cal_housing_test.npy')
    parser.add_argument('--mlp_layers', default=3, help='Number of layers in the mlp')

    args = parser.parse_args()
    parameter_search(args.train_data, args.test_data)
    #run_mlp(args.train_data, args.test_data)


if __name__ == "__main__":
    main()
