import argparse
import numpy as np
import tensorflow as tf

NUM_CLASSES = 6
LEARNING_RATE = 0.01
NUM_EPOCHS = 2


def model(inputs, num_hidden_units, weight_decay):
    with tf.variable_scope('Hidden'):
        hidden = tf.contrib.layers.fully_connected(
                inputs, num_hidden_units, activation_fn=tf.nn.sigmoid,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    with tf.variable_scope('Output'):
        output = tf.contrib.layers.fully_connected(
                hidden, NUM_CLASSES, activation_fn=None,
                biases_initializer=tf.zeros_initializer(),
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    return output


def ops(train_inputs, train_outputs, test_inputs, test_outputs, num_hidden_units, weight_decay,
        global_step):
    MLP = tf.make_template('3-MLP', model, num_hidden_units=num_hidden_units,
                           weight_decay=weight_decay)
    MLP_Train = MLP(train_inputs)
    MLP_Test = MLP(test_inputs)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_outputs, logits=MLP_Train,
                                                          name="Loss")
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, name="Optimizer",
                                                                          global_step=global_step)
    predictions_train = tf.equal(tf.cast(tf.argmax(MLP_Train, 1), dtype=tf.int32), train_outputs)
    train_accuracy = tf.reduce_mean(tf.cast(predictions_train, tf.float32))
    predictions_test = tf.equal(tf.cast(tf.argmax(MLP_Test, 1), dtype=tf.int32), test_outputs)
    test_accuracy = tf.reduce_mean(tf.cast(predictions_test, tf.float32))
    tf.summary.scalar("Loss", tf.reduce_mean(loss))
    tf.summary.scalar("Accuracy", train_accuracy)
    return {"Optimizer": optimizer, "Accuracy": train_accuracy, "Loss": loss}, test_accuracy


def get_sample(filename_queue):
    reader = tf.TextLineReader()
    _, row = reader.read(filename_queue)
    record_defaults = [[0.0]]*37
    values = tf.decode_csv(row, record_defaults=record_defaults)
    return values[:-1], tf.cast(values[-1], tf.int32)


def test_input_batch(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    features, label = get_sample(filename_queue)
    example_batch, label_batch = tf.train.batch([features, label], batch_size=batch_size,
                                                allow_smaller_final_batch=True)
    return example_batch, label_batch


def train_input_batch(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    features, label = get_sample(filename_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


def test(sess, test_op, batch_size, num_samples):
    test_acc, counter = 0, 0.0
    for i in range(int(np.ceil(num_samples/batch_size))):
        acc = sess.run(test_op)
        test_acc += acc
        counter += 1
    return test_acc/counter


def train(filename, num_samples, test_filename, test_num_samples, num_hidden_units, batch_size,
          weight_decay, log_file):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    train_feature_batch, train_label_batch = train_input_batch(filename, batch_size)
    test_feature_batch, test_label_batch = test_input_batch(test_filename, batch_size)
    global_step = tf.Variable(0, trainable=False, name="global_step")
    train_step, test_step = ops(train_feature_batch, train_label_batch, test_feature_batch,
                                test_label_batch, num_hidden_units, weight_decay, global_step)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("./tb_logs")
    train_writer.add_graph(tf.get_default_graph())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    log_file = open('logs/'+log_file, 'w')
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, counter = 0, 0, 0.0
        for i in range(int(np.floor(num_samples/batch_size))):
            step, summaries, g_s = sess.run([train_step, merged, global_step])
            train_loss += np.mean(step['Loss'])
            train_acc += step['Accuracy']
            counter += 1
            train_writer.add_summary(summaries, g_s)
        test_acc = test(sess, test_step, batch_size, test_num_samples)
        log_file.write("Epoch: %d\tTrain Loss: %.2f\tTrain Accuracy: %.2f\tTest Accuracy: %.2f\n" %
                       (epoch+1, train_loss/counter, train_acc/counter, test_acc))
        log_file.flush()
    log_file.close()
    coord.request_stop()
    coord.join(threads)
    print('Done training -- epoch limit reached')
    sess.close()


def parameter_search(filename, num_samples, test_filename, test_num_samples):
    weight_decay_params = [0.0, 10e-3, 10e-6, 10e-9, 10e-12]
    batch_size_params = [4, 8, 16, 32, 64]
    num_hidden_units_params = [5, 10, 15, 20, 25]
    for weight_decay in weight_decay_params:
        for batch_size in batch_size_params:
            for num_hidden_units in num_hidden_units_params:
                log_file = "3_mlp_%d_%d_%d.txt" % (weight_decay, batch_size, num_hidden_units)
                print("Evaluating "+log_file)
                train(filename, num_samples, test_filename, test_num_samples, num_hidden_units,
                      batch_size, weight_decay, log_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='./dataset/sat_train.csv', help='CSV w train data')
    parser.add_argument('--train_samples', default="4435", help='No. of samples in train set')
    parser.add_argument('--test_data', default='./dataset/sat_test.csv', help='CSV w test data')
    parser.add_argument('--test_samples', default="2000", help='No. of samples in test set')

    args = parser.parse_args()
    parameter_search(args.train_data, int(args.train_samples), args.test_data,
                     int(args.test_samples))


if __name__ == "__main__":
    main()
