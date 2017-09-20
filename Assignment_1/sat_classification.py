import argparse
import numpy as np
import tensorflow as tf

NUM_HIDDEN_UNITS = 10
NUM_CLASSES = 6
WEIGHT_DECAY = 10e-6
LEARNING_RATE = 0.01


def model(inputs):
    hidden = tf.contrib.layers.fully_connected(
            inputs, NUM_HIDDEN_UNITS, activation_fn=tf.nn.sigmoid,
            biases_initializer=tf.zeros_initializer(),
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))
    output = tf.contrib.layers.fully_connected(
            hidden, NUM_CLASSES, activation_fn=None,
            biases_initializer=tf.zeros_initializer(),
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))
    return output


def train_op(inputs, outputs, global_step):
    MLP = model(inputs)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=outputs, logits=MLP, name="Loss")
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, name="Optimizer",
                                                                          global_step=global_step)
    predictions = tf.equal(tf.cast(tf.argmax(MLP, 1), dtype=tf.int32), outputs)
    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))
    tf.summary.scalar("Loss", tf.reduce_mean(loss))
    tf.summary.scalar("Accuracy", accuracy)
    return {"Optimizer": optimizer,
            "Accuracy": accuracy,
            "Loss": loss}


def get_sample(filename_queue):
    reader = tf.TextLineReader()
    _, row = reader.read(filename_queue)
    record_defaults = [[0.0]]*37
    values = tf.decode_csv(row, record_defaults=record_defaults)
    return values[:-1], tf.cast(values[-1], tf.int32)


def input_batch(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    features, label = get_sample(filename_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


def train(filename, batch_size, num_epochs, num_samples):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    feature_batch, label_batch = input_batch(filename, batch_size)
    global_step = tf.Variable(0, trainable=False, name="global_step")
    train_step = train_op(feature_batch, label_batch, global_step)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("./logs")
    train_writer.add_graph(tf.get_default_graph())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer()) 
    for epoch in range(num_epochs):
        for i in range(int(np.floor(num_samples/batch_size))):
            step, summaries, g_s = sess.run([train_step, merged, global_step])
            # print(sess.run([feature_batch, label_batch]))
            train_writer.add_summary(summaries, g_s)
    coord.request_stop()
    coord.join(threads)
    print('Done training -- epoch limit reached')
    sess.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', help='CSV with train data')
    parser.add_argument('--train_samples', help='No. of samples in train set')
    parser.add_argument('--batch_size', help='Batch size for training')
    parser.add_argument('--num_epochs', help='Epochs for training')

    args = parser.parse_args()
    train(args.train_data, int(args.batch_size), int(args.num_epochs), int(args.train_samples))


if __name__ == "__main__":
    main()
