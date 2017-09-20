import argparse
import numpy as np
import tensorflow as tf


def get_sample(filename_queue):
    reader = tf.TextLineReader()
    _, row = reader.read(filename_queue)
    record_defaults = [[0.0]]*37
    values = tf.decode_csv(row, record_defaults=record_defaults)
    return values[:-1], values[-1]


def input_batch(filename, batch_size, num_epochs=None):
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
    sess.run(tf.global_variables_initializer())
    feature_batch, label_batch = input_batch(filename, batch_size, num_epochs)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for epoch in range(num_epochs):
            for i in range(int(np.floor(num_samples/batch_size))):
                print (sess.run([feature_batch, label_batch]))
    except tf.errors.OutOfRangeError:
        print ('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
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
