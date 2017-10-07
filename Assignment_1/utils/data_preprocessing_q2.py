import argparse
import numpy as np
import os


def scale(x):
    x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
    return (x - x_min)/(x_max-x_min)


def normalize(x):
    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    return (x - x_mean)/x_std


def preprocess(filepath):
    input_data = np.loadtxt(filepath, delimiter=',')
    x_data, y_data = input_data[:, :8], input_data[:, 8]
    x_data = normalize(scale(x_data))
    y_data = np.expand_dims(y_data, -1)
    data = np.hstack((x_data, y_data))
    np.random.shuffle(data)
    train_split = int(data.shape[0]*0.7)
    train_data = data[:train_split, :]
    test_data = data[train_split:, :]
    output_filepath = filepath[:filepath.rfind(".")]
    np.save(output_filepath+"_train", train_data)
    np.save(output_filepath+"_test", test_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', default=os.path.join("..", "dataset",
                        "cal_housing.data"), help="Path to the training data")
    args = parser.parse_args()
    preprocess(args.filepath)


if __name__ == '__main__':
    main()
