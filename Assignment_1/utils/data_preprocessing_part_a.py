import argparse
import numpy as np
import os


def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)


def preprocess(filepath):
    train_input = np.loadtxt(filepath, delimiter=' ')
    trainX, trainY = train_input[:, :36], train_input[:, -1]
    trainX_min, trainX_max = np.min(trainX, axis=0), np.max(trainX, axis=0)
    trainX = scale(trainX, trainX_min, trainX_max)
    trainY[trainY == 7] = 6
    trainY = trainY-1
    trainY = np.expand_dims(trainY, -1)
    train_data = np.hstack((trainX, trainY))
    output_filepath = filepath[:filepath.rfind(".")] + ".csv"
    np.savetxt(output_filepath, train_data, delimiter=",")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', default=os.path.join("..", "dataset",
                        "sat_train.txt"), help="Path to the training data")
    args = parser.parse_args()
    preprocess(args.filepath)


if __name__ == '__main__':
    main()
