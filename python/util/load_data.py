import gzip
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


def unzip_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    # reshaping and normalizing
    data = data.reshape(-1, 1, 28, 28).astype(np.float32)

    return data


def unzip_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data


def get_zero_one_labels(y, num):
    y_copy = y
    y = []
    for l in y_copy:
        if l == num:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y)
    return y


def getCifarData(num, seed, data_path, flatten=False, normalize=True, transpose=True):

    X = []
    y = []

    count = 1
    filename = data_path + "cifar-10-batches-py/data_batch_" + str(count)
    while os.path.exists(filename):
        with open(filename, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        X.append(batch['data'])
        y.append(batch['labels'])
        count += 1
        filename = data_path + "cifar-10-batches-py/data_batch_" + str(count)

    X = np.concatenate(X)
    y = np.concatenate(y)

    filename = data_path + "cifar-10-batches-py/test_batch"
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')

    X_test = batch['data']
    y_test = np.array(batch['labels'])

    if normalize:
        X = X.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.

    # Reshaping
    if not flatten:
        X = X.reshape((X.shape[0], 3, 32, 32))
        X_test = X_test.reshape((X_test.shape[0], 3, 32, 32))

        if transpose:
            X = np.transpose(X, (0, 2, 3, 1))
            X_test = np.transpose(X_test, (0, 2, 3, 1))

    X = X[np.where(y == num)]
    y = y[np.where(y == num)]

    # Shuffle
    np.random.seed(seed)
    np.random.shuffle(X)

    y[:] = 0
    y_test = get_zero_one_labels(y_test, num)


    return X, y, X_test, y_test