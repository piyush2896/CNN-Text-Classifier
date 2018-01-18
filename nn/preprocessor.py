import numpy as np


def __shuffle(X, Y):
    m = Y.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)

    return X[indices], Y[indices]


def train_test_split(X, Y, shuffle=True, val_size=0.2):
    m = Y.shape[0]
    train_end = int(m * (1-val_size))

    indices = np.arange(m)
    if shuffle:
        np.random.shuffle(indices)

    train_indices = indices[:train_end]
    test_indices = indices[train_end:]

    X_train = X[train_indices, :]
    Y_train = Y[train_indices, :]
    X_test = X[test_indices, :]
    Y_test = Y[test_indices, :]

    return X_train, Y_train, X_test, Y_test

def batches_generator(X_train, Y_train=None, batch_size=32, shuffle=True):
    m = X_train.shape[0]
    
    if Y_train is None:
        for j in range(0, m, batch_size):
            return X_train[j:j+batch_size]
        if j < m:
            return X_train[j:]
    
    else:
        if shuffle:
            X_train, Y_train = __shuffle(X_train, Y_train)

        for j in range(0, m, batch_size):
            yield X_train[j:j+batch_size], Y_train[j:j+batch_size]

        if j < m:
            yield X_train[j:], Y_train[j:]

def vocab_preprocess(file, vocab):
    word_not_in_vocab = '<UNK>'
    pass
