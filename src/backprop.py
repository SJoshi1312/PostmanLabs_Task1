import numpy as np


def one_hot(labels, num_classes=10):

    y = np.zeros((num_classes, len(labels)))

    for i, label in enumerate(labels):
        y[label, i] = 1

    return y


def output_delta(prediction, labels):
    y = one_hot(labels)
    return prediction - y


def relu_derivative(x):
    return (x > 0).astype(float)


def backprop_layer(W, a_previous, delta):

    batch_size = a_previous.shape[1]

    dW = (delta @ a_previous.T) / batch_size
    db = np.sum(delta, axis=1, keepdims=True) / batch_size
    da_previous = W.T @ delta

    return dW, db, da_previous