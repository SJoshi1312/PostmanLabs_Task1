import numpy as np


def one_hot(label, num_classes=10):
    y = np.zeros((num_classes, 1))
    y[label] = 1
    return y


def output_delta(prediction, label):
    y = one_hot(label)
    return prediction - y


def relu_derivative(x):
    return (x > 0).astype(float)


def backprop_layer(W, a_previous, delta):
    dW = delta @ a_previous.T
    db = delta
    da_previous = W.T @ delta

    return dW, db, da_previous