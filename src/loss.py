import numpy as np


def cross_entropy(y_true, y_pred):
    return -np.log(y_pred[y_true, 0])