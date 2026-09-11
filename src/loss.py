import numpy as np

def cross_entropy(y_true, y_pred):

    batch_size = len(y_true)

    losses = -np.log(
        y_pred[y_true, np.arange(batch_size)]
    )

    return np.mean(losses)