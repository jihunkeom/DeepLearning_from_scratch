import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x -= np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        y = y.T
    else:
        x -= np.max(x)
        y = np.exp(x) / np.sum(np.exp(x))

    return y


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)

    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

    return grads
