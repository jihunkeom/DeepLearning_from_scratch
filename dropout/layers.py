import numpy as np


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        self.name = "Affine"

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x

        return out

    def backward(self, dout):
        W, b = self.params
        db = np.sum(dout, axis=0)
        dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, W.T)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Relu:
    def __init__(self):
        self.params, self.grads = [], []
        self.mask = None
        self.name = "Relu"

    def forward(self, x):
        mask = (x <= 0)
        x[mask] = 0
        self.mask = mask
        return x

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, x, t):
        if x.ndim == 2:
            x = x.T
            x -= np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            y = y.T

        else:
            x -= np.max(x)
            y = np.exp(x) / np.sum(np.exp(x))

        if y.ndim == 1:
            y = y.reshape(1, -1)
            t = t.reshape(1, -1)

        if y.size == t.size:
            t = np.armax(t, axis=1)

        batch_size = y.shape[0]
        loss = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

        self.cache = (y, t, batch_size)
        return loss

    def backward(self, dout=1):
        y, t, batch_size = self.cache
        dout = y.copy()
        dout[np.arange(batch_size), t] -= 1
        return dout / batch_size


class Dropout:
    def __init__(self, dropout_ratio):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.name = "Dropout"

    def forward(self, x, train=True):
        self.train = train
        if train:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            x *= self.mask
        else:
            x *= (1.0 - self.dropout_ratio)

        return x

    def backward(self, dout):
        dout *= self.mask
        return dout
