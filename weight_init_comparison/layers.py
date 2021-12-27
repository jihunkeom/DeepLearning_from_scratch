import numpy as np


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        self.x = x
        W, b = self.params
        out = np.matmul(x, W) + b
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Relu:
    def __init__(self):
        self.params, self.grads = [], []
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        x[self.mask] = 0
        return x

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dout):
        dout *= self.y * (1-self.y)
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

        if y.size == t.size:
            t = np.argmax(t, axis=1)

        if y.ndim == 1:
            y = y.reshape(1, -1)
            t = t.reshape(1, -1)

        batch_size = y.shape[0]
        loss = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

        self.cache = [y, t, batch_size]
        return loss

    def backward(self, dout=1):
        y, t, batch_size = self.cache
        dout = y.copy()
        dout[np.arange(batch_size), t] -= 1
        dout /= batch_size
        return dout
