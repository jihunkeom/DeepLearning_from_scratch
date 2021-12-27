import numpy as np


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        self.shape = None

    def forward(self, x):
        self.shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        W, b = self.params
        out = np.dot(x, W) + b
        return out

    def backward(self, dout):
        W, b = self.params
        dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, W.T)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx.reshape(*self.shape)


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


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        if x.ndim == 2:
            x = x.T
            x -= np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            self.y = y.T
        else:
            x -= np.max(x)
            self.y = np.exp(x) / np.sum(np.exp(x))

        if self.y.ndim == 1:
            self.y = self.y.reshape(1, -1)
            self.t = self.t.reshape(1, -1)

        if self.y.size == self.t.size:
            self.t = np.argmax(self.t, axis=1)

        batch_size = self.y.shape[0]
        loss = - \
            np.sum(
                np.log(self.y[np.arange(batch_size), self.t] + 1e-7)) / batch_size
        return loss

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx /= batch_size
        return dx
