import numpy as np

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        self.idx = idx
        W, = self.params
        return W[idx]

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        for i, idx in enumerate(self.idx):
            dW[idx] += dout[i]
        self.grads[0][...] = dW
        return None

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.c = None

    def forward(self, c):
        W, = self.params
        self.c = c
        out = np.dot(c, W)
        return out

    def backward(self, dout):
        W, = self.params
        dW = np.dot(self.c.T, dout)
        dx = np.dot(dout, W.T)
        self.grads[0][...] = dW

        return dx

    def __str__(self):
        return "MatMul"

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            exp_x = np.exp(x)
            y = exp_x / np.sum(exp_x, axis=0)
            self.y = y.T
        else:
            x = x - np.max(x)
            exp_x = np.exp(x)
            self.y = exp_x / np.sum(exp_x)

        if self.y.ndim == 1:
            self.y.reshape(1, -1)
            self.t.reshape(1, -1)

        if self.y.size == self.t.size:
            self.t = np.argmax(self.t, axis=1)

        batch_size = self.y.shape[0]
        loss = -np.sum(np.log(self.y[np.arange(batch_size), self.t] + 1e-7))
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.y.size == self.t.size:
            return (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            return dx

    def __str__(self):
        return "SoftmaxWithLoss"