import numpy as np
from functions import rotate_180

class Convolution:
    def __init__(self, W, b, stride, pad=0):
        self.params = [W, b]
        self.grads = [np.zeros_like(W, dtype='f'), np.zeros_like(b, dtype='f')]
        self.stride = stride
        self.pad = pad
        self.x = None

    def forward(self, x):
        self.x = x
        W, b = self.params
        out_h = int((x.shape[0] + 2*self.pad - W.shape[0])/self.stride) + 1
        out_w = int((x.shape[1] + 2*self.pad - W.shape[1])/self.stride) + 1

        out = np.empty((out_h, out_w), dtype='f')
        r, c = 0, 0
        for i in range(out_h):
            c=0
            for j in range(out_w):
                out[i, j] = np.sum(x[r : r+W.shape[0], c : c+W.shape[1]] * rotate_180(W))
                c += self.stride
            r += self.stride

        return out + b

    def backward(self, dout):
        W, b = self.params
        x = self.x
        dW, db = self.grads

        r, c = 0, 0
        for i in range(W.shape[0]):
            c = 0
            for j in range(W.shape[1]):
                dW[i, j] = (np.sum(x[r : r+dout.shape[0], c : c+dout.shape[1]] * dout)).astype('f')
                c += self.stride
            r += self.stride

        db = (np.sum(dout)).astype('f')

        self.params[0][...] = dW
        self.params[1][...] = db

        dx = np.zeros_like(x)

        r, c = 0, 0
        for i in range(0, dx.shape[0], self.stride):
            c = 0
            for j in range(0, dx.shape[1], self.stride):
                dx[i : i+W.shape[0], j : j+W.shape[1]] = W * dout[r, c]
                c += 1
            r += 1

        return dx

class Pooling:
    def __init__(self, pool):
        self.params, self.grads = [], []
        self.pool = pool
        self.x = None
        self.argmax = None

    def forward(self, x):
        self.x = x
        self.argmax = []
        out_h = int((x.shape[0] - self.pool) / self.pool) + 1
        out_w = int((x.shape[1] - self.pool) / self.pool) + 1

        out = np.empty((out_h, out_w))

        r, c = 0, 0
        for i in range(out_h):
            c = 0
            for j in range(out_w):
                self.argmax.append(np.unravel_index(np.argmax(x[r : r+self.pool, c : c+self.pool], axis=None), x[r : r+self.pool, c : c+self.pool].shape))
                out[i, j] = np.max(x[r : r+self.pool, c : c+self.pool])
                c += self.pool
            r += self.pool

        return out.reshape(1, -1)

    def backward(self, dout):
        dx = (np.zeros_like(self.x)).astype('f')
        dout = dout.flatten()

        cnt = 0
        for i in range(0, dx.shape[0], self.pool):
            for j in range(0, dx.shape[1], self.pool):
                dx[i : i+self.pool, j : j+self.pool][self.argmax[cnt][0]][self.argmax[cnt][1]] = dout[cnt]
                cnt += 1

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


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        db = np.sum(dout, axis=0)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        dx = np.dot(dout, W.T)
        return dx


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
            t = np.argmax(t, axis=1)

        batch_size = y.shape[0]
        loss = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

        self.cache = (y, t, batch_size)
        return loss

    def backward(self, dout=1):
        y, t, batch_size = self.cache
        dout = y.copy()
        dout[np.arange(batch_size), t] -= 1
        dout /= batch_size
        return dout