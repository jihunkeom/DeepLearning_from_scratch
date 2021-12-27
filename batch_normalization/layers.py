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
        self.name = "SoftmaxWithLoss"

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


class BatchNormalization:
    def __init__(self, gamma, beta):
        self.params = [gamma, beta]
        self.grads = [np.zeros_like(gamma), np.zeros_like(beta)]
        self.cache = None
        self.running_mean = None
        self.running_var = None
        self.name = "BatchNorm"

    def forward(self, x, train=True):
        gamma, beta = self.params
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train:
            N = x.shape[0]
            mu = np.mean(x, axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 1e-7)
            xn = xc / std
            gam_xn = xn * gamma
            out = gam_xn + beta
            self.running_mean = (self.running_mean + mu) / 2
            self.running_var = (self.running_var + var) / 2
            self.cache = (xn, gamma, std, xc, var, N)
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 1e-7)))
            out = gamma * xn + beta

        return out

    def backward(self, dout):
        xn, gamma, std, xc, var, N = self.cache
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * xn, axis=0)
        dxn = dout * gamma
        dxc1 = dxn / std
        dstd = np.sum(dxn * xc, axis=0) * (-1/var)
        dvar = dstd / (2*std)
        dvar = dvar.reshape(1, dvar.shape[0])
        dxc2 = np.repeat(dvar/N, N, axis=0) * 2*xc
        dxc = dxc1 + dxc2
        dmu = np.sum(-dxc, axis=0)
        dmu = dmu.reshape(1, dmu.shape[0])
        dprev = dxc + np.repeat(dmu/N, N, axis=0)

        self.grads[0][...] = dgamma
        self.grads[1][...] = dbeta

        return dprev
