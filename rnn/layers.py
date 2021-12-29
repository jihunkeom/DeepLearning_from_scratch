import numpy as np

class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b
        h_next = np.tanh(t)
        self.cache = [x, h_prev, h_next]

        return h_next

    def backward(self, dout):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dout * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None

    def forward(self, xs):
        self.layers = []
        N, T, D = xs.shape
        Wx, Wh, b = self.params
        D, H = Wx.shape

        if self.h is None:
            self.h = np.zeros((N, H))

        hs = np.empty((N, T, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        N, T, H = dhs.shape
        Wx, Wh, b = self.params
        D, H = Wx.shape

        dxs = np.empty((N, T, D))
        dh = 0
        for i in range(len(self.grads)):
            self.grads[i][...] = np.zeros_like(self.params[i])

        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                self.grads[i] += grad

        self.dh = dh
        return dxs


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        N, T, H = x.shape
        self.x = x
        H, V = W.shape

        rx = x.reshape(N * T, H)
        out = np.dot(rx, W) + b
        out = out.reshape((N, T, V))
        return out

    def backward(self, dout):
        W, b = self.params
        N, T, H = self.x.shape

        rx = self.x.reshape(N * T, H)
        dout = dout.reshape(N * T, -1)

        dW = np.dot(rx.T, dout)
        db = np.sum(dout, axis=0)
        dx = np.dot(dout, W.T)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        dx = dx.reshape((N, T, H))

        return dx


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        self.idx = idx
        W, = self.params
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        for i, idx in enumerate(self.idx):
            dW[idx] += dout[i]

        self.grads[0][...] = dW
        return None


class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
        self.layers = None

    def forward(self, x):
        self.layers = []
        self.x = x
        W, = self.params
        N, T = x.shape
        V, D = W.shape
        xs = np.empty((N, T, D), dtype='f')
        for t in range(T):
            layer = Embedding(W)
            xs[:, t, :] = layer.forward(x[:, t])
            self.layers.append(layer)
        return xs

    def backward(self, dout):
        N, T, D = dout.shape
        W, = self.params
        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad

        return None


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.ignore = -1
        self.cache = None

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:
            ts = np.argmax(ts, axis=2)
        mask = (ts != self.ignore)

        rx = xs.reshape(N * T, -1)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        rx = rx.T
        rx -= np.max(rx, axis=0)
        tmp = np.exp(rx) / np.sum(np.exp(rx), axis=0)
        ys = tmp.T

        ls = np.log(ys[np.arange(N * T), ts] + 1e-7)
        ls *= mask
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ys, ts, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ys, ts, mask, (N, T, V) = self.cache

        ys[np.arange(N * T), ts] -= 1
        ys /= mask.sum()
        ys *= mask[:, np.newaxis]

        return ys
