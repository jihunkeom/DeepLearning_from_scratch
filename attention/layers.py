import numpy as np
from functions import sigmoid


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        self.idx = idx
        W, = self.params
        return W[self.idx]

    def backward(self, dout):
        dW, = self.grads

        for i, idx in enumerate(self.idx):
            dW[idx] += dout[i]

        self.grads[0][...] = dW
        return None


class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None

    def forward(self, xs):
        W, = self.params
        N, T = xs.shape
        D = W.shape[1]
        out = np.empty((N, T, D), dtype='f')

        self.layers = []
        for t in range(T):
            layer = Embedding(W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            self.grads[0][...] += layer.grads[0]

        return None


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, xs):
        if xs.ndim == 2:
            xs = xs.T
            xs -= np.max(xs, axis=0)
            ys = np.exp(xs) / np.sum(np.exp(xs), axis=0)
            ys = ys.T
        elif xs.ndim == 1:
            xs -= np.max(xs)
            ys = np.exp(xs) / np.sum(np.exp(xs))

        self.out = ys
        return ys

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        H = Wh.shape[0]
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = c_prev*f + g*i
        h_next = np.tanh(c_next) * o
        self.cache = (f, g, i, o, x, h_prev, c_prev, c_next, h_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        f, g, i, o, x, h_prev, c_prev, c_next, h_next = self.cache

        ds = dc_next + (dh_next * o) * (1 - np.tanh(c_next)**2)
        dc_prev = ds * f

        df = ds * c_prev
        dg = ds * i
        di = ds * g
        do = dh_next * np.tanh(c_next)

        df *= f * (1 - f)
        dg *= (1 - g**2)
        di *= i * (1 - i)
        do *= o * (1 - o)

        dA = np.hstack((df, dg, di, do))
        db = np.sum(dA, axis=0)
        dWx = np.dot(x.T, dA)
        dWh = np.dot(h_prev.T, dA)
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.stateful = stateful
        self.layers = None
        self.h, self.c = None, None
        self.dh = None

    def set_state(self, h, c=None):
        self.h = h
        self.c = c

    def reset_state(self):
        self.h, self.c = None, None

    def forward(self, xs):
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        N, T, D = xs.shape

        if self.h is None or not self.stateful:
            self.h = np.zeros((N, H), dtype='f')
        if self.c is None or not self.stateful:
            self.c = np.zeros((N, H), dtype='f')

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]
        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = np.zeros((N, H), dtype='f'), np.zeros((N, H), dtype='f')

        for i in range(len(self.grads)):
            self.grads[i][...] = np.zeros_like(self.params[i])

        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                self.grads[i][...] += grad

        self.dh = dh

        return dxs


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.rx = None

    def forward(self, x):
        W, b = self.params
        N, T, H = x.shape
        rx = x.reshape(N*T, -1)
        self.rx = rx

        out = np.dot(rx, W) + b
        return out.reshape((N, T, -1))

    def backward(self, dout):
        W, b = self.params
        N, T, V = dout.shape
        dout = dout.reshape(N*T, -1)
        dW = np.dot(self.rx.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db

        dx = np.dot(dout, W.T)
        return dx.reshape((N, T, -1))


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:
            ts = np.argmax(ts, axis=2)

        mask = (ts != -1).reshape(N*T)
        ts = ts.reshape(N*T)
        rx = xs.reshape(N*T, V)

        rx = rx.T
        rx -= np.max(rx, axis=0)
        ys = np.exp(rx) / np.sum(np.exp(rx), axis=0)
        ys = ys.T

        ls = np.log(ys[np.arange(N*T), ts] + 1e-7)
        ls *= mask
        loss = -np.sum(ls) / mask.sum()

        self.cache = (ys, ts, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ys, ts, mask, (N, T, V) = self.cache
        dout = ys.copy()
        dout[np.arange(N*T), ts] -= 1
        dout /= mask.sum()
        dout *= mask[:, np.newaxis]
        return dout.reshape((N, T, V))


class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, h, enc_hs):
        N, T, H = enc_hs.shape

        hr = h.reshape((N, 1, H)).repeat(T, axis=1)
        t = enc_hs * hr
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s)
        self.cache = (hr, enc_hs)
        return a

    def backward(self, da):
        hr, enc_hs = self.cache
        N, T, H = enc_hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape((N, T, 1)).repeat(H, axis=2)
        dhs = dt * hr
        dhr = dt * enc_hs
        dh = np.sum(dhr, axis=1)

        return dh, dhs


class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, a, enc_hs):
        N, T, H = enc_hs.shape
        ar = a.reshape((N, T, 1)).repeat(H, axis=2)
        t = enc_hs * ar
        c = np.sum(t, axis=1)
        self.cache = (ar, enc_hs)

        return c

    def backward(self, dc):
        ar, enc_hs = self.cache
        N, T, H = enc_hs.shape

        dt = dc.reshape((N, 1, H)).repeat(T, axis=1)
        dhs = dt * ar
        dar = dt * enc_hs
        da = np.sum(dar, axis=2)

        return da, dhs


class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight = AttentionWeight()
        self.weight_sum = WeightSum()
        self.weight = None

    def forward(self, h, enc_hs):
        a = self.attention_weight.forward(h, enc_hs)
        c = self.weight_sum.forward(a, enc_hs)
        self.weight = a
        return c

    def backward(self, dc):
        da, denc_hs1 = self.weight_sum.backward(dc)
        dh, denc_hs2 = self.attention_weight.backward(da)
        denc_hs = denc_hs1 + denc_hs2

        return dh, denc_hs


class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, dec_hs, enc_hs):
        self.layers = []
        self.attention_weights = []
        N, T, H = dec_hs.shape
        out = np.empty((N, T, H), dtype='f')

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(dec_hs[:, t, :], enc_hs)
            self.attention_weights.append(layer.weight)
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, H = dout.shape
        denc_hs = 0
        ddec_hs = np.empty((N, T, H), dtype='f')

        for t in range(T):
            layer = self.layers[t]
            dh, dhs = layer.backward(dout[:, t, :])
            denc_hs += dhs
            ddec_hs[:, t, :] = dh

        return ddec_hs, denc_hs
