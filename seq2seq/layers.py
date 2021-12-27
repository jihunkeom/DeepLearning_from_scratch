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


class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (0.01 * np.random.randn(V, D)).astype('f')
        lstm_Wx = (np.random.randn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (np.random.randn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = (np.zeros(4*H)).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]

    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh
        dxs = self.lstm.backward(dhs)
        dxs = self.embed.backward(dxs)
        return dxs


class Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (0.01 * np.random.randn(V, D)).astype('f')
        lstm_Wx = (np.random.randn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (np.random.randn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = (np.zeros(4*H)).astype('f')
        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype('f')
        affine_b = (np.zeros(V)).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, True),
            TimeAffine(affine_W, affine_b)
        ]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.lstm_layer = self.layers[1]

    def forward(self, xs, enc_h):
        self.lstm_layer.set_state(enc_h)
        for layer in self.layers:
            xs = layer.forward(xs)

        return xs

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        dh = self.lstm_layer.dh
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm_layer.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            for layer in self.layers:
                x = layer.forward(x)
            sample_id = np.argmax(x.flatten())
            sampled.append(int(sample_id))

        return sampled


class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (0.01 * np.random.randn(V, D)).astype('f')
        lstm_Wx = (np.random.randn(H+D, 4*H) / np.sqrt(H+D)).astype('f')
        lstm_Wh = (np.random.randn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = (np.zeros(4*H)).astype('f')
        affine_W = (np.random.randn(2*H, V) / np.sqrt(2*H)).astype('f')
        affine_b = (np.zeros(V)).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params = self.embed.params + self.lstm.params + self.affine.params
        self.grads = self.embed.grads + self.lstm.grads + self.affine.grads
        self.H = None

    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape
        self.H = H
        self.lstm.set_state(h)

        hs = h.reshape((N, 1, H)).repeat(T, axis=1)

        xs = self.embed.forward(xs)
        out = np.concatenate((hs, xs), axis=2)
        out = self.lstm.forward(out)
        out = np.concatenate((hs, out), axis=2)
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dhs1, dout = dout[:, :, :self.H], dout[:, :, self.H:]
        dout = self.lstm.backward(dout)
        dhs2, dout = dout[:, :, :self.H], dout[:, :, self.H:]
        dhs = dhs1 + dhs2
        dxs = self.embed.backward(dout)
        dhs = np.sum(dhs, axis=1) + self.lstm.dh
        return dhs

    def generate(self, h, start_id, sample_size):
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            char_id = np.argmax(score.flatten())
            sampled.append(char_id)

        return sampled
