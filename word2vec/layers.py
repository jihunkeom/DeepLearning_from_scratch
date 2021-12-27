import numpy as np
from sampler import UnigramSampler

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


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, x, t):
        y = 1 / (1 + np.exp(-x))

        # if y.size == t.size:
        #     t = np.argmax(t, axis=1)

        if y.ndim == 1:
            y = y.reshape(1, -1)
            t = t.reshape(1, -1)

        batch_size = y.shape[0]
        loss = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
        self.cache = (y, t, batch_size)
        return loss

    def backward(self, dout=1):
        y, t, batch_size = self.cache
        y[np.arange(batch_size), t] -= 1
        y /= batch_size
        return y


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, x, t):
        y = 1 / (1 + np.exp(-x))

        tmp = np.c_[1-y, y]

        if tmp.ndim == 1:
            tmp = tmp.reshape(1, -1)
            t = t.reshape(1, -1)

        if tmp.size == t.size:
            t = np.argmax(t, axis=1)

        batch_size = y.shape[0]
        loss = - \
            np.sum(np.log(tmp[np.arange(batch_size), t] + 1e-7)) / batch_size
        self.cache = (y, t, batch_size)
        return loss

    def backward(self, dout=1):
        y, t, batch_size = self.cache
        dout = (y - t) / batch_size

        return dout


class NegativeSamplingLoss:
    def __init__(self, W, corpus, sample_size):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, sample_size)

        self.embed_dot_layers = [EmbeddingDot(
            W) for _ in range(sample_size + 1)]
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        ns = self.sampler.get_negative_samples(target)
        batch_size = target.shape[0]

        label = np.ones(batch_size, dtype=np.int32)
        score = self.embed_dot_layers[0].forward(h, target)
        loss = self.loss_layers[0].forward(score, label)

        label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            target = ns[:, i]
            score = self.embed_dot_layers[1+i].forward(h, target)
            loss += self.loss_layers[1+i].forward(score, label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l1, l2 in zip(self.embed_dot_layers, self.loss_layers):
            dscore = l2.backward(dout)
            dh += l1.backward(dscore)

        return dh
