import numpy as np
from layers import Embedding, NegativeSamplingLoss


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus, sample_size):
        V, D = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, D)
        W_out = 0.01 * np.random.randn(V, D)

        self.window_size = window_size
        self.sample_size = sample_size

        self.in_layers = [Embedding(W_in) for _ in range(2*window_size)]
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, sample_size)

        self.layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, context, target):
        h = 0
        for i in range(2*self.window_size):
            h += self.in_layers[i].forward(context[:, i])
        h /= 2*self.window_size
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dh = self.ns_loss.backward(dout)
        dh /= 2*self.window_size
        for layer in self.in_layers:
            layer.backward(dh)

        return None
