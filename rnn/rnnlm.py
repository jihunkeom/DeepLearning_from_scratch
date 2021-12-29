import numpy as np
from layers import *

class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = 0.01 * np.random.randn(V, D)
        rnn_Wx = np.random.randn(D, H) / np.sqrt(D)
        rnn_Wh = np.random.randn(H, H) / np.sqrt(H)
        rnn_b = np.zeros(H)
        Affine_W = np.random.randn(H, V) / np.sqrt(H)
        Affine_b = np.zeros(V)

        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b),
            TimeAffine(Affine_W, Affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)

        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout