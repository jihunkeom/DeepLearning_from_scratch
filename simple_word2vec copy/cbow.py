import numpy as np
from layers import *

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size, window_size):
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H)
        W_out = 0.01 * np.random.randn(H, V)
        self.in_layers = []
        for _ in range(window_size * 2):
            self.in_layers.append(Embedding(W_in))
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = self.in_layers + [self.out_layer]
        self.params, self.grads = [], []

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i in range(len(self.in_layers)):
            h += self.in_layers[i].forward(contexts[:, i])
        h /= 2 * len(self.in_layers)
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self):
        dout = 1
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 2 * len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(da)

        return None