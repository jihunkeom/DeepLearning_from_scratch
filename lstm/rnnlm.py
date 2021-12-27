import numpy as np
from layers import TimeAffine, TimeEmbedding, TimeLSTM, TimeSoftmaxWithLoss


class RNNlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D) * 0.01).astype('f')
        lstm_Wx = (np.random.randn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (np.random.randn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = (np.zeros(4*H)).astype('f')
        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype('f')
        affine_b = (np.zeros(V)).astype('f')

        self.layers = [TimeEmbedding(embed_W), TimeLSTM(
            lstm_Wx, lstm_Wh, lstm_b, False), TimeAffine(affine_W, affine_b)]
        self.loss_layer = TimeSoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.lstm_layer = self.layers[1]

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)

        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self):
        dout = 1
        dout = self.loss_layer.backward(dout)

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()
