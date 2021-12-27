import numpy as np
from layers import *


class Seq2seq:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.loss_layer = TimeSoftmaxWithLoss()
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        dec_xs, dec_ts = ts[:, :-1], ts[:, 1:]
        h = self.encoder.forward(xs)
        score = self.decoder.forward(dec_xs, h)
        loss = self.loss_layer.forward(score, dec_ts)
        return loss

    def backward(self):
        dout = 1
        dout = self.loss_layer.backward(dout)
        dhs = self.decoder.backward(dout)
        dxs = self.encoder.backward(dhs)
        return dxs

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled


class PeekySeq2seq:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.loss_layer = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        dec_xs, dec_ts = ts[:, :-1], ts[:, 1:]
        h = self.encoder.forward(xs)
        score = self.decoder.forward(dec_xs, h)
        loss = self.loss_layer.forward(score, dec_ts)
        return loss

    def backward(self):
        dout = 1
        dout = self.loss_layer.backward(dout)
        dh = self.decoder.backward(dout)
        dxs = self.encoder.backward(dh)
        return dxs

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
