import numpy as np
from partials import *


class AttentionSeq2seq:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(V, D, H)
        self.decoder = AttentionDecoder(V, D, H)
        self.loss_layer = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        dec_xs, dec_ts = ts[:, :-1], ts[:, 1:]
        enc_hs = self.encoder.forward(xs)
        score = self.decoder.forward(dec_xs, enc_hs)
        loss = self.loss_layer.forward(score, dec_ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        dout = self.decoder.backward(dout)
        dout = self.encoder.backward(dout)

        return dout

    def generate(self, xs, start_id, sample_size):
        enc_hs = self.encoder.forward(xs)
        sampled = self.decoder.generate(enc_hs, start_id, sample_size)
        return sampled
