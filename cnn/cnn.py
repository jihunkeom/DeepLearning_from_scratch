import numpy as np
from layers import *

class CNN:
    def __init__(self, input_shape, conv_W, conv_b, stride, pool, hidden1):
        out_h = int((input_shape - conv_W)/stride) + 1
        out_w = int((input_shape - conv_W)/stride) + 1
        pool_out_w = int((out_w - pool)/pool) + 1
        pool_out_h = int((out_h - pool)/pool) + 1


        conv_filter = (np.random.rand(conv_W, conv_W)).astype('f')
        affine_W1 = (np.random.randn(pool_out_h * pool_out_w, hidden1)).astype('f')
        affine_b1 = (np.random.randn(hidden1)).astype('f')
        affinw_W2 = (np.random.randn(hidden1, 10)).astype('f')
        affine_b2 = (np.random.randn(10)).astype('f')

        self.layers = [
                       Convolution(conv_filter, conv_b, stride),
                       Relu(),
                       Pooling(pool),
                       Affine(affine_W1, affine_b1),
                       Relu(),
                       Affine(affinw_W2, affine_b2)
        ]
        self.loss_layer = SoftmaxWithLoss()
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x, t):
        for layer in self.layers:
            x = layer.forward(x)

        loss = self.loss_layer.forward(x, t)

        return loss

    def backward(self, x, t):
        dout = 1
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return None
