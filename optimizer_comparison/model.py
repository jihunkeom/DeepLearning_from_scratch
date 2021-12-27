import numpy as np
from layers import Affine, Relu, SoftmaxWithLoss


class MultiLayerNet1:
    def __init__(self, input_size, hidden_size_list, output_size):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size

        self.params1 = []
        self.params2 = []

        self.__init_weights()
        self.layers = []
        for idx in range(len(self.params1)-1):
            self.layers.append(Affine(self.params1[idx], self.params2[idx]))
            self.layers.append(Relu())

        idx = len(self.params1)-1
        self.layers.append(Affine(self.params1[idx], self.params2[idx]))
        self.loss_layer = SoftmaxWithLoss()

        self.params = []
        for i in range(len(self.params1)):
            self.params.append(self.params1[i])
            self.params.append(self.params2[i])

        self.grads = []
        for layer in self.layers:
            self.grads += layer.grads

    def __init_weights(self):
        all_size = [self.input_size] + \
            self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size)):
            self.params1.append(
                0.01 * np.random.randn(all_size[idx-1], all_size[idx]))
            self.params2.append(np.zeros(all_size[idx]))

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.loss_layer.forward(y, t)
        return loss

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.loss_layer.backward(dout)
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)

        return None


class MultiLayerNet2:
    def __init__(self, input_size, hidden_size_list, output_size):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size

        self.params1 = []
        self.params2 = []

        self.__init_weights()
        self.layers = []
        for idx in range(len(self.params1)-1):
            self.layers.append(Affine(self.params1[idx], self.params2[idx]))
            self.layers.append(Relu())

        idx = len(self.params1)-1
        self.layers.append(Affine(self.params1[idx], self.params2[idx]))
        self.loss_layer = SoftmaxWithLoss()

        self.params = []
        for i in range(len(self.params1)):
            self.params.append(self.params1[i])
            self.params.append(self.params2[i])

        self.grads = []
        for layer in self.layers:
            self.grads += layer.grads

    def __init_weights(self):
        all_size = [self.input_size] + \
            self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size)):
            self.params1.append(np.sqrt(
                2.0 / all_size[idx-1]) * np.random.randn(all_size[idx-1], all_size[idx]))
            self.params2.append(np.zeros(all_size[idx]))

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.loss_layer.forward(y, t)
        return loss

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.loss_layer.backward(dout)
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)

        return None
