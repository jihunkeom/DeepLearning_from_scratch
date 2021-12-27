import numpy as np
from layers import Affine, BatchNormalization, Relu, SoftmaxWithLoss


class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size, std):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        all_size = [input_size] + hidden_size_list + [output_size]

        self.params1, self.params2, self.params3, self.params4 = [], [], [], []
        for idx in range(1, len(all_size)):
            self.params1.append(np.random.randn(
                all_size[idx-1], all_size[idx]) * std)
            self.params2.append(np.zeros(all_size[idx]))
            self.params3.append(np.ones(all_size[idx]))
            self.params4.append(np.zeros(all_size[idx]))

        self.layers = []
        for idx in range(len(self.params1) - 1):
            self.layers.append(Affine(self.params1[idx], self.params2[idx]))
            self.layers.append(BatchNormalization(
                self.params3[idx], self.params4[idx]))
            self.layers.append(Relu())

        idx = len(self.params1) - 1
        self.layers.append(Affine(self.params1[idx], self.params2[idx]))

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.loss_layer = SoftmaxWithLoss()

    def predict(self, x, train):
        for layer in self.layers:
            if layer.name == "BatchNorm":
                x = layer.forward(x, train)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t):
        score = self.predict(x, True)
        return self.loss_layer.forward(score, t)

    def accuracy(self, X, T):
        Y = self.predict(X, False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1:
            T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def backward(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.loss_layer.backward(dout)
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)

        return None


class MultiLayerNet2:
    def __init__(self, input_size, hidden_size_list, output_size, std):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        all_size = [input_size] + hidden_size_list + [output_size]

        self.params1, self.params2 = [], []
        for idx in range(1, len(all_size)):
            self.params1.append(np.random.randn(
                all_size[idx-1], all_size[idx]) * std)
            self.params2.append(np.zeros(all_size[idx]))

        self.layers = []
        for idx in range(len(self.params1) - 1):
            self.layers.append(Affine(self.params1[idx], self.params2[idx]))
            self.layers.append(Relu())

        idx = len(self.params1) - 1
        self.layers.append(Affine(self.params1[idx], self.params2[idx]))

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.loss_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        score = self.predict(x)
        return self.loss_layer.forward(score, t)

    def accuracy(self, X, T):
        Y = self.predict(X)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1:
            T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def backward(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return None
