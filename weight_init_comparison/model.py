from layers import *

class MultiLayerNet1:
    def __init__(self, input_size, hidden_size_list, output_size, weight_init_std):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size

        activation = {"he": Relu, "xavier": Sigmoid, 0.01: Relu}

        self.params1 = []
        self.params2 = []
        self.__init_weights(weight_init_std)

        self.layers = []
        for idx in range(len(hidden_size_list)):
            self.layers.append(Affine(self.params1[idx], self.params2[idx]))
            self.layers.append(activation[weight_init_std]())

        idx = len(hidden_size_list)
        self.layers.append(Affine(self.params1[idx], self.params2[idx]))
        self.loss_layer = SoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def __init_weights(self, weight_init_std):
        all_size = [self.input_size] + \
            self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size)):
            scale = weight_init_std
            if str(scale) in ("relu", "he"):
                scale = np.sqrt(2.0 / all_size[idx-1])
            elif str(scale) in ("xavier", "sigmoid"):
                scale = np.sqrt(1.0 / all_size[idx-1])

            self.params1.append(
                scale * np.random.randn(all_size[idx-1], all_size[idx]))
            self.params2.append(np.zeros(all_size[idx]))

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout


class MultiLayerNet2:
    def __init__(self, input_size, hidden_size_list, output_size, weight_init_std):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size

        activation = {"he": Relu, "xavier": Sigmoid, 0.01: Relu}

        self.params1 = []
        self.params2 = []
        self.__init_weights(weight_init_std)

        self.layers = []
        for idx in range(len(hidden_size_list)):
            self.layers.append(Affine(self.params1[idx], self.params2[idx]))
            self.layers.append(Relu())

        idx = len(hidden_size_list)
        self.layers.append(Affine(self.params1[idx], self.params2[idx]))
        self.loss_layer = SoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def __init_weights(self, weight_init_std):
        all_size = [self.input_size] + \
            self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size)):
            scale = weight_init_std
            if str(scale) in ("relu", "he"):
                scale = np.sqrt(2.0 / all_size[idx-1])
            elif str(scale) in ("xavier", "sigmoid"):
                scale = np.sqrt(1.0 / all_size[idx-1])

            self.params1.append(
                scale * np.random.randn(all_size[idx-1], all_size[idx]))
            self.params2.append(np.zeros(all_size[idx]))

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout
