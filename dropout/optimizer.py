import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.g = None

    def update(self, params, grads):
        if self.g is None:
            self.g = []
            for val in params:
                self.g.append(np.zeros_like(val))

        for i in range(len(params)):
            self.g[i] += grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / np.sqrt(self.g[i] + 1e-7)
