import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] = params[i] - self.lr * grads[i]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for val in params:
                self.v.append(np.zeros_like(val))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]


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


class RMSProp:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.g = None

    def update(self, params, grads):
        if self.g is None:
            self.g = []
            for val in params:
                self.g.append(np.zeros_like(val))

        for i in range(len(params)):
            self.g[i] = self.lr*self.g[i] + (1-self.lr)*(grads[i]*grads[i])
            params[i] -= self.lr*grads[i]/np.sqrt(self.g[i] + 1e-7)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        self.iter += 1

        if self.m is None:
            self.m = []
            for val in params:
                self.m.append(np.zeros_like(val))

        if self.v is None:
            self.v = []
            for val in params:
                self.v.append(np.zeros_like(val))

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + \
                (1 - self.beta2) * (grads[i] * grads[i])
            m_hat = self.m[i] / (1 - (self.beta1**self.iter))
            v_hat = self.v[i] / (1 - (self.beta2**self.iter))
            params[i] -= self.lr * m_hat/(np.sqrt(v_hat) + 1e-7)
