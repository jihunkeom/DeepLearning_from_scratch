import numpy as np


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
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
            self.v = []
            for val in params:
                self.m.append(np.zeros_like(val))
                self.v.append(np.zeros_like(val))

        for i in range(len(params)):
            self.m[i] = (self.beta1 * self.m[i]) + ((1-self.beta1)*grads[i])
            self.v[i] = (self.beta2 * self.v[i]) + \
                ((1-self.beta2)*(grads[i]*grads[i]))
            m_adj = self.m[i] / (1 - self.beta1**self.iter)
            v_adj = self.v[i] / (1 - self.beta2**self.iter)
            params[i] -= self.lr * m_adj / np.sqrt(v_adj + 1e-7)
