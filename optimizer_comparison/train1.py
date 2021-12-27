import numpy as np
import matplotlib.pyplot as plt
from optimizers import SGD, Momentum, RMSProp, AdaGrad, Adam
from model import MultiLayerNet1
from from_book.data import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

max_iter = 2000
opt = {}
opt["SGD"] = SGD()
opt["Momentum"] = Momentum()
opt["RMSProp"] = RMSProp()
opt["AdaGrad"] = AdaGrad()
opt["Adam"] = Adam()

losses = {}
networks = {}

for key in opt.keys():
    losses[key] = []
    networks[key] = MultiLayerNet1(784, [100, 100, 100, 100], 10)

for i in range(max_iter):
    batch = np.random.choice(x_train.shape[0], 128)
    x_batch = x_train[batch]
    t_batch = t_train[batch]

    for key in opt.keys():
        network = networks[key]
        optimizer = opt[key]
        network.gradient(x_batch, t_batch)
        params = network.params
        grads = network.grads
        optimizer.update(params, grads)
        loss = network.loss(x_batch, t_batch)
        losses[key].append(loss)

        if i % 100 == 0:
            print("iteration: " + str(i))
            print(key)
            print(loss)

markers = {"SGD": "o", "Momentum": "x",
           "RMSProp": "s", "AdaGrad": "v", "Adam": "."}
x = np.arange(max_iter)
for key in opt.keys():
    plt.plot(x, losses[key], marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 3)
plt.legend()
plt.show()
