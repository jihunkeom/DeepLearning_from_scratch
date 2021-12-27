import numpy as np
import matplotlib.pyplot as plt
from from_book.data import load_mnist
from optimizers import SGD
from model import MultiLayerNet1

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

optimizer = SGD()

models = {
    0.01: MultiLayerNet1(784, [100, 100, 100, 100], 10, 0.01),
    "he": MultiLayerNet1(784, [100, 100, 100, 100], 10, "he"),
    "xavier": MultiLayerNet1(784, [100, 100, 100, 100], 10, "xavier")

}

loss_list = {
    0.01: [],
    "he": [],
    "xavier": []
}

for i in range(2000):
    batch = np.random.choice(x_train.shape[0], 128)
    x_batch = x_train[batch]
    t_batch = t_train[batch]
    for key in models.keys():
        models[key].loss(x_batch, t_batch)
        models[key].backward()
        params, grads = models[key].params, models[key].grads
        optimizer.update(params, grads)
        if i % 100 == 0:
            loss = models[key].loss(x_batch, t_batch)
            loss_list[key].append(loss)
            print(key)
            print(loss)

markers = {0.01: "o", "he": "x", "xavier": "s"}
x = np.arange(20)
for key in models.keys():
    plt.plot(x, loss_list[key], marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 3)
plt.legend()
plt.show()
