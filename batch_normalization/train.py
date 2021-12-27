import numpy as np
import matplotlib.pyplot as plt
from from_book.data import load_mnist
from model import MultiLayerNet, MultiLayerNet2
from optimizer import SGD


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train, t_train = x_train[:1000], t_train[:1000]

gamma = np.zeros(784)
beta = np.zeros(784)
std = 0.0158489319246
models = {
    "BN": MultiLayerNet(784, [100, 100, 100, 100], 10, std),
    "Vanilla": MultiLayerNet2(784, [100, 100, 100, 100], 10, std)
}
optimizer = SGD()
acc_list = {}
acc_list["BN"] = []
acc_list["Vanilla"] = []
max_epochs = 21
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    if epoch_cnt >= max_epochs:
        break
    batch = np.random.choice(x_train.shape[0], 128)
    x_batch = x_train[batch]
    t_batch = t_train[batch]

    loss = models["BN"].loss(x_batch, t_batch)
    models["BN"].backward(x_batch, t_batch)
    params, grads = models["BN"].params, models["BN"].grads
    optimizer.update(params, grads)

    if i % iter_per_epoch == 0:
        print(epoch_cnt)
        acc = models["BN"].accuracy(x_batch, t_batch)
        acc_list["BN"].append(acc)
        print(acc)
        epoch_cnt += 1


epoch_cnt = 0
for i in range(1000000000):
    if epoch_cnt >= max_epochs:
        break
    batch = np.random.choice(x_train.shape[0], 128)
    x_batch = x_train[batch]
    t_batch = t_train[batch]

    loss = models["Vanilla"].loss(x_batch, t_batch)
    models["Vanilla"].backward(x_batch, t_batch)
    params, grads = models["Vanilla"].params, models["Vanilla"].grads
    optimizer.update(params, grads)

    if i % iter_per_epoch == 0:
        print(epoch_cnt)
        acc = models["Vanilla"].accuracy(x_batch, t_batch)
        acc_list["Vanilla"].append(acc)
        print(acc)
        epoch_cnt += 1

markers = {"BN": "o", "Vanilla": "x"}
x = np.arange(21)
for key in markers.keys():
    plt.plot(x, acc_list[key], marker=markers[key], label=key)
plt.xlabel("epochs")
plt.ylabel("acc")
plt.ylim(0, 1)
plt.legend()
plt.show()
