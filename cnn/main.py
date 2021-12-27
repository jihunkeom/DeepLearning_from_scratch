import numpy as np
import matplotlib.pyplot as plt
from cnn import CNN
from optimizers import *
from from_book.data import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

model = CNN(28, 2, np.array([1]), 2, 2, 10)
optimizer = SGD()
loss_list = []
for i in range(x_train.shape[0]):
    x = x_train[i].reshape(28, 28)
    t = np.array([t_train[i]])
    loss = model.forward(x, t)
    model.backward(x, t)

    params, grads = model.params, model.grads
    optimizer.update(model.params, model.grads)

    if i % 1000 == 0:
        print("iter: ", i, "loss: ", loss)
        loss_list.append(loss)

x = np.arange(60)
plt.plot(x, loss_list)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 16)
plt.legend()
plt.show()