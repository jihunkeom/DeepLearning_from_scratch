import numpy as np
import matplotlib.pyplot as plt
from cnn import CNN
from optimizers import *
from from_book.data import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
model = CNN(28, 2, np.array([1]), 2, 2, 10)
optimizer = SGD()
loss_list = []
train_acc_list = []
test_acc_list = []
train_acc = 0
test_acc = 0

for i in range(x_train.shape[0]):
    x = x_train[i].reshape(28, 28)
    t = np.array([t_train[i]])
    loss = model.forward(x, t)
    model.backward(x, t)
    train_acc += model.accuracy(x, t)
    test_acc += model.accuracy(x_test[i].reshape((28,28)), t_test[i])

    params, grads = model.params, model.grads
    optimizer.update(model.params, model.grads)

    if i % 1000 == 0:
        print("iter: ", i, "loss: ", loss)
        loss_list.append(loss)
        train_acc /= 1000
        test_acc /= 1000
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc:" + str(train_acc) + ", test acc:" + str(test_acc))