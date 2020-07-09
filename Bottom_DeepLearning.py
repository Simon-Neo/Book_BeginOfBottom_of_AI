
import numpy as np
import matplotlib.pyplot as plt


import sys, os
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
import pickle
from TwoLayerNet import TwoLayerNet

from Function import sigmoid, softmax, cross_entropy_error, mean_squared_error
from Gradient import numerical_gradient, gradient_descent

from PIL import Image



# ============================================ MNIST
def get_data():
    # Mnist Data import
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=True)
    return (x_train, t_train)#, (x_test, t_test)

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

# ------------------------------------------------------ main

def mean_squared_error(y, t):
    return 0.5 * np.sum((t - y) ** 2)


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= grad * lr

    return x

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

import math
if(__name__ == '__main__'):

    # (x_train, t_train) , (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    #
    # iters_num = 10000
    # lerning_rate = 0.1
    #
    # train_size = x_train.shape[0]
    # batch_size = 100
    #
    # train_loss_list = []
    # train_acc_list = []
    # test_acc_list = []
    #
    # iter_per_epoch = max(train_size / batch_size ,1)
    #
    # net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    #
    # for i in range(iters_num):
    #     # batch
    #     batch_mask = np.random.choice(train_size, batch_size)
    #     x_batch = x_train[batch_mask]
    #     t_batch = t_train[batch_mask]
    #
    #     # Gradient
    #     grad = net.numerical_gradient(x_batch, t_batch)
    #
    #     for key in ('W1', 'B1', 'W2', 'B2'):
    #         net.params[key] -= lerning_rate * grad[key]
    #
    #     loss = net.loss(x_batch, t_batch)
    #     train_loss_list.append(loss)
    #
    # print(train_loss_list)
    pass


class Monster():
    def __init__(self, name, att):
        self.name = name
        self.att = att
        self.hp = 100

    def get_hp(self):
        return self.hp
    def get_name(self):
        return self.name

    def __del__(self):
        print("I'm Die.. ㅜㅠ")

class Goblin(Monster):
    def __init__(self, name, att):
        Monster.__init__(self, name, att)

    def __del__(self):
        Monster.__del__(self)



x = ['Abc', 'EGD', 'ESFD']

search_word = input('What are you looling for the Word? : ')


try:
    result = x.index(search_word)
except ValueError as v_error:
    print(v_error)
else:
    print(result)
finally:
    print('interllect')






