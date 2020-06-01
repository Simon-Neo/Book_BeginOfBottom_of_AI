
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

from PIL import Image

import pickle

def img_show(img):
    img = img.astype(np.uint8)
    pil_img = Image.fromarray(img)
    pil_img.show()

def step_function(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def softmax(a):
    a_max = np.max(a)
    exp_a = np.exp(a - a_max)
    return exp_a / (np.sum(exp_a))

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2 , W3) + b3
    y = softmax(a3)
    return y

# ============================================ MNIST
def get_data():
    # Mnist Data import
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=True)
    return (x_test, t_test)#, (x_test, t_test)

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]

(x_train, t_train) = get_data()

dic_network = init_network()
accuracy_cnt = 0
batch_size = 10
train_batch_size = 100

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
train_batch_x = None
train_batch_t = None
train_batch_y = None

# for i in range(0, train_size, train_batch_size):
#     train_batch_x = x_train[i : train_batch_size]
#     train_batch_t = t_train[i : train_batch_size]
#     train_batch_y = forward(dic_network, train_batch_x)


batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_batch.shape)
print(t_batch.shape)


test = np.array([[11, 22,33  ,44], [4 ,5 ,7 ,6]])
print('8888888888')
print(len(test))
print(test.shape)
print(test.size)

print(np.arange(len(test)))
test1 = test[np.arange(len(test)) , 3]

print(type(test1))
print(test1.shape)

print(test1)

print('TTTTTTTTTTTTT')
print(len(test1))
print(test1.size)




# print(t_test.shape)
#
# x_batch = None
# y_batch = None
# correct_batch = None
# result_batch = None
# for i in range(0, len(x_test), batch_size):
#     x_batch = x_test[i : i + batch_size]
#     y_batch = forward(dic_network, x_batch)
#     result_batch = y_batch.argmax(axis=1)
#
#     accuracy_cnt += np.sum(t_test[i : i + batch_size] == result_batch)
#
# print('AccuracyPercent = ', ((float(accuracy_cnt) / len(x_test) ) * 100) , "%")
#

# i = 0
# for i, rabel in enumerate(t_test):
#     result = forward(dic_network, x_test[i])
#     maxIndex = np.argmax(result)
#     if rabel == maxIndex:
#         accuracy_cnt += 1

# result = None
# result_Max = 0
# resultNum = None
#
# w_list = np.arange(-10, 11, 1)
#
# x = np.array([2, 4, 8])
# y = np.array([5, 12, 21])
#
# Cost = []
#
# temp = 0
# sum = 0
# for w in w_list:
#     temp = ((w* x) - y)
#     print(temp)
#     temp = temp ** 2
#     sum = np.sum(temp)
#     sum /= len(x)
#
#     Cost.append(sum)
#
#
# print(len(w_list))

# cost = np.array(Cost)
#
# for i , data in enumerate(cost):
#     print(i , data)
# plt.plot(w_list, Cost)
# plt.ylim(-0.1, 15)
# plt.show()
