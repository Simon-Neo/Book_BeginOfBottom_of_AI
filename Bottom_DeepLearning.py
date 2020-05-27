
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
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

(x_test, t_test) = get_data()

dic_network = init_network()

accuracy_cnt = 0
result = None
result_Max = 0
resultNum = None

i = 0
for i, rabel in enumerate(t_test):
    result = forward(dic_network, x_test[i])
    maxIndex = np.argmax(result)
    if rabel == maxIndex:
        accuracy_cnt += 1

print('AccuracyPercent = ', (float(accuracy_cnt) / (i + 1) * 100) , "%")
