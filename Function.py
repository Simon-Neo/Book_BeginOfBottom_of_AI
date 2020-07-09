import numpy as np

#--------------------------------- Activation Function

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def step_function(x):
    y = x > 0
    return y.astype(np.int)

def softmax(a):
    a_max = np.max(a)
    exp_a = np.exp(a - a_max)
    return exp_a / (np.sum(exp_a))


#--------------------------------- Error

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    if(y.ndim == 1):
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


