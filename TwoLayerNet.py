import numpy as np

from Function import sigmoid, softmax, cross_entropy_error, mean_squared_error
from Gradient import numerical_gradient, gradient_descent

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['B1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['B2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        B1, B2 = self.params['B1'], self.params['B2']

        a1 = np.dot(x, W1) + B1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + B2
        return softmax(a2)

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        func_loss = lambda W : self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(func_loss, self.params['W1'])
        grads['B1'] = numerical_gradient(func_loss, self.params['B1'])
        grads['W2'] = numerical_gradient(func_loss, self.params['W2'])
        grads['B2'] = numerical_gradient(func_loss, self.params['B2'])

        return grads