import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)

class NeuralNetwork:

    def __init__(self):
        self.W1 = np.random.randn(128, 784) * 0.01
        self.b1 = np.zeros((128, 1))

        self.W2 = np.random.randn(64, 128) * 0.01
        self.b2 = np.zeros((64, 1))

        self.W3 = np.random.randn(10, 64) * 0.01
        self.b3 = np.zeros((10, 1))
    def forward(self, x):

        z1 = self.W1 @ x + self.b1
        a1 = relu(z1)

        z2 = self.W2 @ a1 + self.b2
        a2 = relu(z2)

        z3 = self.W3 @ a2 + self.b3
        a3 = softmax(z3)

        return a3