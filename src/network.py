import numpy as np
from src.backprop import output_delta, relu_derivative, backprop_layer

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)

class NeuralNetwork:

    def __init__(self):

        self.W1 = np.random.randn(128, 784) * np.sqrt(2 / 784)
        self.b1 = np.zeros((128, 1))

        self.W2 = np.random.randn(64, 128) * np.sqrt(2 / 128)
        self.b2 = np.zeros((64, 1))

        self.W3 = np.random.randn(10, 64) * np.sqrt(2 / 64)
        self.b3 = np.zeros((10, 1))
    def forward(self, x):

        z1 = self.W1 @ x + self.b1
        a1 = relu(z1)

        z2 = self.W2 @ a1 + self.b2
        a2 = relu(z2)

        z3 = self.W3 @ a2 + self.b3
        a3 = softmax(z3)

        self.x = x

        self.z1 = z1
        self.a1 = a1

        self.z2 = z2
        self.a2 = a2

        self.z3 = z3
        self.a3 = a3
        return a3
    def backward(self, label):
        delta3 = output_delta(self.a3, label)

        dW3, db3, da2 = backprop_layer(
            self.W3,
            self.a2,
            delta3
        )

        # Hidden layer 2
        delta2 = da2 * relu_derivative(self.z2)

        dW2, db2, da1 = backprop_layer(
            self.W2,
            self.a1,
            delta2
        )

        # Hidden layer 1
        delta1 = da1 * relu_derivative(self.z1)

        dW1, db1, da0 = backprop_layer(
            self.W1,
            self.x,
            delta1
        )

        return dW1, db1, dW2, db2, dW3, db3
    def update_parameters(self, gradients, learning_rate=0.01):

        dW1, db1, dW2, db2, dW3, db3 = gradients

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3