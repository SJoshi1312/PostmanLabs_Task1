import numpy as np
from data.MNIST import x_train, y_train, x_test, y_test
import sys
sys.path.append("src")

from network import NeuralNetwork
from src.loss import cross_entropy



nn = NeuralNetwork()
x = x_train[0].reshape(784, 1)
output = nn.forward(x)
loss = cross_entropy(y_train[0], output)

print("Loss:", loss)




print(output)
print("Predicted:", np.argmax(output))
print("Actual:", y_train[0])