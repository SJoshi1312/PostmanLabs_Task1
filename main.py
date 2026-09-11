from data.MNIST import x_train, y_train, x_test, y_test
from src.network import NeuralNetwork
from src.loss import cross_entropy
import numpy as np

nn = NeuralNetwork()

learning_rate = 0.001
print("Initial W3 norm:", np.linalg.norm(nn.W3))
for i in range(1000):

    x = x_train[i].reshape(784, 1)
    label = y_train[i]

    # Forward pass
    output = nn.forward(x)

    # Loss
    loss = cross_entropy(label, output)

    # Backpropagation
    gradients = nn.backward(label)

    # Update weights
    nn.update_parameters(gradients, learning_rate)

    # Debug first 10 predictions


    if i < 10:
        print("Image:", i, "Label:", label, "Prediction:", np.argmax(output))

    if i % 100 == 0:
        print(f"Image {i}, Loss: {loss:.4f}")

    gradients = nn.backward(label)

    if i == 0:
        print("dW3 norm:", np.linalg.norm(gradients[4]))
        print("db3:", gradients[5].flatten())

    nn.update_parameters(gradients, learning_rate)
print("Final W3 norm:", np.linalg.norm(nn.W3))


