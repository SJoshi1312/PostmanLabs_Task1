import numpy as np

from data.MNIST import x_train, y_train, x_test, y_test
from src.network import NeuralNetwork
from src.loss import cross_entropy


nn = NeuralNetwork()

batch_size = 32
epochs = 5

for epoch in range(epochs):

    # Shuffle the training data
    indices = np.random.permutation(len(x_train))

    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]

    total_loss = 0

    for i in range(0, len(x_train), batch_size):

        x_batch = x_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        x_batch = x_batch.T

        predictions = nn.forward(x_batch)

        loss = cross_entropy(y_batch, predictions)
        total_loss += loss

        gradients = nn.backward(y_batch)

        nn.update_parameters(gradients, learning_rate=0.001)

    average_loss = total_loss / (len(x_train) // batch_size)

    print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")

    correct = 0

for i in range(0, len(x_test), batch_size):

    x_batch = x_test[i:i + batch_size].T
    y_batch = y_test[i:i + batch_size]

    predictions = nn.forward(x_batch)

    predicted_labels = np.argmax(predictions, axis=0)

    correct += np.sum(predicted_labels == y_batch)

accuracy = correct / len(x_test)

print(f"Test accuracy: {accuracy:.4f}")

confusion_matrix = np.zeros((10, 10), dtype=int)

for i in range(0, len(x_test), batch_size):

    x_batch = x_test[i:i + batch_size].T
    y_batch = y_test[i:i + batch_size]

    predictions = nn.forward(x_batch)
    predicted_labels = np.argmax(predictions, axis=0)

    for actual, predicted in zip(y_batch, predicted_labels):
        confusion_matrix[actual, predicted] += 1

print("\nConfusion Matrix:")
print(confusion_matrix)
print("In above matrix, rows correspond to Actual values, and columns correspond to Predicted values. As seen, majority of the numbers fall on the diagonal, i.e., actual = predicted. ")