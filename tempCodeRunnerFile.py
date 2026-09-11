    x = x_test[i].reshape(784, 1)
        label = y_test[i]

        output = nn.forward(x)
        prediction = np.argmax(output)

        print("Actual:", label, "Predicted:", prediction)
