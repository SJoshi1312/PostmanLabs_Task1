import numpy as np

from src.network import NeuralNetwork, relu, softmax
from src.backprop import relu_derivative
from src.loss import cross_entropy


def test_relu():
    x = np.array([-2, -1, 0, 1, 2])
    expected = np.array([0, 0, 0, 1, 2])

    assert np.array_equal(relu(x), expected)


def test_relu_derivative():
    x = np.array([-2, -1, 0, 1, 2])
    expected = np.array([0, 0, 0, 1, 1])

    assert np.array_equal(relu_derivative(x), expected)


def test_softmax():
    x = np.random.randn(10, 5)

    output = softmax(x)

    # Probabilities must be non-negative
    assert np.all(output >= 0)

    # Probabilities for each image must sum to 1
    assert np.allclose(np.sum(output, axis=0), 1)


def test_loss():
    y_true = np.array([2])

    y_pred = np.array([
        [0.1],
        [0.1],
        [0.5],
        [0.1],
        [0.2]
    ])

    loss = cross_entropy(y_true, y_pred)

    assert np.isclose(loss, -np.log(0.5))


def test_network_shapes():

    nn = NeuralNetwork()

    # 5 images
    x = np.random.randn(784, 5)

    output = nn.forward(x)

    assert output.shape == (10, 5)

    labels = np.array([0, 1, 2, 3, 4])

    gradients = nn.backward(labels)

    dW1, db1, dW2, db2, dW3, db3 = gradients

    assert dW1.shape == nn.W1.shape
    assert db1.shape == nn.b1.shape

    assert dW2.shape == nn.W2.shape
    assert db2.shape == nn.b2.shape

    assert dW3.shape == nn.W3.shape
    assert db3.shape == nn.b3.shape


def test_forward_probabilities():

    nn = NeuralNetwork()

    x = np.random.randn(784, 5)

    output = nn.forward(x)

    assert np.all(output >= 0)
    assert np.allclose(np.sum(output, axis=0), 1)

def test_gradients():

    nn = NeuralNetwork()

    # Small input batch
    x = np.random.randn(784, 2)
    labels = np.array([3, 7])

    # Get analytical gradients from backpropagation
    nn.forward(x)
    analytical = nn.backward(labels)

    dW1, db1, dW2, db2, dW3, db3 = analytical

    epsilon = 1e-5

    # Check a few randomly selected weights
    for W, dW in [
        (nn.W1, dW1),
        (nn.W2, dW2),
        (nn.W3, dW3)
    ]:

        for _ in range(5):

            i = np.random.randint(W.shape[0])
            j = np.random.randint(W.shape[1])

            original = W[i, j]

            # f(w + epsilon)
            W[i, j] = original + epsilon
            plus = cross_entropy(labels, nn.forward(x))

            # f(w - epsilon)
            W[i, j] = original - epsilon
            minus = cross_entropy(labels, nn.forward(x))

            # Restore original value
            W[i, j] = original

            # Numerical gradient
            numerical = (plus - minus) / (2 * epsilon)

            # Analytical gradient
            analytical_gradient = dW[i, j]

            assert np.isclose(
                numerical,
                analytical_gradient,
                rtol=1e-3,
                atol=1e-5
            ), (
                f"Gradient mismatch: "
                f"numerical={numerical}, "
                f"analytical={analytical_gradient}"
            )

def run_tests():

    tests = [
    ("ReLU", test_relu),
    ("ReLU derivative", test_relu_derivative),
    ("Softmax", test_softmax),
    ("Loss", test_loss),
    ("Network shapes", test_network_shapes),
    ("Forward probabilities", test_forward_probabilities),
    ("Gradients", test_gradients)
]

    passed = 0

    for name, test in tests:

        try:
            test()
            print(f"{name}: PASS")
            passed += 1

        except AssertionError:
            print(f"{name}: FAIL")

        except Exception as e:
            print(f"{name}: ERROR")
            print(f"    {e}")

    print()
    print(f"{passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()