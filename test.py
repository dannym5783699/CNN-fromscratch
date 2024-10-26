import numpy as np
from feed_forward_neural_network import (
    FeedforwardNeuralNetwork, sigmoid, sigmoid_derivative
)

if __name__ == "__main__":
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = FeedforwardNeuralNetwork([2, 2, 1])

    nn.train(
        x, y,
        epochs=10000,
        learning_rate=0.1,
        activation_function=sigmoid,
        activation_derivative=sigmoid_derivative
    )

    print("\nTesting Trained Network:")
    for xi in x:
        output = nn.forward(xi, sigmoid)
        print(f"Input: {xi} -> Output: {output}")
