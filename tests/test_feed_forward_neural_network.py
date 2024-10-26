import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from midterm_nueralnetworks.feed_forward_neural_network import (
    FeedforwardNeuralNetwork, relu, relu_derivative, sigmoid, sigmoid_derivative
)


class TestFeedforwardNeuralNetwork(unittest.TestCase):

    def setUp(self):
        """
        Load and preprocess the Iris dataset to be used for testing.
        This includes one-hot encoding of labels, normalization, and dataset splitting.
        """
        iris = load_iris()
        X = iris.data  # Input features (4 features)
        y = iris.target.reshape(-1, 1)  # Target labels (3 classes)

        # One-hot encode the target labels
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize the data
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        # Initialize the neural network
        self.nn = FeedforwardNeuralNetwork([4, 5, 3])  # 4 input, 5 hidden, 3 output nodes

    def test_forward_pass(self):
        """
        Test the forward pass on a sample input to check if the output has the correct shape.
        """
        sample_input = self.X_train[0]
        output = self.nn.forward(sample_input, sigmoid)

        # Expected output shape should match the number of neurons in the output layer
        self.assertEqual(output.shape, (3,))
        print("Forward pass output shape test passed.")

    def test_backward_pass(self):
        """
        Test the backward pass to check if gradients are calculated correctly.
        """
        # Forward pass
        sample_input = self.X_train[0]
        target = self.y_train[0]
        output = self.nn.forward(sample_input, sigmoid)

        # Backward pass
        gradients = self.nn.backward(output, target, sigmoid_derivative)

        # Check if gradients were calculated for each layer
        self.assertEqual(len(gradients), len(self.nn.layers))
        for grad, layer in zip(gradients, self.nn.layers):
            self.assertEqual(grad.shape, layer.weights.shape)
        print("Backward pass gradient shape test passed.")

    def test_training_process(self):
        """
        Train the network on the Iris dataset for a few epochs and test if it improves accuracy.
        """
        initial_accuracy = self.calculate_accuracy(self.nn, self.X_test, self.y_test)

        # Train for a small number of epochs to see if accuracy improves
        self.nn.train(
            self.X_train, self.y_train,
            epochs=10,
            learning_rate=0.1,
            activation_function=sigmoid,
            activation_derivative=sigmoid_derivative
        )

        final_accuracy = self.calculate_accuracy(self.nn, self.X_test, self.y_test)

        # Check if accuracy improved after training
        self.assertGreater(final_accuracy, initial_accuracy)
        print("Training process accuracy improvement test passed.")

    def calculate_accuracy(self, network, X, y):
        """
        Calculate the accuracy of the neural network on a given dataset.
        """
        correct = 0
        for x, target in zip(X, y):
            output = network.forward(x, sigmoid)
            predicted = np.argmax(output)
            actual = np.argmax(target)
            if predicted == actual:
                correct += 1
        return correct / len(X) * 100


if __name__ == "__main__":
    unittest.main()
