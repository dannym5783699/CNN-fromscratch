import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
import unittest
import numpy as np
from midterm_nueralnetworks.feed_forward_neural_network import FeedforwardNeuralNetwork, relu, sigmoid
from midterm_nueralnetworks.layer import Layer


class TestFeedforwardNeuralNetwork(unittest.TestCase):

    def test_forward_with_relu_activation(self):
        """
        Test the forward pass with ReLU activation function.
        """
        input_size = 3
        hidden_size = 2
        output_size = 1
        # Initialize the network with 3 input neurons, 2 hidden neurons, and 1 output neuron
        nn = FeedforwardNeuralNetwork(
            Layer(input_size,hidden_size).setActivation(relu), 
            Layer(hidden_size,output_size).setActivation(relu)
            )

        # Input data
        inputs = np.array([0.5, 0.1, -0.3])
        #add 1 for manual calculation of bias
        expected_output = np.array([0.5, 0.1, -0.3])
    


        # Set a known weight matrix in the Layer for predictable behavior
        for layer in nn.layers:
            layer.weights = np.ones((layer.weights.shape))
            # Manually calculate the expected output for the given input and weights, weights and bias are 1.
            expected_output = relu(np.dot(np.ones(layer.weights.shape), np.append(expected_output, 1)))



        # Forward pass using ReLU activation
        output = nn.forward(inputs)

        # Compare the output from the forward pass with the expected output
        np.testing.assert_array_almost_equal(output, expected_output, decimal=6)

    def test_forward_with_sigmoid_activation(self):
        """
        Test the forward pass with Sigmoid activation function.
        """
        input_size = 3
        hidden_size = 2
        output_size = 1
        # Initialize the network with 3 input neurons, 2 hidden neurons, and 1 output neuron
        nn = FeedforwardNeuralNetwork(
                Layer(input_size,hidden_size).setActivation(sigmoid), 
                Layer(hidden_size,output_size).setActivation(sigmoid)
                )

        # Input data
        inputs = np.array([0.5, 0.1, -0.3])
        expected_output = np.array([0.5, 0.1, -0.3])

        # Set a known weight matrix in the Layer for predictable behavior
        for layer in nn.layers:
            layer.weights = np.ones((layer.weights.shape))
            expected_output = sigmoid(np.dot(np.ones(layer.weights.shape), np.append(expected_output, 1)))


        # Forward pass using Sigmoid activation
        output = nn.forward(inputs)


        # Compare the output from the forward pass with the expected output
        np.testing.assert_array_almost_equal(output, expected_output, decimal=6)


if __name__ == '__main__':
    unittest.main()
