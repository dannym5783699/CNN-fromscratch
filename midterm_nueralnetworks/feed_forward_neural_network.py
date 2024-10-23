import numpy as np

from midterm_nueralnetworks.layer import Layer


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FeedforwardNeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initializes the Feedforward Neural Network with the given layer sizes.

        Parameters:
        ----------
        layer_sizes : list of int
            A list containing the sizes of each layer in the network,
            where each element represents the number of neurons in that layer.
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, inputs, activation_function):
        """
        Perform a forward pass through the network, applying the given activation function.

        Parameters:
        ----------
        inputs : numpy.ndarray
            The input data for the neural network.
        activation_function : callable
            The activation function to apply at each layer (e.g., ReLU, Sigmoid).

        Returns:
        -------
        numpy.ndarray
            The final output after passing through all layers and applying the activation function.
        """
        for layer in self.layers:
            inputs = activation_function(layer.forward(inputs))
        return inputs

