import numpy as np
from midterm_nueralnetworks.neural_network.layer import Layer
from typing import List

class FeedforwardNeuralNetwork:
    def __init__(self, layers : List[Layer]):
        """
        Initializes the Feedforward Neural Network with the given layer sizes.
        """
        self.layers = layers

    def forward(self, X):
        """
        Perform a forward pass through the network.
        """
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)

        return activation

    def backward(self, output, target, loss_derivative):
        """
        Perform backpropagation to calculate gradients for weights in all layers.

        Parameters:
        ----------
        output : numpy.ndarray
            The output from the network.
        target : numpy.ndarray
            The true labels for the output.
        activation_derivative : callable
            The derivative of the activation function used in the output layer.
        loss_derivative : callable
            The derivative of the loss function with respect to the output.
        """
        # Calculate the delta for the output layer using the loss derivative
        delta = loss_derivative(output, target)

        for layer in reversed(self.layers):
            delta = layer.backward(delta)


    def gd(self, learning_rate, friction=0, lambda_reg=0):
        """
        Performs gradient descent to update weights based on the computed gradients.

        Parameters:
        ----------
        gradients : list of numpy.ndarray
            A list containing the gradients for each layer, computed from backpropagation.
        learning_rate : float
            The learning rate to control the size of the weight updates.
        """
        for layer in self.layers:
            np.clip(layer.grad_weights, -1, 1, out=layer.grad_weights)
            layer.momentum = (layer.momentum*friction) - (learning_rate * (layer.grad_weights + lambda_reg * layer.weights))
            layer.weights += layer.momentum


    def zero_grad(self):
        """
        Zero out the gradients for all layers.
        """
        for layer in self.layers:
            layer.grad_weights = np.zeros_like(layer.weights)

    def train(self, x, y, epochs, learning_rate, loss_derivative, friction):
        """
        Train the neural network using gradient descent.
        """
        for epoch in range(epochs):
            for xi, yi in zip(x, y):
                output = self.forward(xi)  # Forward pass
                self.backward(output, yi, loss_derivative)  # Backpropagation
                self.gd(self, learning_rate, friction)  # Update weights using gradient descent
