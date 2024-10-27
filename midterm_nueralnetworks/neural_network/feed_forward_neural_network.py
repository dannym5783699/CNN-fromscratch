import numpy as np
from midterm_nueralnetworks.neural_network.layer import Layer


class FeedforwardNeuralNetwork:
    def __init__(self, *layers):
        """
        Initializes the Feedforward Neural Network with the given layer sizes.
        """
        self.layers = list(layers)
        self.inputs = []  # To store inputs for backpropagation

    def forward(self, inputs):
        """
        Perform a forward pass through the network, applying the given activation function.
        """
        self.inputs = [inputs]  # Store initial input for backpropagation
        for layer in self.layers:
            inputs = layer.forward(inputs)
            self.inputs.append(inputs)  # Store inputs for each layer
            inputs = layer.applyActivation(inputs)
        return inputs

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
        delta = loss_derivative(output, target) * self.layers[len(self.layers)-1].activation_derivative(output)

        gradients = []  # Store gradients for each layer

        # Backpropagate through the layers in reverse
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            # Include bias in the previous layer's inputs
            prev_input = np.append(self.inputs[i], 1)

            # Calculate gradients for weights
            grad_weights = np.outer(delta, prev_input)
            gradients.append(grad_weights)

            # Compute delta for the next layer if it's not the input layer
            if i != 0:
                delta = np.dot(layer.weights[:, :-1].T, delta) * layer.activation_derivative(self.inputs[i])

        # Reverse to match forward order if necessary
        gradients.reverse()
        return gradients

    def gd(self, gradients, learning_rate):
        """
        Performs gradient descent to update weights based on the computed gradients.

        Parameters:
        ----------
        gradients : list of numpy.ndarray
            A list containing the gradients for each layer, computed from backpropagation.
        learning_rate : float
            The learning rate to control the size of the weight updates.
        """
        for layer, grad_weights in zip(self.layers, gradients):
            layer.weights -= learning_rate * grad_weights

    def train(self, x, y, epochs, learning_rate, loss_derivative):
        """
        Train the neural network using gradient descent.
        """
        for epoch in range(epochs):
            for xi, yi in zip(x, y):
                output = self.forward(xi)  # Forward pass
                gradients = self.backward(output, yi, loss_derivative)  # Backpropagation
                self.gd(gradients, learning_rate)  # Update weights using gradient descent
