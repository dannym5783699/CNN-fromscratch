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

    def gd(self, learning_rate, lambda_reg=0):
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
            layer.weights -= learning_rate * (layer.grad_weights + lambda_reg * layer.weights)

    def newtons_method(self, learning_rate, lambda_reg=0):
        """
        Performs Newton's method to update weights based on the Hessian and gradient.

        Parameters:
        ----------
        learning_rate : float
            The learning rate to scale the weight update.
        lambda_reg : float
            Regularization parameter.
        """
        for layer in self.layers:
            # Hessian approximation computed by the layer's own method
            hessian_approx = layer.compute_hessian_approx(lambda_reg)

            try:
                hessian_inv = np.linalg.pinv(hessian_approx)
            except np.linalg.LinAlgError:
                hessian_inv = np.eye(hessian_approx.shape[0]) * lambda_reg

            update_step = learning_rate * np.dot(hessian_inv, layer.grad_weights.flatten()).reshape(layer.weights.shape)
            layer.weights += update_step

    def zero_grad(self):
        """
        Zero out the gradients for all layers.
        """
        for layer in self.layers:
            layer.grad_weights = np.zeros_like(layer.weights)

    def train(self, x, y, epochs, learning_rate, loss_derivative, method="gd"):
        """
        Train the neural network using the specified optimization method.

        Parameters:
        ----------
        method : str
            The optimization method to use: "gd" for gradient descent or "newton" for Newton's method.
        """
        for epoch in range(epochs):
            for xi, yi in zip(x, y):
                output = self.forward(xi)
                self.backward(output, yi, loss_derivative)

                if method == "gd":
                    self.gd(learning_rate)
                elif method == "newton":
                    self.newtons_method(learning_rate)
