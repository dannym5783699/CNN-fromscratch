import numpy as np
from midterm_nueralnetworks.neural_network.layer import Linear, Conv2D, MaxPool2D
from typing import List


class FeedforwardNeuralNetwork:
    def __init__(self, layers: List[Linear]):
        """
        Initializes the Feedforward Neural Network with the given layer sizes.
        """
        self.layers = layers
        self.p1 = 0.9
        self.p2 = 0.999
        self.t = 1

    def forward(self, X):
        """
        Perform a forward pass through the network.
        """
        activation = X
        for layer in self.layers:
            #print(f"Layer Name: ({layer.__class__.__name__}): Input shape = {activation.shape}")
            activation = layer.forward(activation)
            #print(f"Layer Name: ({layer.__class__.__name__}): Output shape = {activation.shape}")
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
            #print(f"Layer Name: ({layer.__class__.__name__}): Input delta shape = {delta.shape}")
            delta = layer.backward(delta)
            #print(f"Layer Name: ({layer.__class__.__name__}): Output delta shape = {delta.shape}")

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
            # If layer is max pooling we do not need to update since it just computes max.
            if (isinstance(layer, Linear)):
                np.clip(layer.grad_weights, -1, 1, out=layer.grad_weights)
                gradientcalc = (learning_rate * (layer.grad_weights + lambda_reg * layer.weights))
                if friction != 0:
                    layer.momentum = (layer.momentum * friction) - gradientcalc
                    layer.weights += layer.momentum
                else:
                    layer.weights -= gradientcalc
            # Convolution is similar but update the filters instead of weights.
            elif (isinstance(layer, Conv2D)):
                gradientcalc = (learning_rate * (layer.grad_filters + lambda_reg * layer._filters))
                if friction != 0:
                    layer.momentum = (layer.momentum * friction) - gradientcalc
                    layer._filters += layer.momentum
                else:
                    layer._filters -= gradientcalc

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
            layer.weights += layer.get_newtons_update(learning_rate, lambda_reg)

    def adam(self, learning_rate, p1=0.9, p2=0.999, lambda_reg=0):
        """
        Performs gradient descent to update weights based on the computed gradients.

        Parameters:
        ----------
        gradients : list of numpy.ndarray
            A list containing the gradients for each layer, computed from backpropagation.
        learning_rate : float
            The learning rate to control the size of the weight updates.
        """
        self.p1 = p1
        self.p2 = p2
        for layer in self.layers:
            if (isinstance(layer, Linear)):
                np.clip(layer.grad_weights, -1, 1, out=layer.grad_weights)
                # update moments
                layer.secondm = (self.p2 * layer.secondm) + ((1 - self.p2) * (layer.grad_weights ** 2))
                layer.firstm = (self.p1 * layer.firstm) + ((1 - self.p1) * (layer.grad_weights))
                # bias correction before weight update
                second = layer.secondm / (1 - (self.p2 ** self.t))
                first = layer.firstm / (1 - (self.p1 ** self.t))
                self.t += 1
                # update weights
                layer.weights -= (learning_rate / (np.sqrt(second) + 1e-8)) * (first)
            elif (isinstance(layer, Conv2D)):
                np.clip(layer.grad_filters, -1, 1, out=layer.grad_filters)
                # update moments
                layer.secondm = (self.p2 * layer.secondm) + ((1 - self.p2) * (layer.grad_filters ** 2))
                layer.firstm = (self.p1 * layer.firstm) + ((1 - self.p1) * (layer.grad_filters))
                # bias correction before weight update
                second = layer.secondm / (1 - (self.p2 ** self.t))
                first = layer.firstm / (1 - (self.p1 ** self.t))
                self.t += 1
                # update weights
                layer._filters -= (learning_rate / (np.sqrt(second) + 1e-8)) * (first)

    def zero_grad(self):
        """
        Zero out the gradients for all layers.
        """
        for layer in self.layers:
            if(isinstance(layer, Linear)):
              layer.grad_weights = np.zeros_like(layer.weights)
            if(isinstance(layer, Conv2D)):
                layer.grad_filters = np.zeros_like(layer._filters) 

    def train(self, x, y, epochs, learning_rate, loss_derivative, friction, method="gd"):
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
                    self.gd(learning_rate, friction)
                elif method == "newton":
                    self.newtons_method(learning_rate)
