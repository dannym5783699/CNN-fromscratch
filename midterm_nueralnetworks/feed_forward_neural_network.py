import numpy as np
from midterm_nueralnetworks.layer import Layer


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    # Clip input to prevent overflow in exp() for large values
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    x = np.clip(x, 1e-7, 1 - 1e-7)  # Clip values to avoid overflow
    return x * (1 - x)



class FeedforwardNeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initializes the Feedforward Neural Network with the given layer sizes.
        """
        self.layers = [Layer(layer_sizes[i], layer_sizes[i + 1])
                       for i in range(len(layer_sizes) - 1)]
        self.inputs = []  # To store inputs for backpropagation

    def forward(self, inputs, activation_function):
        """
        Perform a forward pass through the network, applying the given activation function.
        """
        self.inputs = [inputs]  # Store initial input for backpropagation
        for layer in self.layers:
            inputs = layer.forward(inputs)
            self.inputs.append(inputs)  # Store inputs for each layer
            print(inputs)
            inputs = activation_function(inputs)
        return inputs

    def backward(self, output, target, activation_derivative):
        """
        Perform backpropagation to calculate gradients for weights in all layers.
        """
        # Calculate the delta for the output layer
        delta = (output - target) * activation_derivative(output)  # Applies the activation derivative

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
                delta = np.dot(layer.weights[:, :-1].T, delta) * activation_derivative(self.inputs[i])

        # Reverse to match forward order if necessary
        gradients.reverse()
        return gradients

    def update_weights(self, gradients, learning_rate):
        """
        Update the weights of all layers using the gradients and learning rate.
        """
        for layer, grad_weights in zip(self.layers, gradients):
            print(f"Updating weights for layer with gradient: {grad_weights}")
            layer.weights -= learning_rate * grad_weights  # Update weights

    def train(self, x, y, epochs, learning_rate, activation_function, activation_derivative):
        """
        Train the neural network using gradient descent.
        """
        for epoch in range(epochs):
            for xi, yi in zip(x, y):
                output = self.forward(xi, activation_function)  # Forward pass
                gradients = self.backward(output, yi, activation_derivative)  # Backpropagation
                self.update_weights(gradients, learning_rate)  # Update weights
