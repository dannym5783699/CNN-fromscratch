import numpy as np
from layer import Layer

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    # Clip input to prevent overflow in exp() for large values
    x = np.clip(x, -500, 500)  
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
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
            inputs = activation_function(inputs)
        return inputs

    def backward(self, output, target, activation_derivative):
        """
        Perform backpropagation to calculate gradients for weights in all layers.
        """
        error = target - output  # Initial error
        delta = error * activation_derivative(output)  # Delta for output layer

        gradients = []  # Store gradients for all layers

        # Backpropagate through the layers in reverse
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            prev_input = np.append(self.inputs[i], 1)  # Include bias in input

            # Calculate gradients for weights
            grad_weights = np.outer(delta, prev_input)
            gradients.append(grad_weights)

            # Compute delta for the next layer if not input layer
            if i != 0:
                delta = np.dot(layer.weights[:, :-1].T, delta) * activation_derivative(self.inputs[i])

        gradients.reverse()  # Match the forward order
        return gradients

    def update_weights(self, gradients, learning_rate):
        """
        Update the weights of all layers using the gradients and learning rate.
        """
        for layer, grad_weights in zip(self.layers, gradients):
            print(f"Updating weights for layer with gradient: {grad_weights}")
            layer.weights += learning_rate * grad_weights  # Update weights

    def train(self, x, y, epochs, learning_rate, activation_function, activation_derivative):
        """
        Train the neural network using gradient descent.
        """
        for epoch in range(epochs):
            for xi, yi in zip(x, y):
                output = self.forward(xi, activation_function)  # Forward pass
                gradients = self.backward(output, yi, activation_derivative)  # Backpropagation
                self.update_weights(gradients, learning_rate)  # Update weights

