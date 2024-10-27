import numpy as np
from midterm_nueralnetworks.neural_network.layer import Layer

# Activation and loss derivatives remain unchanged
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    x = np.clip(x, 1e-7, 1 - 1e-7)
    return x * (1 - x)

def mse_derivative(output, target):
    return output - target

def cross_entropy_derivative(output, target):
    return - (target / output) + ((1 - target) / (1 - output))

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
        self.inputs = [inputs]
        for layer in self.layers:
            inputs = layer.forward(inputs)
            self.inputs.append(inputs)
            inputs = activation_function(inputs)
        return inputs

    def backward(self, output, target, activation_derivative, loss_derivative):
        """
        Perform backpropagation to calculate gradients for weights in all layers.
        """
        delta = loss_derivative(output, target) * activation_derivative(output)
        gradients = []

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            prev_input = np.append(self.inputs[i], 1)
            grad_weights = np.outer(delta, prev_input)
            gradients.append(grad_weights)
            if i != 0:
                delta = np.dot(layer.weights[:, :-1].T, delta) * activation_derivative(self.inputs[i])

        gradients.reverse()
        return gradients

    def gd(self, gradients, learning_rate):
        """
        Performs gradient descent to update weights based on the computed gradients.
        """
        for layer, grad_weights in zip(self.layers, gradients):
            layer.weights -= learning_rate * grad_weights

    def train(self, x, y, epochs, learning_rate, activation_function, activation_derivative, loss_derivative,
              batch_size=32):
        """
        Train the neural network using mini-batch gradient descent and return final predictions.
        """
        num_samples = x.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Initialize cumulative gradients for mini-batch averaging
                batch_gradients = [np.zeros_like(layer.weights) for layer in self.layers]

                for xi, yi in zip(x_batch, y_batch):
                    output = self.forward(xi, activation_function)
                    gradients = self.backward(output, yi, activation_derivative, loss_derivative)

                    for i, grad in enumerate(gradients):
                        batch_gradients[i] += grad

                # Average gradients for the batch and update weights
                batch_gradients = [g / len(x_batch) for g in batch_gradients]
                self.gd(batch_gradients, learning_rate)

        # Generate and return predictions for the entire dataset after training
        predictions = np.array([self.forward(xi, activation_function) for xi in x])
        return predictions

