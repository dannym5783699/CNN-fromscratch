import numpy as np


class Layer:
    def __init__(self, input_size, output_size):
        # Weights include an additional column for biases
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)  # Biases for each neuron


class FeedforwardNeuralNetwork:
    def __init__(self, layers_sizes):
        self.layers = []
        for i in range(len(layers_sizes) - 1):
            self.layers.append(Layer(layers_sizes[i], layers_sizes[i + 1]))

    def forward(self, x):
        # Forward propagation
        x = x.reshape(-1, 1)  # Reshape input to be a column vector
        self.inputs = [x]  # Store inputs for backpropagation
        for layer in self.layers:
            x = np.dot(layer.weights, x) + layer.biases  # Weighted sum (weights * input) + bias
            x = self.activation(x)  # Apply activation function
            self.inputs.append(x)  # Store output for backpropagation
        return x

    def activation(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        # Derivative of sigmoid function
        return x * (1 - x)

    def backward(self, output, target):
        # Backpropagation to calculate gradients
        gradients = []
        error = target - output  # Initial error (target - output)
        delta = error * self.activation_derivative(output)  # Initial delta
        dL_dw = []  # Store gradients for weight updates

        # Backpropagate through each layer in reverse
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            prev_output = self.inputs[i]  # The output of the previous layer (or input for the first layer)

            # Compute gradient for weights and biases
            grad_weights = np.dot(delta, prev_output.T)
            grad_biases = delta  # Gradient for the biases
            gradients.append((grad_weights, grad_biases))

            if i != 0:  # No need to calculate delta for the input layer
                delta = np.dot(layer.weights.T, delta) * self.activation_derivative(prev_output)

        gradients.reverse()  # Reverse to match the layer order
        return gradients

    def gd(self, gradients, learning_rate):
        # Gradient descent to update weights and biases
        for layer, (grad_weights, grad_biases) in zip(self.layers, gradients):
            layer.weights -= learning_rate * grad_weights
            layer.biases -= learning_rate * grad_biases

    def train(self, x, y, epochs, learning_rate, loss_fn):
        for epoch in range(epochs):
            for xi, yi in zip(x, y):
                # Forward pass
                output = self.forward(xi)

                # Backward pass
                gradients = self.backward(output, yi)

                # Update weights
                self.gd(gradients, learning_rate)


# Example usage
if __name__ == '__main__':
    # Define layer sizes: 2 input neurons, 2 hidden neurons, 1 output neuron
    layers_sizes = [2, 2, 1]

    # Initialize the FNN
    fnn = FeedforwardNeuralNetwork(layers_sizes)

    # Dummy training data
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])

    # Train the network
    fnn.train(X_train, y_train, epochs=1000, learning_rate=0.1, loss_fn=lambda o, t: t - o)

    # Test the network
    output = fnn.forward(np.array([0, 0]))
    print("Output for [0, 0]:", output)

    output = fnn.forward(np.array([0, 1]))
    print("Output for [0, 1]:", output)

    output = fnn.forward(np.array([1, 0]))
    print("Output for [1, 0]:", output)

    output = fnn.forward(np.array([1, 1]))
    print("Output for [1, 1]:", output)
