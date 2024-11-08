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

def sigmoid_derivative(output):
    # 'output' is already the result of sigmoid(x)
    return output * (1 - output)

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
        self.inputs = []  # Reset inputs for each forward pass
        for layer in self.layers:
            pre_activation = layer.forward(inputs)
            self.inputs.append(pre_activation)  # Store pre-activation inputs
            inputs = activation_function(pre_activation)
        return inputs

    def backward(self, output, target, activation_derivative):
        """
        Perform backpropagation to calculate gradients for weights in all layers.
        """
        error = target - output  # Initial error at output layer
        delta = error * activation_derivative(output)  # Delta for output layer

        gradients = [None] * len(self.layers)  # Initialize gradient storage

        # Backpropagate through the layers in reverse
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            prev_input = self.inputs[i]  # Use stored pre-activation input

            # Calculate gradient for current layer's weights
            grad_weights = np.outer(delta, np.append(prev_input, 1))  # Bias included in the gradient
            gradients[i] = grad_weights

            # Compute delta for the next layer if not the input layer
            if i != 0:
                delta = np.dot(layer.weights[:, :-1].T, delta) * activation_derivative(self.inputs[i - 1])

        return gradients

    def update_weights(self, gradients, learning_rate):
        """
        Update the weights of all layers using the gradients and learning rate.
        """
        for layer, grad_weights in zip(self.layers, gradients):
            print(f"Updating weights for layer with gradient:\n{grad_weights}")
            layer.weights += learning_rate * grad_weights  # Update weights

    def train(self, x, y, epochs, learning_rate, activation_function, activation_derivative, batch_size=1):
        """
        Train the neural network using gradient descent with optional mini-batch processing.
        """
        for epoch in range(epochs):
            batch_gradients = [np.zeros_like(layer.weights) for layer in self.layers]

            # Process inputs in mini-batches
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i + batch_size]
                batch_y = y[i:i + batch_size]

                # Accumulate gradients for the mini-batch
                for xi, yi in zip(batch_x, batch_y):
                    output = self.forward(xi, activation_function)
                    gradients = self.backward(output, yi, activation_derivative)
                    for j in range(len(batch_gradients)):
                        batch_gradients[j] += gradients[j]

                # Update weights using averaged gradients
                for j in range(len(self.layers)):
                    self.layers[j].weights += learning_rate * (batch_gradients[j] / batch_size)

            print(f"Epoch {epoch + 1}/{epochs} complete.")

# Example Usage
if __name__ == "__main__":
    # Define the architecture: 2 input neurons, 2 hidden neurons, 1 output neuron
    nn = FeedforwardNeuralNetwork([2, 2, 1])

    # XOR problem dataset
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train the neural network
    nn.train(x, y, epochs=10000, learning_rate=0.1,
             activation_function=sigmoid, activation_derivative=sigmoid_derivative)

    # Test the network
    for xi in x:
        output = nn.forward(xi, sigmoid)
        print(f"Input: {xi} -> Output: {output}")
