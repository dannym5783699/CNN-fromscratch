import numpy as np
from midterm_nueralnetworks.neural_network.activation import activation_funcs, activation_derivatives

class Layer:
    """
    A class representing a fully connected layer in a feedforward neural network, 
    with the bias term absorbed into the weight matrix.
    """

    def __init__(self, input_size : int, output_size : int, activation : str = "relu"):
        """
        Initializes the Layer class.

        Parameters:
        ----------
        input_size : int
            The number of neurons in the previous layer (or input layer).
        output_size : int
            The number of neurons in the current layer (or output layer).
        activation : str
            The activation function to use for the layer. Default is 'relu'.
        """

        try:
            self.activation = activation
            self._activation_func = activation_funcs[activation]
            self._activation_derivative = activation_derivatives[activation]
        except KeyError:
            raise ValueError(f"Activation function {activation} not supported")

        self.prev_input = None
        self.activations = None
        self.preactivations = None
        self.grad_weights = None

        self.weights = np.random.randn(output_size, input_size + 1)

    def forward(self, X):
        """
        Performs a forward pass through the layer by calculating the weighted sum of inputs and the bias.

        Parameters:
        ----------
        inputs : numpy.ndarray
            A 1D array of shape (input_size,) representing the input data for the layer.
        
        Returns:
        -------
        numpy.ndarray
            The output of the layer after calculating the weighted sum of inputs and bias, but 
            without applying an activation function.

        Notes:
        -----
        The input is augmented by appending 1 to it to account for the bias term, and the weights matrix 
        already contains the bias information.
        """

        self.prev_input = X
        self.preactivations = np.dot(self.concat_bias(X), self.weights.T)
        self.activations = self._activation_func(self.preactivations)

        return self.activations

    def backward(self, delta):
        """
        Performs the backward pass, calculating gradients for the layer's weights and the delta to pass to previous layers.

        Parameters:
        ----------
        delta : numpy.ndarray
            The error signal from the subsequent layer, scaled by the derivative of the loss with respect to
            this layer's output.

        Returns:
        -------
        numpy.ndarray
            The delta to propagate to the previous layer.
        """
        # Calculate the derivative of the activation function on the preactivations

        delta *= self._activation_derivative(self.preactivations)

        prev_input_with_bias = self.concat_bias(self.prev_input)

        # Compute the gradient for weights as the outer product of delta and previous input
        self.grad_weights = np.dot(delta.T, prev_input_with_bias) / delta.shape[0]

        # Compute the delta to pass to the previous layer
        delta_prev = np.dot(delta, self.weights[:, :-1])

        return delta_prev
    
    @staticmethod
    def concat_bias(X : np.ndarray) -> np.ndarray:
        """Concatenates a bias term to the input data."""
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)