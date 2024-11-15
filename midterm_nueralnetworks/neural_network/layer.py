import numpy as np
from midterm_nueralnetworks.neural_network.activation import activation_funcs, activation_derivatives

class Layer:
    """
    A class representing a fully connected layer in a feedforward neural network, 
    with the bias term absorbed into the weight matrix.
    """

    def __init__(self, input_size : int, output_size : int, activation : str = "relu", final_layer=False, weight_init = "he_normal"):
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
        weight_init : str
            The initialization method to use for the weights. Default is 'he_normal'. Can be one of
            'he_normal', 'xavier_uniform', 'random', or 'zeros'.
        """

        try:
            self.activation = activation
            self._activation_func = activation_funcs[activation]
            self._activation_derivative = activation_derivatives[activation]
        except KeyError:
            raise ValueError(f"Activation function {activation} not supported")

        self.final_layer = final_layer
        self.prev_input = None
        self.activations = None
        self.preactivations = None
        self.grad_weights = None

        self.weights = self.initialize_weights(input_size, output_size, weight_init)
        self.momentum = np.zeros_like(self.weights)
        self.firstm = np.zeros_like(self.weights)
        self.secondm = np.zeros_like(self.weights)
    
    def initialize_weights(self, input_size : int, output_size : int, weight_init : str = "he_normal") -> np.ndarray:
        """
        Initializes the weights of the layer.

        Parameters:
        ----------
        input_size : int
            The number of neurons in the previous layer (or input layer).
        output_size : int
            The number of neurons in the current layer (or output layer).
        weight_init : str
            The initialization method to use for the weights. Default is 'he_normal'. Can be one of
            'he_normal', 'xavier_uniform', 'random', or 'zeros'.
            
        Returns:
        -------
        np.ndarray
            The initialized weights.
        """
        if weight_init == "he_normal":
            weights = np.random.randn(output_size, input_size + 1) * np.sqrt(2 / input_size)
        elif weight_init == "xavier_uniform":
            limit = np.sqrt(6 / (input_size + output_size))
            weights = np.random.uniform(-limit, limit, (output_size, input_size + 1))
        elif weight_init == "random":
            weights = np.random.randn(output_size, input_size + 1)
        elif weight_init == "zeros":
            weights = np.zeros((output_size, input_size + 1))
        else:
            raise ValueError(f"Weight initialization method {weight_init} not supported")
        
        return weights

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

    def backward(self, delta, delta_threshold=1e-6):
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

        if not(self.final_layer and self.activation == "softmax"):
            # Standard derivative handling
            delta *= self._activation_derivative(self.preactivations)

        delta = np.where(np.abs(delta) < delta_threshold, 0, delta) # Apply thresholding to the delta for stability

        prev_input_with_bias = self.concat_bias(self.prev_input)

        # Compute the gradient for weights as the outer product of delta and previous input
        self.grad_weights = np.dot(delta.T, prev_input_with_bias) / delta.shape[0]

        # Compute the delta to pass to the previous layer
        delta_prev = np.dot(delta, (self.weights[:, :-1] + self.momentum[:,:-1]))

        return delta_prev

    def compute_hessian_approx(self, lambda_reg=1e-5):
        """
        Approximates the Hessian matrix for this layer using the outer product of the gradient.

        Parameters:
        ----------
        lambda_reg : float
            Small regularization term to ensure invertibility.

        Returns:
        -------
        np.ndarray
            The approximated Hessian matrix for this layer.
        """
        flattened_grad = self.grad_weights.flatten()
        hessian_approx = np.outer(flattened_grad, flattened_grad)
        hessian_approx += np.eye(hessian_approx.shape[0]) * lambda_reg  # Regularization
        return hessian_approx
    
    @staticmethod
    def concat_bias(X : np.ndarray) -> np.ndarray:
        """Concatenates a bias term to the input data."""
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)