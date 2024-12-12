import numpy as np
from midterm_nueralnetworks.neural_network.activation import activation_funcs, activation_derivatives
from midterm_nueralnetworks.neural_network._kernels import _2dconvolve, _kernel_op_size, _2dmaxpool, _convolve
from abc import ABC, abstractmethod
from typing import Tuple, Union
from functools import cached_property

_2DShape = Union[int, Tuple[int, ...]]


class Layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Linear(Layer):
    """
    A class representing a fully connected layer in a feedforward neural network, 
    with the bias term absorbed into the weight matrix.
    """

    def __init__(self, input_size: int, output_size: int, activation: str = "relu", final_layer=False,
                 weight_init="he_normal"):
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

        super().__init__()
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

    def initialize_weights(self, input_size: int, output_size: int, weight_init: str = "he_normal") -> np.ndarray:
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
        print(f"prev input Linear: {self.prev_input.shape}")
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
        # Calculate the derivative of the activation function on the pre-activations

        if not (self.final_layer and self.activation == "softmax"):
            # Standard derivative handling
            delta *= self._activation_derivative(self.preactivations)

        delta = np.where(np.abs(delta) < delta_threshold, 0, delta)  # Apply thresholding to the delta for stability

        prev_input_with_bias = self.concat_bias(self.prev_input)

        # Compute the gradient for weights as the outer product of delta and previous input
        self.grad_weights = np.dot(delta.T, prev_input_with_bias) / delta.shape[0]

        # Compute the delta to pass to the previous layer
        delta_prev = np.dot(delta, (self.weights[:, :-1] + self.momentum[:, :-1]))

        return delta_prev

    def get_newtons_update(self, learning_rate, lambda_reg):
        """
        Computes the update step for the weights using Newton's method.

        Parameters:
        ----------
        learning_rate : float
            The learning rate to scale the weight update.
        lambda_reg : float
            Regularization parameter.

        Returns:
        -------
        np.ndarray
            The update step for the weights.
        """
        flattened_grad = self.grad_weights.flatten()

        # Compute the diagonal approximation of the Hessian matrix instead of the inverse
        hessian_diag_approx = np.square(flattened_grad) + lambda_reg

        # Compute the update step using the diagonal approximation
        update_step_flat = -flattened_grad / hessian_diag_approx

        update_step = update_step_flat.reshape(self.grad_weights.shape) * learning_rate

        return update_step

    @staticmethod
    def concat_bias(X: np.ndarray) -> np.ndarray:
        """Concatenates a bias term to the input data."""
        if X.ndim == 4:
            X = X.reshape(X.shape[0], -1)
        elif X.ndim != 2:
            raise ValueError(f"Unsupported input dimensions: {X.ndim}. Expected 2D or 4D input.")
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


class KernelLayer(Layer):
    """A abstract class representing a layer that operates on patches of the input data using a kernel.
    Examples include convolutional layers and max pooling layers.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _2DShape, stride: int, padding: int) -> object:

        super().__init__()
        self.prev_input = None
        self.activations = None
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    @abstractmethod
    def _kernel_function(self, X, sample):
        """The kernel function that operates on each patch of the input data."""
        pass

    def _pad_input(self, X: np.ndarray):
        """Pad the input data with zeros to account for the padding.

        Args:
            X (np.ndarray): A 4D array of shape (batch_size, in_channels, height, width)

        Returns:
            np.ndarray: Padded input data
        """
        if self.padding == 0:
            return X
        elif X.ndim == 4:
            return np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          mode='constant')
        elif X.ndim == 3:
            return np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")

    def _stride_size(self, dim_len, kernel_len):
        """Calculate the output size of a kernel operation for a single dimension."""
        if (dim_len + 2 * self.padding - kernel_len) % self.stride != 0:
            raise ValueError("Stride and kernel size are incompatible with input dimension.")
        return (dim_len + 2 * self.padding - kernel_len) // self.stride + 1

    def _activation_shape(self, X: np.ndarray):
        """Calculate the output shape of the layer.
        Should need to be calculated once since the input shape should not change.
        """
        if X.ndim == 4:
            batch_size, X_in_channels, input_height, input_width = X.shape
        elif X.ndim == 3:
            X_in_channels, input_height, input_width = X.shape
            batch_size = 1
        else:
            raise ValueError(f"Expected input with 2 or 4 dimensions, got {X.ndim} dimension")

        height = self._stride_size(input_height, self.kernel_size[0])
        width = self._stride_size(input_width, self.kernel_size[1])

        return batch_size, self.out_channels, height, width

    def forward(self, X: np.ndarray):
        self.prev_input = X
        print(f"prev input Kernel Layer: {self.prev_input.shape}")

        batch_size, X_in_channels, input_height, input_width = X.shape

        # If the channels are not specified, assume the input channels are the same as the output channels
        if self.in_channels is None and self.out_channels is None:
            self.in_channels = X_in_channels
            self.out_channels = X_in_channels
        else:
            if X_in_channels != self.in_channels:
                raise ValueError(
                    f"Number of input channels ({X_in_channels}) does not match expected input channels ({self.in_channels})"
                )

        self.activations = np.empty(self._activation_shape(X))

        for sample in range(batch_size):
            self.activations[sample] = self._kernel_function(X[sample], sample)

        return self.activations

    @abstractmethod
    def backward(self, delta, delta_threshold=1e-6):
        pass


class Conv2D(KernelLayer):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _2DShape,
            stride: int = 1,
            padding: int = 0
    ):
        """A class representing a 2D convolutional layer in a convolutional neural network.

        Args:
            in_channels (int): number of input channels 
            out_channels (int): number of output channels
            kernel_size (_2DShape): size of the kernel, assuming square kernel. Can be a tuple of (height, width) or a single integer.
                where height and width are the same.
            stride (int, optional): Horizontal and vertical stride . Defaults to 1.
            padding (int, optional): Amount of padding. Defaults to 0.
        """

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # Channel first kernels
        # TODO: Improve initialization method
        scale = np.sqrt(2 / np.prod(self.kernel_size))
        self._filters = np.random.randn(out_channels, in_channels, *self.kernel_size) * scale
        self.bias = np.zeros(out_channels)
        self.grad_filters = None
        self.grad_bias = None
        self.grad_input = None
        self.momentum = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.firstm = np.zeros_like(self._filters)
        self.secondm = np.zeros_like(self._filters)

    def _kernel_function(self, X, sample):
        padded_X = self._pad_input(X)

        output = _2dconvolve(
            kernels=self._filters,
            X=padded_X,
            stride=self.stride,
            padding=0
        )

        output += self.bias[:, None, None]
        return output

    def backward(self, delta, delta_threshold=1e-6):
        """
        Backward pass for the Conv2D layer
        """

        self.grad_bias = np.sum(delta, axis=(0, 2, 3))

        # Partial Derivative of Filter
        # pL/pF = X (*) pL / pY
        # self.grad_filters = _2dconvolve(
        #     kernels = delta,
        #     X = self.prev_input,
        #     stride = 1,
        #     padding = 0, # Valid convolution so no padding
        # )
        test = _convolve(delta[0,0], self.prev_input[0,0], 1, 0)
        print(test.shape)
        print(self._filters.shape)
        # Partial Derivative of Input
        # pL/pX = F' (*) pL / pY
        flipped_filter = np.flip(self._filters, axis=(2,3)) # Double check that this does the flip correctly
        # self.grad_input = _2dconvolve(
        #     kernels = flipped_filter,
        #     X = delta,
        #     stride = 1,
        #     # We need to do full convolution, so we have to add padding
        #     # of kernel_size - 1
        #     padding= self.kernel_size[1] - 1
        # )
        test2 = _convolve(flipped_filter[0,0], delta[0,0], 1, self.kernel_size[1]-1)
        print(test2.shape)
        print(self.prev_input.shape)
        
        # Return the gradient of the input for the previous layer
        return self.grad_input


class MaxPool2D(KernelLayer):

    def __init__(
            self,
            kernel_size: _2DShape,
            stride: int = 1,
            padding: int = 0
    ):
        """A class representing a 2D max pooling layer in a convolutional neural network.

        Args:
            kernel_size (int): size of the pooling window, assuming square window
            stride (int, optional): Horizontal and vertical stride . Defaults to 1.
            padding (int, optional): Amount of padding. Defaults to 0.
        """

        super().__init__(
            in_channels=None,
            out_channels=None,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        self.max_positions = None

    def forward(self, X):
        if self.in_channels is None and self.out_channels is None:
            self.in_channels = X.shape[1]
            self.out_channels = X.shape[1]

        # Initialize the max positions array
        self.max_positions = np.empty((*self._activation_shape(X), 2), dtype=int)

        return super().forward(X)

    def _kernel_function(self, X, sample):
        """Perform the max pooling operation for a single sample."""

        res, pos = _2dmaxpool(
            self.kernel_size,
            X,
            self.stride,
            self.padding
        )

        # Store the positions of the max values for backpropagation
        self.max_positions[sample] = pos

        return res

    def backward(self, delta, delta_threshold=1e-6):

        batch_size, n_channels, input_height, input_width = self.prev_input.shape
        _, _, output_height, output_width = delta.shape

        grad_input = np.zeros_like(self.prev_input)

        for sample in range(batch_size):
            for channel in range(n_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        index1 = self.max_positions[sample, channel, i, j, 0]
                        index2 = self.max_positions[sample, channel, i, j, 1]
                        grad_input[sample, channel, index1, index2] = delta[sample, channel, i, j]

        return grad_input


class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.prev_input = X
        print(f"prev input Flatten: {self.prev_input.shape}")
        X.reshape(X.shape[0], -1)
        return X

    def backward(self, delta):
        return delta.reshape(self.prev_input.shape)


class ActivationLayer(Layer):
    def __init__(self, activation: str):
        super().__init__()
        try:
            self.activation_func = activation_funcs[activation]
            self.activation_derivative = activation_derivatives[activation]
        except KeyError:
            raise ValueError(f"Activation function {activation} is not supported")

    def forward(self, X):
        self.prev_input = X
        print(f"prev input Activation: {self.prev_input.shape}")
        return self.activation_func(X)

    def backward(self, delta):
        return self.activation_derivative(self.prev_input)
