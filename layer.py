import numpy as np

class Layer:
    """
    A class representing a fully connected layer in a feedforward neural network, 
    with the bias term absorbed into the weight matrix.
    """

    def __init__(self, input_size, output_size):
        """
        Initializes the Layer class.

        Parameters:
        ----------
        input_size : int
            The number of neurons in the previous layer (or input layer).
        output_size : int
            The number of neurons in the current layer (or output layer).
        """

        self.weights = np.random.randn(output_size, input_size + 1)

    def forward(self, inputs):
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

        inputs_with_bias = np.append(inputs, 1)

        self.z = np.dot(self.weights, inputs_with_bias)
        return self.z
