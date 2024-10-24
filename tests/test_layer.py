import unittest
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
from midterm_nueralnetworks.layer import Layer


class TestLayer(unittest.TestCase):

    def test_forward_output_shape(self):
        """
        Test if the forward method produces an output of the correct shape.
        """
        input_size = 3
        output_size = 2
        layer = Layer(input_size, output_size)
        
        inputs = np.array([0.5, 0.1, -0.3])
        print(layer.weights)
        output = layer.forward(inputs)
        print(output)
        
        # Check if the output shape is correct (should match output_size)
        self.assertEqual(output.shape, (output_size,))

    def test_forward_output_values(self):
        """
        Test if the forward method computes the weighted sum correctly.
        """
        input_size = 2
        output_size = 1
        
        # Initialize the layer with known weights for testing
        layer = Layer(input_size, output_size)
        layer.weights = np.array([[0.2, 0.4, 0.5]])  # Manually set weights for testing, including bias
        
        inputs = np.array([0.5, -0.3])
        output = layer.forward(inputs)
        
        # Manually calculate expected output (z = 0.2*0.5 + 0.4*(-0.3) + 0.5*1)
        expected_output = 0.2 * 0.5 + 0.4 * (-0.3) + 0.5
        self.assertAlmostEqual(output, expected_output, places=6)

    def test_forward_with_zero_weights(self):
        """
        Test the forward method when all weights and biases are zero.
        """
        input_size = 3
        output_size = 2
        layer = Layer(input_size, output_size)
        
        # Set weights and biases to zero
        layer.weights = np.zeros((output_size, input_size + 1))
        
        inputs = np.array([1.0, 2.0, 3.0])
        output = layer.forward(inputs)
        
        # Expect output to be a vector of zeros, as the weights and biases are all zero
        expected_output = np.zeros(output_size)
        np.testing.assert_array_equal(output, expected_output)

if __name__ == '__main__':
    unittest.main()
