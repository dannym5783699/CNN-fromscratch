#Activation functions
import numpy as np
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def sigmoid(x):
    # Clip input to prevent overflow in exp() for large values
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    x = np.clip(x, 1e-7, 1 - 1e-7)  # Clip values to avoid overflow
    return x * (1 - x)