import numpy as np
from sklearn.metrics import mean_squared_error, log_loss

def mse_derivative(output, target):
    """
    Derivative of Mean Squared Error (MSE) loss with respect to the output.
    """
    return output - target


def cross_entropy_derivative(output, target):
    """
    Derivative of Cross-Entropy loss with respect to the output.
    """
    return - (target / output) + ((1 - target) / (1 - output))

def NLL_derivative_softmax(output, target):
    """
    Derivative of Negative Log Likelihood loss with respect to the output for softmax activation.
    """
    return output - target

get_loss_derivative = {
    "mse": mse_derivative,
    "cross_entropy": cross_entropy_derivative,
    "nll_softmax": NLL_derivative_softmax
}

def nll_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

get_loss = {
    "mse": mean_squared_error,
    "cross_entropy": log_loss,
    "nll_softmax": nll_loss
}