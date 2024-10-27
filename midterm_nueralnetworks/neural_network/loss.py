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

get_loss_derivative = {
    "mse": mse_derivative,
    "cross_entropy": cross_entropy_derivative
}