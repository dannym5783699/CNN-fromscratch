import numpy as np
import os
import matplotlib.pyplot as plt
from midterm_nueralnetworks.neural_network.feed_forward_neural_network import (
    FeedforwardNeuralNetwork, tanh, tanh_derivative, mse_derivative
)


def generate_vanderpol_data(num_samples=1000, range_min=-3, range_max=3):
    """
    Generate random samples based on the Van der Pol system in the range [range_min, range_max].
    """
    x1 = np.random.uniform(range_min, range_max, num_samples)
    x2 = np.random.uniform(range_min, range_max, num_samples)

    # Van der Pol equations
    x1_dot = x2
    x2_dot = -x1 + (1 - x2 ** 2) * x2

    # Define state and one-step updated state (reachability relation)
    X = np.column_stack((x1, x2))  # Initial states
    Y = np.column_stack((x1 + 0.5 * x1_dot, x2 + 0.5 * x2_dot))  # One-step reachability after 0.5 seconds
    return X, Y


def save_plot(X, Y_true, Y_pred, output_dir="./docs"):
    """
    Save the true vs. predicted values plot to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "vanderpol_approximation_plot.png")

    plt.figure(figsize=(10, 6))
    plt.scatter(Y_true[:, 0], Y_true[:, 1], color='blue', label="True values", alpha=0.6)
    plt.scatter(Y_pred[:, 0], Y_pred[:, 1], color='red', label="Predicted values", alpha=0.6)
    plt.legend()
    plt.xlabel("x1'")
    plt.ylabel("x2'")
    plt.title("Van der Pol System Reachability Approximation")
    plt.savefig(file_path)
    print(f"Plot saved to {file_path}")
    plt.close()


def main():
    # Generate Van der Pol data
    X_train, Y_train = generate_vanderpol_data()

    # Initialize network with input-output layer size (2, 10, 10, 2) suitable for Van der Pol system
    nn = FeedforwardNeuralNetwork([2, 10, 10, 2])

    # Train the network using mini-batch gradient descent
    predictions = nn.train(
        X_train, Y_train, epochs=1000, learning_rate=0.005,
        activation_function=tanh, activation_derivative=tanh_derivative,
        loss_derivative=mse_derivative, batch_size=32
    )

    # Save and visualize the plot of true vs. predicted values
    save_plot(X_train, Y_train, predictions)


if __name__ == "__main__":
    main()
