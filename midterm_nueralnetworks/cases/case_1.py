import os
import numpy as np
import matplotlib.pyplot as plt
from midterm_nueralnetworks.neural_network.layer import Layer
from midterm_nueralnetworks.neural_network import activation as act
from midterm_nueralnetworks.neural_network import loss as loss
from midterm_nueralnetworks.neural_network.feed_forward_neural_network import (
    FeedforwardNeuralNetwork
)


def generate_data(num_samples=100, range_min=-3, range_max=3):
    """
    Generate random samples and their sine values in the specified range.
    """
    np.random.seed(42)  # For reproducibility
    x_train = np.random.uniform(range_min, range_max, num_samples).reshape(-1, 1)
    y_train = np.sin(x_train)
    return x_train, y_train


def initialize_network():
    """
    Initialize the Feedforward Neural Network with a specified architecture.
    """
    return FeedforwardNeuralNetwork(Layer(1,10).setActivation(act.tanh, act.tanh_derivative),
                                    Layer(10,1).setActivation(act.tanh, act.tanh_derivative) )  # 1 input, 10 hidden, 1 output


def train_network(nn, x_train, y_train, epochs=1000, learning_rate=0.005):
    """
    Train the neural network to approximate sin(x).
    """
    for epoch in range(epochs):
        predictions = []
        for xi, yi in zip(x_train, y_train):
            # Forward pass, backward pass, and gradient descent update
            output = nn.forward(xi)
            gradients = nn.backward(output, yi, loss.mse_derivative)
            nn.gd(gradients, learning_rate)
            predictions.append(output.item())  # Collect predictions for evaluation

        # Print MSE every 100 epochs
        if epoch % 100 == 0:
            mse_loss = np.mean((y_train.flatten() - np.array(predictions)) ** 2)
            print(f"Epoch {epoch}, MSE Loss: {mse_loss}")

    # Calculate final MSE
    final_mse = np.mean((y_train.flatten() - np.array(predictions)) ** 2)
    print(f"Final MSE Loss: {final_mse}")
    return np.array(predictions)


def save_plot(x_vals, true_vals, predicted_vals, output_dir="./docs"):
    """
    Save the true vs. predicted values plot to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    file_path = os.path.join(output_dir, "sin_approximation_plot.png")

    # Generate and save the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_vals, true_vals, color='blue', label="True sin(x)", alpha=0.6)
    plt.scatter(x_vals, predicted_vals, color='red', label="Predicted sin(x)", alpha=0.6)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.title("Feedforward Neural Network Approximation of sin(x)")
    plt.savefig(file_path)
    print(f"Plot saved to {file_path}")
    plt.close()


def main():
    """
    Main function to coordinate data generation, network training, and plotting.
    """
    # Generate data
    x_train, y_train = generate_data()

    # Initialize network
    nn = initialize_network()

    # Train network and get predictions
    predictions = train_network(nn, x_train, y_train)

    # Save plot of true vs. predicted values
    save_plot(x_train.flatten(), y_train.flatten(), predictions)


if __name__ == "__main__":
    main()
