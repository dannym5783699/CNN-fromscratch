from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from midterm_nueralnetworks.cases.vanderpol_data import Phi, generate_data
from midterm_nueralnetworks.neural_network.feed_forward_neural_network import \
    FeedforwardNeuralNetwork
from midterm_nueralnetworks.neural_network.layer import Layer
from midterm_nueralnetworks.neural_network.loss import mse_derivative
from midterm_nueralnetworks.neural_network.utils import get_batches


def main():
    fig_folder = Path(__file__).parents[2] / "figures"
    fig_folder.mkdir(exist_ok=True)

    print("Preparing data")
    X, Y = generate_data(101, -3, 3)
    print(X.shape, Y.shape)
    print(X[:5], Y[:5])

    net = FeedforwardNeuralNetwork([
        Layer(2, 64, "relu"),
        Layer(64, 64, "relu"),
        Layer(64, 64, "relu"),
        Layer(64, 2, "linear")
    ]
    )

    MAX_EPOCHS = 10
    #LR = 1e-2
    LR = 1e-2
    BATCH_SIZE = 32

    train_losses = np.zeros(MAX_EPOCHS)
    test_losses = np.zeros(MAX_EPOCHS)

    print("training model")
    for epoch in range(MAX_EPOCHS):
        batch_losses = []
        for i, (x_batch, y_batch) in enumerate(get_batches(X, Y, BATCH_SIZE)):
            y_hat = net.forward(x_batch)
            batch_losses.append(mean_squared_error(y_hat, y_batch))
            net.backward_nest(y_hat, y_batch, mse_derivative)
            net.gd_nest(LR)
        mean_train_loss = np.mean(batch_losses)
        net.zero_grad()

        train_losses[epoch] = mean_train_loss
        if epoch % 10 == 0:
            print(f"Epoch {str(epoch).zfill(2)}, Mean Train Loss: {mean_train_loss}")

    fig, ax = plt.subplots()
    num_timesteps = 150

    x0 = np.array([1.25, 2.35])
    phi_values = np.zeros((num_timesteps, 2))
    net_values = np.zeros((num_timesteps, 2))

    for i in range(num_timesteps):
        y = Phi(x0)
        phi_values[i] = y
        x0 = y

    for i in range(num_timesteps):
        y = net.forward(x0.reshape(1, -1))[0]
        net_values[i] = y
        x0 = y

    ax.scatter(phi_values[:, 0], phi_values[:, 1], label="Phi Predictions")
    ax.scatter(net_values[:, 0], net_values[:, 1], label="Neural Network Predictions")
    ax.set(
        title="Van der Pol Oscillator Model Predictions",
        xlabel="x1",
        ylabel="x2"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig.savefig(fig_folder / "vanderpol_prediction.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
