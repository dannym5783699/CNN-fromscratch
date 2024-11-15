import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from pathlib import Path

from midterm_nueralnetworks.midterm_test_cases.vanderpol_data import Phi, generate_data
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

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    MAX_EPOCHS = 1000
    LR = 1e-2
    batch_sizes = [8, 32, 64, 128]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

    for batch_size, ax in zip(batch_sizes, axs.flatten()):

        net = FeedforwardNeuralNetwork([
            Layer(2, 64, "relu"),
            Layer(64, 64, "relu"),
            Layer(64, 64, "relu"),
            Layer(64, 2, "linear")
        ]
        )

        train_losses = np.zeros(MAX_EPOCHS)
        test_losses = np.zeros(MAX_EPOCHS)

        print("training model")
        print(f"{X_train.shape=}, {Y_train.shape=}")
        print(f"{X_test.shape=}, {Y_test.shape=}")
        for epoch in range(MAX_EPOCHS):
            batch_losses = []
            for i, (x_batch, y_batch) in enumerate(get_batches(X_train, Y_train, batch_size)):
                y_hat = net.forward(x_batch)
                batch_losses.append(mean_squared_error(y_hat, y_batch))
                net.backward(y_hat, y_batch, mse_derivative)
                net.gd(LR)
                net.zero_grad()
            mean_train_loss = np.mean(batch_losses)
            test_loss = mean_squared_error(net.forward(X_test), Y_test)
            net.zero_grad()

            train_losses[epoch] = mean_train_loss
            test_losses[epoch] = test_loss

            if epoch % 10 == 0:
                print(f"Epoch {str(epoch).zfill(2)}, Mean Train Loss: {mean_train_loss}, Test Loss: {test_loss}")

        ax.plot(train_losses, label="Batch Mean Train Loss")
        ax.plot(test_losses, label="Test Loss")
        ax.set(
            xlabel="Epoch",
            ylabel="Log MSE Loss",
            title=f"Batch Size: {batch_size}"
        )
        ax.set_yscale('log')
        ax.legend()

    plt.tight_layout()
    fpath = fig_folder / "case_2_batch_size_comparison.png"
    plt.savefig(fpath, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
