import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from midterm_nueralnetworks.neural_network.feed_forward_neural_network import FeedforwardNeuralNetwork
from midterm_nueralnetworks.neural_network.layer import Linear
from midterm_nueralnetworks.neural_network.loss import NLL_derivative_softmax


def batch_size_test_he_weight():
    print("Preparing data")
    X, Y = load_digits(return_X_y=True)

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(Y.reshape(-1, 1)).toarray()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42, stratify=Y)

    MAX_EPOCHS = 500
    LR = 1e-3
    batch_sizes = [8, 32, 64, 128]  # List of batch sizes to experiment with

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

    for batch_size, ax in zip(batch_sizes, axs.flatten()):

        # Initialize the network
        net = FeedforwardNeuralNetwork([
            Linear(64, 64, "relu"),
            Linear(64, 32, "relu"),
            Linear(32, 10, "softmax", final_layer=True)
        ])

        train_losses = np.zeros(MAX_EPOCHS)
        test_losses = np.zeros(MAX_EPOCHS)

        print(f"Training model with batch size {batch_size}")
        for epoch in range(MAX_EPOCHS):
            batch_losses = []
            for i, (x_batch, y_batch) in enumerate(get_batches(X_train, Y_train, batch_size)):
                y_pred = net.forward(x_batch)
                batch_losses.append(nll_loss(y_pred, y_batch))
                net.backward(y_pred, y_batch, NLL_derivative_softmax)
                net.gd(LR)
                net.zero_grad()
            mean_train_loss = np.mean(batch_losses)
            test_loss = nll_loss(net.forward(X_test), Y_test)
            net.zero_grad()

            train_losses[epoch] = mean_train_loss
            test_losses[epoch] = test_loss

            if epoch % 10 == 0:
                print(f"Epoch {str(epoch).zfill(2)}, Mean Train Loss: {mean_train_loss}, Test Loss: {test_loss}")

        ax.plot(train_losses, label="Train Loss")
        ax.plot(test_losses, label="Test Loss")
        ax.set(
            xlabel="Epoch",
            ylabel="Log Loss",
            title=f"Batch Size: {batch_size}"
        )
        ax.set_yscale('log')
        ax.legend()

    plt.tight_layout()
    plt.show()

def batch_size_test_xavier_weight():
    print("Preparing data")
    X, Y = load_digits(return_X_y=True)

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(Y.reshape(-1, 1)).toarray()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42, stratify=Y)

    MAX_EPOCHS = 500
    LR = 1e-3
    batch_sizes = [8, 32, 64, 128]  # List of batch sizes to experiment with

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

    for batch_size, ax in zip(batch_sizes, axs.flatten()):

        # Initialize the network
        net = FeedforwardNeuralNetwork([
            Linear(64, 64, "relu", weight_init="xavier_uniform"),
            Linear(64, 32, "relu", weight_init="xavier_uniform"),
            Linear(32, 10, "softmax", final_layer=True, weight_init="xavier_uniform")
        ])

        train_losses = np.zeros(MAX_EPOCHS)
        test_losses = np.zeros(MAX_EPOCHS)

        print(f"Training model with batch size {batch_size}")
        for epoch in range(MAX_EPOCHS):
            batch_losses = []
            for i, (x_batch, y_batch) in enumerate(get_batches(X_train, Y_train, batch_size)):
                y_pred = net.forward(x_batch)
                batch_losses.append(nll_loss(y_pred, y_batch))
                net.backward(y_pred, y_batch, NLL_derivative_softmax)
                net.gd(LR)
                net.zero_grad()
            mean_train_loss = np.mean(batch_losses)
            test_loss = nll_loss(net.forward(X_test), Y_test)
            net.zero_grad()

            train_losses[epoch] = mean_train_loss
            test_losses[epoch] = test_loss

            if epoch % 10 == 0:
                print(f"Epoch {str(epoch).zfill(2)}, Mean Train Loss: {mean_train_loss}, Test Loss: {test_loss}")

        ax.plot(train_losses, label="Train Loss")
        ax.plot(test_losses, label="Test Loss")
        ax.set(
            xlabel="Epoch",
            ylabel="Log Loss",
            title=f"Batch Size: {batch_size}"
        )
        ax.set_yscale('log')
        ax.legend()

    plt.tight_layout()
    plt.show()


def get_batches(X, Y, batch_size):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    for i in range(0, len(X), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield X[batch_indices], Y[batch_indices]


def nll_loss(output, target):
    """Calculate the Negative Log Likelihood Loss for a batch."""
    output = np.clip(output, 1e-10, 1 - 1e-10)
    return -np.sum(target * np.log(output)) / target.shape[0]


if __name__ == "__main__":
    batch_size_test_he_weight()
    batch_size_test_xavier_weight()
