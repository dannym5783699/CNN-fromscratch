import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from midterm_nueralnetworks.neural_network.feed_forward_neural_network import \
    FeedforwardNeuralNetwork
from midterm_nueralnetworks.neural_network.layer import Layer
from midterm_nueralnetworks.neural_network.loss import NLL_derivative_softmax
from sklearn.preprocessing import OneHotEncoder


def main():
    print("Preparing data")
    X, Y = load_digits(return_X_y=True)

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(Y.reshape(-1, 1)).toarray()

    print(X.shape, Y.shape)
    print(X[:5], Y[:5])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42, stratify=Y)

    net = FeedforwardNeuralNetwork([
        Layer(64, 64, "relu"),
        Layer(64, 32, "relu"),
        Layer(32, 10, "softmax", final_layer=True)
    ]
    )

    MAX_EPOCHS = 500
    LR = 1e-3
    BATCH_SIZE = 32

    train_losses = np.zeros(MAX_EPOCHS)
    test_losses = np.zeros(MAX_EPOCHS)

    print("training model")
    print(f"{X_train.shape=}, {Y_train.shape=}")
    print(f"{X_test.shape=}, {Y_test.shape=}")
    for epoch in range(MAX_EPOCHS):
        batch_losses = []
        for i, (x_batch, y_batch) in enumerate(get_batches(X_train, Y_train, BATCH_SIZE)):
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

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.show()


def get_batches(X, Y, batch_size):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    for i in range(0, len(X), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield X[batch_indices], Y[batch_indices]


def nll_loss(output, target):
    """Calculate the Negative Log Likelihood Loss for a batch."""
    # Clip output to avoid log(0)
    output = np.clip(output, 1e-10, 1 - 1e-10)
    return -np.sum(target * np.log(output)) / target.shape[0]


if __name__ == "__main__":
    main()
