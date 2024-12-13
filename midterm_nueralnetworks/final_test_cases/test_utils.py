import numpy as np

from tqdm import tqdm


def train(network, train_loader, test_loader, epochs, learning_rate, loss_derivative):
    train_accuracies = []
    test_accuracies = []

    for epoch in tqdm(range(epochs), desc="Epoch Train", leave=False):
        with tqdm(total=len(train_loader), desc="Batch Train", leave=False) as batch_bar:
            i = 0
            for batch in train_loader:
                X_batch, y_batch = batch
                y_one_hot = np.zeros((y_batch.size(0), 10))
                y_one_hot[np.arange(y_batch.size(0)), y_batch.numpy()] = 1

                y_pred = network.forward(X_batch.numpy())
                network.backward(y_pred, y_one_hot, loss_derivative)
                network.adam(learning_rate)
                network.zero_grad()
                i += 1

        train_acc = evaluate(network, train_loader)
        test_acc = evaluate(network, test_loader)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        tqdm.write(f"Epoch {epoch + 1}, Train Accuracy: {train_acc:.6f}, Test Accuracy {test_acc:.6f}")

    return train_accuracies, test_accuracies


def evaluate(network, data_loader):
    correct = 0
    total = 0
    for batch in data_loader:
        X_batch, y_batch = batch
        y_pred = network.forward(X_batch.numpy())
        predictions = np.argmax(y_pred, axis=1)
        correct += np.sum(predictions == y_batch.numpy())
        total += y_batch.size(0)
    return correct / total
