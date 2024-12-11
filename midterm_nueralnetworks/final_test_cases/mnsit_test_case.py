import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from midterm_nueralnetworks.neural_network.layer import Layer, Conv2D, MaxPool2D, FlattenLayer, Linear, ActivationLayer
from midterm_nueralnetworks.neural_network.feed_forward_neural_network import FeedforwardNeuralNetwork as FNN
from midterm_nueralnetworks.neural_network.loss import get_loss_derivative, get_loss


def get_data_loader(is_train):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy().flatten())
    ])
    dataset = MNIST(root="", train=is_train, transform=transform, download=True)
    return DataLoader(dataset, batch_size=64, shuffle=is_train)  # batch size tag


def train(network, train_loader, test_loader, epochs, learning_rate):
    for epoch in range(epochs):
        for batch in train_loader:
            X_batch, y_batch = batch
            y_one_hot = np.zeros((y_batch.size(0), 10))
            y_one_hot[np.arange(y_batch.size(0)), y_batch.numpy()] = 1

            y_pred = network.forward(X_batch.numpy())
            network.backward(y_pred, y_one_hot, loss_derivative)
            network.adam(learning_rate)
            network.zero_grad()

        train_acc = evaluate(network, train_loader)
        test_acc = evaluate(network, test_loader)
        print(f"Epoch {epoch + 1}, Train Accuracy: {train_acc:.6f}, Test Accuracy {test_acc:.6f}")


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


layers = [
    Conv2D(in_channels=1, out_channels=6, kernel_size=5),
    ActivationLayer(activation="relu"),
    MaxPool2D(kernel_size=2, stride=2),
    Conv2D(in_channels=6, out_channels=16, kernel_size=5),
    ActivationLayer(activation="relu"),
    MaxPool2D(kernel_size=2, stride=2),
    FlattenLayer(),
    Linear(input_size=400, output_size=120, activation="relu"),
    Linear(input_size=120, output_size=84, activation="relu"),
    Linear(input_size=84, output_size=10, activation="softmax", final_layer=True)]

lenet = FNN(layers)

loss_func = "nll_softmax"
loss_derivative = get_loss_derivative[loss_func]
learning_rate = 0.001
epochs = 5

train_loader = get_data_loader(True)
test_loader = get_data_loader(False)

train(lenet, train_loader, test_loader, epochs, learning_rate)
