import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from midterm_nueralnetworks.neural_network.layer import Layer, Conv2D, MaxPool2D, FlattenLayer, Linear, ActivationLayer
from midterm_nueralnetworks.neural_network.feed_forward_neural_network import FeedforwardNeuralNetwork as FNN
from midterm_nueralnetworks.neural_network.loss import get_loss_derivative, get_loss
import matplotlib.pyplot as plt


def get_data_loader(is_train):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy())
    ])
    dataset = MNIST(root="", train=is_train, transform=transform, download=True)
    return DataLoader(dataset, batch_size=32, shuffle=is_train)  # batch size tag


def train(network, train_loader, test_loader, epochs, learning_rate):
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        i = 0
        for batch in train_loader:
            if i > 1:
                break;
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

        print(f"Epoch {epoch + 1}, Train Accuracy: {train_acc:.6f}, Test Accuracy {test_acc:.6f}")

    return train_accuracies, test_accuracies


def evaluate(network, data_loader):
    correct = 0
    total = 0
    i = 0
    for batch in data_loader:
        if i > 100:
            break
        X_batch, y_batch = batch
        y_pred = network.forward(X_batch.numpy())
        predictions = np.argmax(y_pred, axis=1)
        correct += np.sum(predictions == y_batch.numpy())
        total += y_batch.size(0)
        i += 1
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
epochs = 20

train_loader = get_data_loader(True)
test_loader = get_data_loader(False)

train_accuracies, test_accuracies = train(lenet, train_loader, test_loader, epochs, learning_rate)

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
