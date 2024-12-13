from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from midterm_nueralnetworks.neural_network.layer import Layer, Conv2D, MaxPool2D, FlattenLayer, Linear, ActivationLayer
from midterm_nueralnetworks.neural_network.feed_forward_neural_network import FeedforwardNeuralNetwork as FNN
from midterm_nueralnetworks.neural_network.loss import get_loss_derivative, get_loss
from midterm_nueralnetworks.final_test_cases.test_utils import train
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import random


def get_data_loader(is_train, subset_size=None):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy())
    ])
    dataset = MNIST(root="", train=is_train, transform=transform, download=True)

    if subset_size:
        indices = random.sample(range(len(dataset)), subset_size)
        dataset = Subset(dataset, indices)
    return dataset

train_loader = get_data_loader(True, 256)
test_loader = get_data_loader(False, 256)

results = {
    "train": {},
    "test": {}
}

for i in range(6):
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
    learning_rate: float
    epochs = 5
    batch_size: int
    if i == 0:
        learning_rate = 0.01
        batch_size = 64
    elif i == 1:
        learning_rate = 0.01
        batch_size = 100
    elif i == 2:
        learning_rate = 0.01
        batch_size = 128
    elif i == 3:
        learning_rate = 0.001
        batch_size = 64
    elif i == 4:
        learning_rate = 0.001
        batch_size = 100
    else:
        learning_rate = 0.001
        batch_size = 128

    train_data = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_loader, batch_size=batch_size, shuffle=False)

    train_accuracies, test_accuracies = train(lenet, train_data, test_data, epochs, learning_rate, loss_derivative)

    key = f"LR={learning_rate}, Batch={batch_size}"
    results["train"][key] = train_accuracies
    results["test"][key] = test_accuracies

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Train vs Test Accuracy Over Epochs Batch Size={batch_size} MNSIT')
    plt.legend()
    plt.grid(True)
    plt.show()

plt.figure(figsize=(12, 8))
for key, acc in results["train"].items():
    plt.plot(range(1, len(acc) + 1), acc, label=f"Train {key}")
for key, acc in results["test"].items():
    plt.plot(range(1, len(acc) + 1), acc, linestyle='--', label=f"Test {key}")

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Combined Accuracy Over Epochs MNSIT')
plt.legend()
plt.grid(True)
plt.show()
