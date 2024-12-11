import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt

lenet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(400, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10))


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("",
                     is_train,
                     transform=transforms.Compose([transforms.Resize((32, 32)),
                                                   transforms.ToTensor(),
                                                   transforms.ConvertImageDtype(torch.float)]),
                     download=True)
    return DataLoader(data_set, batch_size=64, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


train_data = get_data_loader(is_train=True)
test_data = get_data_loader(is_train=False)

print("initial accuracy:", evaluate(test_data, lenet))

optimizer = torch.optim.Adam(lenet.parameters(), lr=0.001)
L = nn.CrossEntropyLoss()

for epoch in range(5):
    for (x, y) in train_data:
        lenet.zero_grad()
        y_hat = lenet.forward(x)
        loss = L(y_hat, y)
        loss.backward()
        optimizer.step()
    print("epoch", epoch, "accuracy:", evaluate(test_data, lenet))
