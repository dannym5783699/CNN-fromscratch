import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from vanderpol_data import Phi, generate_data

from midterm_nueralnetworks.neural_network.feed_forward_neural_network import FeedforwardNeuralNetwork
from midterm_nueralnetworks.neural_network.layer import Layer
from midterm_nueralnetworks.neural_network.loss import mse_derivative

def get_batches(X, Y, batch_size):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    for i in range(0, len(X), batch_size):
        batch_indices = indices[i:i+batch_size]
        yield X[batch_indices], Y[batch_indices]

if __name__ == "__main__":
    print("Preparing data")
    X, Y = generate_data(101, -3, 3)
    print(X.shape, Y.shape)
    print(X[:5], Y[:5])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42)
    


    net = FeedforwardNeuralNetwork([
            Layer(2, 64, "relu"),
            Layer(64, 64, "relu"),
            Layer(64, 64, "relu"),
            Layer(64, 2, "linear")
        ]
    )

    MAX_EPOCHS = 2000
    LR = 1e-5
    BATCH_SIZE = 32

    train_losses = np.zeros(MAX_EPOCHS)
    test_losses = np.zeros(MAX_EPOCHS)

    print("training model")
    print(f"{X_train.shape=}, {Y_train.shape=}")
    print(f"{X_test.shape=}, {Y_test.shape=}")
    for epoch in range(MAX_EPOCHS):
        batch_losses = []
        for i, (x_batch, y_batch) in enumerate(get_batches(X_train, Y_train, BATCH_SIZE)):
            y_hat = net.forward(x_batch)
            batch_losses.append(mean_squared_error(y_hat, y_batch))
            net.backward(y_hat, y_batch, mse_derivative)
            net.gd(LR, 0.1)
            net.zero_grad()            
        mean_train_loss = np.mean(batch_losses)
        test_loss = mean_squared_error(net.forward(X_test), Y_test)
        net.zero_grad()

        train_losses[epoch] = mean_train_loss
        test_losses[epoch] = test_loss

        if epoch % 10 == 0:
            print(f"Epoch {str(epoch).zfill(2)}, Mean Train Loss: {mean_train_loss}, Test Loss: {test_loss}" )
        
        

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.show()

    # test 1
    x0 = np.array([1.25, 2.35])
    for i in range(150):
        y = Phi(x0)
        plt.plot(y[0], y[1], 'b.')
        x0 = y

    for i in range(150):
        y = net.forward(x0.reshape(1, -1))[0]
        plt.plot(y[0], y[1], 'r.')
        x0 = y


    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.show()
