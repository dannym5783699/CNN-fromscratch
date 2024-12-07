import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from midterm_nueralnetworks.neural_network.feed_forward_neural_network import FeedforwardNeuralNetwork
from midterm_nueralnetworks.neural_network.layer import Layer
from midterm_nueralnetworks.neural_network.loss import NLL_derivative_softmax


def generate_titanic_data():
    """
    Load and preprocess the Titanic dataset.
    - Use 'sex' and 'age' as input features.
    - 'Survived' as the target variable.
    """
    # Load Titanic dataset
    titanic = fetch_openml('titanic', version=1, as_frame=True)
    x = titanic.data
    y = titanic.target.to_numpy(dtype=int)

    # Process 'sex' column: female -> 1, male -> 0, NaN -> -1 (treated separately)
    sex_array = x['sex'].to_numpy()
    conv_sex_array = np.where(pd.isnull(sex_array), -1, np.where(sex_array == 'female', 1, 0))

    # Process 'age' column: replace NaN with mean age
    age_array = x['age'].to_numpy()
    age_array = np.nan_to_num(age_array, nan=np.mean(age_array[~np.isnan(age_array)]))

    # Combine features
    x = np.column_stack((conv_sex_array, age_array))

    # One-hot encode the target (binary classification)
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

    return x, y


def get_batches(X, Y, batch_size):
    """
    Yield batches of data for training.
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for i in range(0, len(X), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield X[batch_indices], Y[batch_indices]


def nll_loss(output, target):
    """
    Calculate the Negative Log Likelihood Loss for a batch.
    """
    output = np.clip(output, 1e-10, 1 - 1e-10)  # Avoid log(0) or log(1)
    return -np.sum(target * np.log(output)) / target.shape[0]


def batch_size_test_xavier_uniform():
    """
    Train and evaluate the Titanic dataset using different batch sizes.
    """
    # Prepare data
    X, Y = generate_titanic_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)

    # Define parameters
    MAX_EPOCHS = 1000
    LR = 1e-4
    batch_sizes = [8, 32, 64, 128]

    # Prepare plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

    for batch_size, ax in zip(batch_sizes, axs.flatten()):
        # Initialize the network
        net = FeedforwardNeuralNetwork([
            Layer(2, 10, "tanh", weight_init="xavier_uniform"),  # Input layer: 2 features, 10 neurons
            Layer(10, 10, "tanh", weight_init="xavier_uniform"),  # Hidden layer: 10 neurons
            Layer(10, 2, "softmax", final_layer=True, weight_init="xavier_uniform")  # Output layer: 2 classes, softmax
        ])

        train_losses = np.zeros(MAX_EPOCHS)
        test_losses = np.zeros(MAX_EPOCHS)

        print(f"Training model with batch size {batch_size}")
        for epoch in range(MAX_EPOCHS):
            batch_losses = []
            for x_batch, y_batch in get_batches(X_train, Y_train, batch_size):
                y_pred = net.forward(x_batch)
                batch_losses.append(nll_loss(y_pred, y_batch))
                net.backward(y_pred, y_batch, NLL_derivative_softmax)
                net.adam(LR,.8,.8)
                net.zero_grad()
            mean_train_loss = np.mean(batch_losses)
            test_loss = nll_loss(net.forward(X_test), Y_test)
            train_losses[epoch] = mean_train_loss
            test_losses[epoch] = test_loss

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Train Loss: {mean_train_loss}, Test Loss: {test_loss}")

        # Plot losses
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


if __name__ == "__main__":
    batch_size_test_xavier_uniform()
