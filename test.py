import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from feed_forward_neural_network import (
    FeedforwardNeuralNetwork, sigmoid, sigmoid_derivative
)

if __name__ == "__main__":
    # Load and preprocess the Iris dataset
    iris = load_iris()
    X = iris.data  # Input features (4 features)
    y = iris.target.reshape(-1, 1)  # Target labels (3 classes)

    # One-hot encode the target labels
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data to improve network performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the neural network with appropriate layer sizes
    nn = FeedforwardNeuralNetwork([4, 3])  # 4 input, 5 hidden, 3 output nodes

    # Train the network
    nn.train(
        X_train, y_train,
        epochs=100,
        learning_rate=0.1,
        activation_function= sigmoid,
        activation_derivative= sigmoid_derivative
    )

    # Test the trained network
    print("\nTesting Trained Network:")
    correct = 0
    for x, target in zip(X_test, y_test):
        output = nn.forward(x, sigmoid)
        predicted = np.argmax(output)  # Predicted class (index of max output)
        actual = np.argmax(target)  # Actual class (index of 1 in one-hot)
        print(f"Input: {x} -> Predicted: {predicted}, Actual: {actual}")
        if predicted == actual:
            correct += 1

    # Calculate and print the accuracy
    accuracy = correct / len(X_test) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
