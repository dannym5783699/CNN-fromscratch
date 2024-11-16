from copy import deepcopy
from typing import List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from midterm_nueralnetworks.neural_network.feed_forward_neural_network import \
    FeedforwardNeuralNetwork
from midterm_nueralnetworks.neural_network.layer import Layer
from midterm_nueralnetworks.neural_network.loss import get_loss_derivative, get_loss

class SklearnFFNN(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            layers : List[Layer],
            max_epochs : int = 100,
            learning_rate : float=1e-2,
            lambda_reg : float =1e-5,
            batch_size : int =32,
            optimizer : str ="gd",
            optimizer_kw_args=None,
            loss_function : str ="nll_softmax",
            random_state : int = None,
        ):
        self.layers = layers
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.optimizer = optimizer
        self.optimizer_kw_args = optimizer_kw_args if optimizer_kw_args is not None else {}
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.random_state = random_state

    def _get_batches(self, X, Y):
        """
        Yield batches of data for training.
        """
        if self.batch_size is None:
            yield X, Y
            return
        else:
            indices = np.arange(len(X))
            self.rng_.shuffle(indices)

            for i in range(0, len(X), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                yield X[batch_indices], Y[batch_indices]

    def _init_network(self):
        self.network_ = FeedforwardNeuralNetwork(deepcopy(self.layers))

    def _get_optimizer_method(self):
        # I hate specifying params manually but changing it will require changing the optimizer signatures
        if self.optimizer == "gd":
            friction = self.optimizer_kw_args.get("friction", 0)
            return lambda: self.network_.gd(self.learning_rate, friction, self.lambda_reg)
        elif self.optimizer == "newton":
            return lambda: self.network_.newtons_method(self.learning_rate, self.lambda_reg)
        elif self.optimizer == "adam":
            p1 = self.optimizer_kw_args.get("p1", 0.9)
            p2 = self.optimizer_kw_args.get("p2", 0.999)
            return lambda: self.network_.adam(self.learning_rate, self.lambda_reg, p1, p2)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
    
    def fit(self, X, y):

        if y.ndim == 1:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = np.arange(y.shape[1])

        self._init_network()

        if self.optimizer_kw_args is None:
            self.optimizer_kw_args = {}
        
        self.rng_ = np.random.default_rng(self.random_state)

        optimizer = self._get_optimizer_method()
        loss = get_loss[self.loss_function]
        loss_deriv = get_loss_derivative[self.loss_function]

        
        self.train_loss_ = np.zeros(self.max_epochs)
        self.test_loss_ = np.zeros(self.max_epochs)
        
        for epoch in range(self.max_epochs):
            batch_train_losses = []

            for X_batch, y_batch in self._get_batches(X, y):
                y_pred = self.network_.forward(X_batch)
                self.network_.backward(y_pred, y_batch, loss_deriv)
                optimizer()
                self.network_.zero_grad()

    def predict(self, X):
        """Predict class labels."""
        y_prob = self.predict_proba(X)
        if len(self.classes_) == 1:
            y_pred = (y_prob[:, 0] > 0.5).astype(int)
            return y_pred
        else:
            class_indices = np.argmax(y_prob, axis=1)
            one_hot_predictions = np.zeros_like(y_prob)
            one_hot_predictions[np.arange(len(class_indices)), class_indices] = 1
            return one_hot_predictions

    def predict_proba(self, X):
        """Predict probabilities for each class."""
        y_prob = self.network_.forward(X)
        self.network_.zero_grad()
        return y_prob
    
    def score(self, X, y):
        loss = get_loss[self.loss_function]
        y_pred = self.predict_proba(X)
        return loss(y, y_pred)
    
    def _encode_labels(self, y):
        if len(self.classes_) == 2:  # Binary classification
            return (y == self.classes_[1]).astype(int)[:, np.newaxis]
        return y