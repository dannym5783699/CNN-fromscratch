import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from midterm_nueralnetworks.neural_network.layer import Linear
from midterm_nueralnetworks.neural_network.sklearn_classifier_wrapper import \
    SklearnFFNN

EXPERIMENT_NAME = "titanic"

def load_titanic():
    """https://openml.org/search?type=data&status=active&sort=nr_of_downloads&id=42438"""
    x_matrix, y = fetch_openml(data_id=42438, return_X_y=True)
    return x_matrix, y

def clean_params(params):
    updated_params = {key.replace("model__", ""): value for key, value in params.items()}
    updated_params.pop("random_state")
    return updated_params

if __name__ == "__main__":
    results_dir = Path("proj3_results").resolve()
    results_dir.mkdir(exist_ok=True)

    X, y = load_titanic()
    y = np.array(y).reshape(-1, 1)

    layers = [
        Linear(X.shape[1], 16, "relu"),
        Linear(16, 16, "relu"),
        Linear(16, 8, "relu"),
        Linear(8, 1, "sigmoid", final_layer=True)
    ]

    # Create the pipeline
    pl = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SklearnFFNN(layers=layers))
    ])


    optimizer_param_grids = [
        {
            "model__optimizer": ["gd"],
            "model__optimizer_kw_args": [{"friction": 0.9}, {"friction": 0.99}, {"friction": 0.5}, {"friction": 0}],
        },
        {
            "model__optimizer": ["newton"],
        },
        {
            "model__optimizer": ["adam"],
            "model__optimizer_kw_args": [{"p1": 0.9, "p2": 0.999}, {"p1": 0.7, "p2": 0.6}],
        }
    ]

    base_param_grid = {
        "model__max_epochs": [10, 50, 100, 200],
        "model__learning_rate": [1e-2, 1e-5],
        "model__lambda_reg": [1e-5],
        "model__batch_size": [32, 64, 128],
        "model__random_state": [42],
        "model__loss_function": ["cross_entropy"]
    }
    
    combined_grid = []

    for optimizer_params in optimizer_param_grids:
        optimizer_params.update(**base_param_grid)

    clf = GridSearchCV(pl, optimizer_param_grids, cv=3, n_jobs=-1, scoring="accuracy", refit="accuracy", verbose=2)
    clf.fit(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf.best_estimator_.fit(X_train, y_train)
    y_pred = clf.best_estimator_.predict(X_test)
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

    print(f"Best params: {clf.best_params_}")
    fig, ax = plt.subplots(figsize=(12, 12))
    cm.plot(ax=ax)
    accuracy = accuracy_score(y_test, y_pred)
    ax.set_title(f"Confusion Matrix for Best {EXPERIMENT_NAME} Model\nAccuracy: {accuracy:.2f}")
    best_params = clean_params(clf.best_params_)
    plt.tight_layout()
    plt.savefig(results_dir / f"best_{EXPERIMENT_NAME}_confusion_matrix.png")

    results = pd.DataFrame(clf.cv_results_)
    results.to_csv(results_dir / f"{EXPERIMENT_NAME}_results.csv")
