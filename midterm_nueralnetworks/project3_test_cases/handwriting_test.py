from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import pandas as pd

from midterm_nueralnetworks.neural_network.sklearn_classifier_wrapper import SklearnFFNN
from midterm_nueralnetworks.neural_network.layer import Layer

if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

    layers = [
        Layer(input_size=64, output_size=64, activation="relu"),
        Layer(input_size=64, output_size=32, activation="relu"),
        Layer(input_size=32, output_size=10, activation="softmax", final_layer=True)
    ]

    # Create the pipeline
    pl = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SklearnFFNN(layers=layers))
    ])


    optimizer_param_grids = [
        {
            "model__optimizer": ["gd"],
            "model__optimizer_kw_args": [{"friction": 0.9}, {"friction": 0.99}],
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
        "model__max_epochs": [10],
        "model__learning_rate": [1e-2],
        "model__lambda_reg": [1e-5],
        "model__batch_size": [64, 128],
        "model__random_state": [42],
    }
    
    combined_grid = []

    for optimizer_params in optimizer_param_grids:
        optimizer_params.update(**base_param_grid)

    clf = GridSearchCV(pl, optimizer_param_grids, cv=2, n_jobs=-1, scoring="accuracy")
    clf.fit(X, y)

    results = pd.DataFrame(clf.cv_results_)
    results.to_csv("results.csv")
