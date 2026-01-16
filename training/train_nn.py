import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> MLPClassifier:
    model = MLPClassifier(
        solver="adam",
        activation="relu",
        alpha=1e-8,
        hidden_layer_sizes=(490, 490),
        random_state=1,
        max_iter=1000,
        early_stopping=True
    )

    model.fit(X_train, y_train)
    return model