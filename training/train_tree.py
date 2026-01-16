import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Union

def train_decision_tree(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.DataFrame | np.ndarray,
) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        max_depth=5,
        criterion="entropy",
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    return model