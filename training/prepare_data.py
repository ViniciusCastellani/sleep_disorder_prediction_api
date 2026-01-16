from typing import Dict, Any
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def prepare_data(
    df: pd.DataFrame,
    model_type: str = "tree",
    target_col: str = "sleep_disorder",
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    df = df.copy()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    target_encoder = None

    if model_type == "nn":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = None

    if model_type == "nn":
        scaler = MinMaxScaler()

        x_train = scaler.fit_transform(x_train)  
        x_test = scaler.transform(x_test)     

    smote = SMOTE(random_state=random_state)
    x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_train_bal": x_train_bal,
        "y_train_bal": y_train_bal,
        "scaler": scaler,
        "target_encoder": target_encoder    
    }