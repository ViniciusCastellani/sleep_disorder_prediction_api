import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple


def encode_tree(
    df: pd.DataFrame, target_col: str = "sleep_disorder"
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:

    df = df.copy()

    label_cols = df.select_dtypes(include="object").columns.tolist()
    label_cols = [c for c in label_cols if c != target_col]

    encoders: Dict[str, LabelEncoder] = {}

    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders
