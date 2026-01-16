import pandas as pd
from typing import List, Tuple


def encode_nn(
    df: pd.DataFrame,
    target_col: str = "sleep_disorder",
    dummy_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:

    df = df.copy()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if dummy_columns is None:
        X = pd.get_dummies(X, drop_first=True)
        dummy_columns = X.columns.tolist()
    else:
        X = pd.get_dummies(X, drop_first=True)
        X = X.reindex(columns=dummy_columns, fill_value=0)

    df_encoded = pd.concat([X, y], axis=1)

    return df_encoded, dummy_columns
