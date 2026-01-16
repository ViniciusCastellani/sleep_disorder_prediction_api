import pandas as pd


def base_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"person_id", "blood_pressure"}

    if not required_cols.issubset(df.columns):
        raise ValueError("The dataframe do not contain the required columns")

    df = df.copy()
    df[["systolic", "diastolic"]] = (
        df["blood_pressure"].str.split("/", expand=True).astype(int)
    )
    df = df.drop(["person_id", "blood_pressure"], axis=1)
    return df
