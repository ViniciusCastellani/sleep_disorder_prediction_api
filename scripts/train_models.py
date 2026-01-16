from typing import Dict, Any
import joblib
import os

from data.fetch_data import fetch_sql_sleep_data
from data.db_config import load_env_variables

from preprocessing.base_preprocessing import base_preprocessing
from preprocessing.encode_tree import encode_tree
from preprocessing.encode_nn import encode_nn

from training.prepare_data import prepare_data
from training.train_tree import train_decision_tree
from training.train_nn import train_neural_network

from models.save_models import save_models


def train_models() -> Dict[str, Any]:
    db_config = load_env_variables()
    raw_df = fetch_sql_sleep_data(db_config)
    
    preprocessed_df = base_preprocessing(raw_df)
    
    encoded_tree_df, tree_encoders = encode_tree(preprocessed_df)
    encoded_nn_df, dummy_columns = encode_nn(preprocessed_df)

    tree_data = prepare_data(encoded_tree_df, model_type = "tree")
    nn_data = prepare_data(encoded_nn_df, model_type = "nn")

    tree_model = train_decision_tree(
        X_train = tree_data["x_train_bal"],
        y_train = tree_data["y_train_bal"],
    )

    nn_model = train_neural_network(
        X_train = nn_data["x_train_bal"],
        y_train = nn_data["y_train_bal"],
    )

    return {
        "tree_model": tree_model,
        "tree_encoders": tree_encoders,

        "nn_model": nn_model,
        "nn_scaler": nn_data["scaler"],
        "nn_target_encoder": nn_data["target_encoder"],
        "nn_dummy_columns": dummy_columns
    }


if __name__ == "__main__":
    results = train_models()
    save_models(results)
    print("Modelos treinados com sucesso!")
    print(results.keys())
   