import os
import joblib
from typing import Dict, Any


def save_models(models: Dict[str, Any], base_path: str = "models") -> None:
    os.makedirs(base_path, exist_ok=True)

    # Tree
    tree_path = os.path.join(base_path, "tree")
    os.makedirs(tree_path, exist_ok=True)

    joblib.dump(models["tree_model"], f"{tree_path}/model.joblib")
    joblib.dump(models["tree_encoders"], f"{tree_path}/encoders.joblib")

    # Neural Network
    nn_path = os.path.join(base_path, "nn")
    os.makedirs(nn_path, exist_ok=True)

    joblib.dump(models["nn_model"], f"{nn_path}/model.joblib")
    joblib.dump(models["nn_scaler"], f"{nn_path}/scaler.joblib")
    joblib.dump(models["nn_target_encoder"], f"{nn_path}/target_encoder.joblib")
    joblib.dump(models["nn_dummy_columns"], f"{nn_path}/dummy_columns.joblib")
