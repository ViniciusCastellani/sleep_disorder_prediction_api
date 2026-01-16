import joblib
from typing import Dict, Any


def load_models(base_path: str = "models") -> Dict[str, Any]:
    tree_path = f"{base_path}/tree"
    nn_path = f"{base_path}/nn"

    return {
        "tree_model": joblib.load(f"{tree_path}/model.joblib"),
        "tree_encoders": joblib.load(f"{tree_path}/encoders.joblib"),

        "nn_model": joblib.load(f"{nn_path}/model.joblib"),
        "nn_scaler": joblib.load(f"{nn_path}/scaler.joblib"),
        "nn_target_encoder": joblib.load(f"{nn_path}/target_encoder.joblib"),
        "nn_dummy_columns": joblib.load(f"{nn_path}/dummy_columns.joblib"),
    }
