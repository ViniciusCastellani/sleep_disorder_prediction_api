from typing import Dict, Any

from predict.predict_tree import predict_tree
from predict.predict_nn import predict_nn


def predict_combined(
    input_data: Dict[str, Any],
    tree_model,
    nn_model,
    scaler,
    target_encoder,
    encoders,
    dummy_columns,
    threshold: float = 0.85,
) -> Dict[str, Any]:

    result_tree = predict_tree(input_data, tree_model, encoders)

    if result_tree["probability"] >= threshold:
        result_tree["model_used"] = "decision_tree"
        return result_tree

    result_nn = predict_nn(input_data, nn_model, scaler, target_encoder, dummy_columns)

    if result_nn["probability"] > result_tree["probability"]:
        result_nn["model_used"] = "neural_network"
        return result_nn
    else:
        result_tree["model_used"] = "decision_tree"
        return result_tree
