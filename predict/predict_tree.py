import pandas as pd
import numpy as np
from typing import Dict, Any

CLASS_MAPPING = {0: "Insomnia", 1: "None", 2: "Sleep Apnea"}


def predict_tree(
    input_data: Dict[str, Any], model, encoders: Dict[str, Any]
) -> Dict[str, Any]:

    df = pd.DataFrame([input_data])

    for col, encoder in encoders.items():
        if col in df:
            df[col] = df[col].apply(
                lambda x: (
                    encoder.transform([x])[0] if x in encoder.classes_ else "Other"
                )
            )

    proba = model.predict_proba(df)[0]
    class_idx = int(np.argmax(proba))
    confidence = float(proba[class_idx])

    return {
        "class_id": int(class_idx),
        "class_name": CLASS_MAPPING[class_idx],
        "probability": round(confidence, 4),
    }
