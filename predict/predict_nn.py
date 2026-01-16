import pandas as pd
import numpy as np
from typing import Dict, Any


def predict_nn(
    input_data: Dict[str, Any], model, scaler, target_encoder, dummy_columns
) -> Dict[str, Any]:

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=dummy_columns, fill_value=0)

    X_scaled = scaler.transform(df)

    proba = model.predict_proba(X_scaled)[0]
    class_idx = int(np.argmax(proba))
    confidence = float(proba[class_idx])

    class_name = target_encoder.inverse_transform([class_idx])[0]

    return {
        "class_id": int(class_idx),
        "class_name": class_name,
        "probability": round(confidence, 4),
    }
