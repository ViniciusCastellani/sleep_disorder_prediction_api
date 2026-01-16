from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from ai_module.selector import AISelector
from ai_module.validation import validate_extraction, normalize_fields, split_blood_pressure
from predict.predict_combined import predict_combined
from predict.feature_mapper import map_to_model_features
from models.load_models import load_models

app = Flask(__name__)
CORS(app)

MODELS = load_models()

TREE_MODEL = MODELS["tree_model"]
TREE_ENCODERS = MODELS["tree_encoders"]

NN_MODEL = MODELS["nn_model"]
NN_SCALER = MODELS["nn_scaler"]
NN_TARGET_ENCODER = MODELS["nn_target_encoder"]
NN_DUMMY_COLUMNS = MODELS["nn_dummy_columns"]

@app.route("/predict", methods=["POST"])
def predict_sleep_disorder():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Field 'text' is required"}), 400

        ai = AISelector(data["text"])
        extraction_result = ai.extract_information()
        print(extraction_result)

        validated = validate_extraction(extraction_result)
        
        if validated["missing_fields"]:
            return jsonify({
                "status": "incomplete",
                "missing_fields": validated["missing_fields"]
            }), 200
        
        extracted = normalize_fields(validated["extracted"])
        extracted = split_blood_pressure(extracted)
        print(extracted)

        model_input = map_to_model_features(extracted)

        prediction = predict_combined(
            input_data=model_input,
            tree_model=TREE_MODEL,
            nn_model=NN_MODEL,
            scaler=NN_SCALER,
            target_encoder=NN_TARGET_ENCODER,
            encoders=TREE_ENCODERS,
            dummy_columns=NN_DUMMY_COLUMNS
        )

        return jsonify({
            "status": "success",
            "input": extracted,
            "prediction": prediction
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "online"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)