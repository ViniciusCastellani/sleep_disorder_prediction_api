from typing import Dict

FIELD_MAP = {
    "Gender": "gender",
    "Age": "age",
    "Occupation": "occupation",
    "Sleep Duration": "sleep_duration",
    "Quality of Sleep": "quality_of_sleep",
    "Physical Activity Level": "physical_activity_level",
    "Stress Level": "stress_level",
    "BMI Category": "bmi_category",
    "Heart Rate": "heart_rate",
    "Daily Steps": "daily_steps",
    "systolic": "systolic",
    "diastolic": "diastolic",
}


def map_to_model_features(extracted: dict) -> dict:
    mapped = {}

    for human_key, model_key in FIELD_MAP.items():
        if human_key in extracted:
            mapped[model_key] = extracted[human_key]
    return mapped
