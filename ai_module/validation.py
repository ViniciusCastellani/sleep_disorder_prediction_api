from typing import Dict, Set

ALLOWED_FIELDS: Set[str] = {
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Blood Pressure",
    "Heart Rate",
    "Daily Steps",
}


def validate_extraction(data: Dict) -> Dict:
    extracted = data.get("extracted", {})

    cleaned = {"extracted": {}, "missing_fields": []}

    for key, value in extracted.items():
        if key in ALLOWED_FIELDS:
            cleaned["extracted"][key] = value

    required_user_fields = ALLOWED_FIELDS

    cleaned["missing_fields"] = sorted(
        required_user_fields - cleaned["extracted"].keys()
    )

    return cleaned


def normalize_fields(extracted: Dict) -> Dict:
    normalized = extracted.copy()

    if "Gender" in normalized and isinstance(normalized["Gender"], str):
        normalized["Gender"] = normalized["Gender"].capitalize()

    if "BMI Category" in normalized and isinstance(normalized["BMI Category"], str):
        normalized["BMI Category"] = normalized["BMI Category"].capitalize()

    if "Occupation" in normalized and isinstance(normalized["Occupation"], str):
        normalized["Occupation"] = normalized["Occupation"].capitalize()

    return normalized


def split_blood_pressure(extracted: Dict) -> Dict:
    extracted = extracted.copy()

    bp = extracted.get("Blood Pressure")

    if isinstance(bp, str) and "/" in bp:
        try:
            systolic, diastolic = bp.split("/")
            extracted["systolic"] = int(systolic.strip())
            extracted["diastolic"] = int(diastolic.strip())
        except ValueError:
            pass

    extracted.pop("Blood Pressure", None)
    return extracted
