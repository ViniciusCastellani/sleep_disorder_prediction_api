import yaml
import json
from ai_module.gemini_client import call_gemini


class AISelector:
    def __init__(self, user_input: str):
        self.user_input = user_input
        self.extract_prompt_path = "ai_module/prompt/extract_sleep.yaml"

    def extract_information(self) -> dict:
        with open(self.extract_prompt_path, "r", encoding="utf-8") as f:
            prompt_data = yaml.safe_load(f)

        system_prompt = prompt_data["system"]["content"]
        user_prompt = prompt_data["user"]["content"].format(user_input=self.user_input)

        full_prompt = f"""SYSTEM:{system_prompt}
                          USER:{user_prompt}
                       """

        output = call_gemini(full_prompt)
        cleaned = self._clean_json_output(output)

        return json.loads(cleaned)

    def _clean_json_output(self, text: str) -> str:
        text = text.strip()

        if text.startswith("```"):
            text = text.split("```", 1)[1].strip()
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0].strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("Model output does not contain valid JSON")

        return text[start : end + 1]
