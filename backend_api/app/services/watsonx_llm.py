import os

from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference


class WatsonxLLM:
    MODEL_ID = "ibm/granite-3-8b-instruct"

    def __init__(self) -> None:
        load_dotenv()

        api_key = os.getenv("WATSONX_API_KEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        url = os.getenv("WATSONX_URL")

        missing = [
            name
            for name, value in {
                "WATSONX_API_KEY": api_key,
                "WATSONX_PROJECT_ID": project_id,
                "WATSONX_URL": url,
            }.items()
            if not value
        ]
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(
                f"Missing required Watsonx environment variables: {missing_list}"
            )

        credentials = Credentials(api_key=api_key, url=url)
        self.client = APIClient(credentials=credentials, project_id=project_id)
        self.model = ModelInference(
            model_id=self.MODEL_ID,
            api_client=self.client,
            project_id=project_id,
            validate=False,
            params={
                "max_new_tokens": 800,
                "min_new_tokens": 10,
                "temperature": 0.2,  # low = more factual, less creative
                "repetition_penalty": 1.1,  # prevents repeating the same sentence
            },
        )

    def generate(self, prompt: str) -> str:
        response = self.model.generate(prompt=prompt)
        return response["results"][0]["generated_text"].strip()
