import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    chroma_persist_directory: str = os.getenv(
        "CHROMA_PERSIST_DIRECTORY", "./embeddings"
    )
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    watsonx_api_key: str = os.getenv("WATSONX_API_KEY", os.getenv("IBM_WATSONX_APIKEY", ""))
    watsonx_project_id: str = os.getenv(
        "WATSONX_PROJECT_ID", os.getenv("IBM_WATSONX_PROJECT_ID", "")
    )
    watsonx_url: str = os.getenv("WATSONX_URL", os.getenv("IBM_WATSONX_URL", ""))


settings = Settings()
