from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application settings
    app_name: str = "SmartCampus Assistant"
    debug: bool = True

    # LLM settings
    llm_model: str = "gpt-4o-mini"
    openai_api_key: SecretStr | None = None

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector database settings
    chroma_persist_dir: Path = Path("storage/chroma_db")

    #cross- encoder
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
