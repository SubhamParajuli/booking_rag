from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    groq_api_key: str
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite:///./app.db"
    qdrant_path: str = "./qdrant_data"
    collection_name: str = "knowledge_base"
    llm_model: str = "qwen/qwen3.8-27b"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()