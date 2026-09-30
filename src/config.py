from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    pinecone_api_key: str
    pinecone_index_name: str = "agentic-ai-index"

    llm_provider: str = "ollama"
    embedding_provider: str = "local"
    ollama_model: str = "llama3.2:3b"
    embedding_model: str = "all-MiniLM-L6-v2"

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    top_k: int = 3
    chunk_size: int = 1000
    chunk_overlap: int = 200
    relevance_threshold: float = 0.35

    pdf_url: str = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
    pdf_path: str = "data/Ebook-Agentic-AI.pdf"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
