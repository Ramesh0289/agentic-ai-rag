import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_settings


class EmbeddingService:
    def __init__(self):
        settings = get_settings()
        self.provider = settings.embedding_provider.lower()
        self.settings = settings

        if self.provider == "local":
            self.model = SentenceTransformer(settings.embedding_model)
        elif self.provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embedding mode.")
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.openai_api_key)
        else:
            raise ValueError("EMBEDDING_PROVIDER must be 'local' or 'openai'.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "local":
            vectors = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(vectors).tolist()

        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        if self.provider == "local":
            return self.model.get_sentence_embedding_dimension()
        return 1536
