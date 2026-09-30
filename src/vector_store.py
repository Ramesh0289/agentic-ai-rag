from pinecone import Pinecone, ServerlessSpec

from src.config import get_settings
from src.embeddings import EmbeddingService


class VectorStore:
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = EmbeddingService()
        self.pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self.index = self._get_index()

    def _get_index(self):
        index_name = self.settings.pinecone_index_name
        existing = self.pc.list_indexes()
        names = [item["name"] for item in existing]

        if index_name not in names:
            self.pc.create_index(
                name=index_name,
                dimension=self.embeddings.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        return self.pc.Index(index_name)

    def upsert(self, chunks: list[dict]) -> int:
        total = 0
        for start in range(0, len(chunks), 50):
            batch = chunks[start:start + 50]
            vectors = self.embeddings.embed([item["text"] for item in batch])
            records = [
                {
                    "id": item["id"],
                    "values": vector,
                    "metadata": {
                        "text": item["text"],
                        "page": item["page"],
                        "source": "Ebook-Agentic-AI.pdf",
                    },
                }
                for item, vector in zip(batch, vectors)
            ]
            self.index.upsert(vectors=records)
            total += len(records)
            print(f"Indexed {total}/{len(chunks)} chunks")
        return total

    def search(self, query: str, top_k: int) -> list[dict]:
        query_vector = self.embeddings.embed([query])[0]
        result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        return [
            {
                "id": match.id,
                "text": (match.metadata or {}).get("text", ""),
                "page": int((match.metadata or {}).get("page", 0)),
                "score": float(match.score),
            }
            for match in result.matches
        ]
