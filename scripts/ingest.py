from pathlib import Path

from src.config import get_settings
from src.ingestion import download_pdf, load_and_split_pdf
from src.vector_store import VectorStore


def main():
    settings = get_settings()
    pdf_path = Path(settings.pdf_path)

    if not pdf_path.exists():
        download_pdf()

    print("Reading and splitting PDF...")
    chunks = load_and_split_pdf(str(pdf_path))
    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings and storing vectors...")
    total = VectorStore().upsert(chunks)
    print(f"Done. Stored {total} chunks in Pinecone.")


if __name__ == "__main__":
    main()
