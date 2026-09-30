from pathlib import Path
import hashlib

import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings


def download_pdf() -> Path:
    settings = get_settings()
    path = Path(settings.pdf_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(settings.pdf_url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def load_and_split_pdf(pdf_path: str) -> list[dict]:
    settings = get_settings()
    reader = PdfReader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        for chunk_number, chunk in enumerate(splitter.split_text(text)):
            chunk = chunk.strip()
            if not chunk:
                continue
            digest = hashlib.sha1(
                f"{page_number}-{chunk_number}-{chunk}".encode("utf-8")
            ).hexdigest()[:12]
            chunks.append({
                "id": f"page-{page_number}-chunk-{chunk_number}-{digest}",
                "text": chunk,
                "page": page_number,
            })
    return chunks
