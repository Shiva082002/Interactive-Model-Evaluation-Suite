"""Persistent document ingestion and Ollama embedding retrieval."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

try:
    from .ollama_client import embed
    from .settings import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, INDEX_DIR, OLLAMA_EMBED_MODEL, TOP_K
except ImportError:
    from ollama_client import embed
    from settings import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, INDEX_DIR, OLLAMA_EMBED_MODEL, TOP_K


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size])
        if chunk:
            chunks.append(chunk)
    return chunks


def read_documents() -> list[dict[str, str]]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    documents = []
    for path in sorted(DOCS_DIR.glob("**/*")):
        if path.suffix.lower() in {".md", ".txt"} and path.is_file():
            for number, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
                documents.append({"source": str(path.relative_to(DOCS_DIR)), "chunk": str(number), "text": chunk})
    return documents


def build_index() -> dict[str, Any]:
    documents = read_documents()
    if not documents:
        raise ValueError(f"Add .md or .txt files to {DOCS_DIR} before indexing.")
    vectors = embed([item["text"] for item in documents], OLLAMA_EMBED_MODEL)
    matrix = np.asarray(vectors, dtype=np.float32)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", matrix)
    (INDEX_DIR / "metadata.json").write_text(json.dumps(documents, indent=2), encoding="utf-8")
    return {"chunks": len(documents), "dimensions": int(matrix.shape[1]), "embedding_model": OLLAMA_EMBED_MODEL}


def index_exists() -> bool:
    return (INDEX_DIR / "embeddings.npy").exists() and (INDEX_DIR / "metadata.json").exists()


def search(query: str, limit: int = TOP_K) -> list[dict[str, Any]]:
    if not index_exists():
        build_index()
    metadata = json.loads((INDEX_DIR / "metadata.json").read_text(encoding="utf-8"))
    matrix = np.load(INDEX_DIR / "embeddings.npy")
    query_vector = np.asarray(embed([query], OLLAMA_EMBED_MODEL)[0], dtype=np.float32)
    scores = matrix @ query_vector / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector) + 1e-8)
    selected = np.argsort(scores)[::-1][:limit]
    return [{**metadata[index], "score": round(float(scores[index]), 4)} for index in selected]
