"""Build the local document index with Ollama embeddings."""
from __future__ import annotations

try:
    from .database import initialize_database
    from .ollama_client import status
    from .rag_pipeline import build_index
except ImportError:
    from database import initialize_database
    from ollama_client import status
    from rag_pipeline import build_index


def main() -> dict[str, object]:
    ollama_info = status()
    database_info = initialize_database(rows=20000)
    index_info = build_index()
    print(f"Ollama: {ollama_info}")
    print(f"Database: {database_info}")
    print(f"Index: {index_info}")
    print("Add your .md or .txt files under docs/ and rerun this command after changes.")
    return {"ollama": ollama_info, "database": database_info, "index": index_info}


if __name__ == "__main__":
    main()
