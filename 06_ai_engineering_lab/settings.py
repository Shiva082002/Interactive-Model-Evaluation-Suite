"""Configuration for the local Ollama AI application."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DOCS_DIR = PROJECT_DIR / "docs"
DATA_DIR = PROJECT_DIR / "data"
INDEX_DIR = PROJECT_DIR / "index"
DATABASE_PATH = DATA_DIR / "analytics.db"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma3:12b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TOP_K = 5
