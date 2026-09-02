"""Small dependency-free Ollama HTTP client used by the lab."""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .settings import OLLAMA_BASE_URL
except ImportError:
    from settings import OLLAMA_BASE_URL


class OllamaError(RuntimeError):
    """Raised when Ollama is unavailable or returns an invalid response."""


def _post(path: str, payload: dict) -> dict:
    request = Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise OllamaError(f"Ollama request failed at {path}: {error}") from error


def embed(texts: list[str], model: str) -> list[list[float]]:
    """Create normalized embeddings using Ollama's batch embed endpoint."""
    result = _post("/api/embed", {"model": model, "input": texts})
    embeddings = result.get("embeddings")
    if not embeddings:
        raise OllamaError("Ollama returned no embeddings. Check the embedding model name.")
    return embeddings


def chat(messages: list[dict[str, str]], model: str, json_mode: bool = False) -> str:
    """Generate a non-streaming answer from the configured Ollama chat model."""
    result = _post("/api/chat", {"model": model, "messages": messages, "stream": False, "format": "json" if json_mode else ""})
    try:
        return result["message"]["content"]
    except (KeyError, TypeError) as error:
        raise OllamaError("Ollama returned an invalid chat response.") from error


def status() -> dict[str, object]:
    """Return a lightweight health result for the UI."""
    try:
        request = Request(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags")
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {"connected": True, "models": [item.get("name") for item in payload.get("models", [])]}
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        return {"connected": False, "error": str(error)}
