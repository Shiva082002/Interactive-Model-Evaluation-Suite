"""Deterministic local building blocks for RAG and NL2SQL demos."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import re
import sqlite3
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Document:
    """A source document with stable metadata for grounded answers."""

    content: str
    source: str


KNOWLEDGE_BASE = [
    Document("RAG retrieves relevant source passages before generation, reducing unsupported answers.", "rag.md"),
    Document("Vector embeddings represent text as numeric coordinates so semantically related text can be compared.", "embeddings.md"),
    Document("Prompt templates separate instructions, context, examples, and the user's question for repeatable behavior.", "prompting.md"),
    Document("LangGraph models multi-step AI workflows as stateful nodes and explicit conditional edges.", "langgraph.md"),
]


class LocalEmbedder:
    """TF-IDF vector embeddings with an optional transformer backend."""

    def __init__(self, model_name: str | None = None) -> None:
        self.backend = "tfidf"
        self._model: Any = None
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
                self.backend = "sentence-transformers"
            except (ImportError, OSError):
                self._model = None
        self._vectorizer = TfidfVectorizer(stop_words="english")

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            return np.asarray(self._model.encode(texts, normalize_embeddings=True))
        return self._vectorizer.fit_transform(texts).toarray()

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            return np.asarray(self._model.encode(texts, normalize_embeddings=True))
        return self._vectorizer.transform(texts).toarray()


class Retriever:
    """Simple cosine retrieval over a small, inspectable corpus."""

    def __init__(self, documents: list[Document] | None = None) -> None:
        self.documents = documents or KNOWLEDGE_BASE
        self.embedder = LocalEmbedder()
        self.matrix = self.embedder.fit_transform([doc.content for doc in self.documents])

    def search(self, query: str, limit: int = 2) -> list[Document]:
        query_vector = self.embedder.transform([query])[0]
        norms = np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(query_vector)
        scores = np.divide(self.matrix @ query_vector, norms, out=np.zeros_like(norms), where=norms != 0)
        indices = np.argsort(scores)[::-1][:limit]
        return [self.documents[index] for index in indices if scores[index] > 0]


PROMPT_TEMPLATES = {
    "zero_shot": "Answer the question clearly and briefly.\nQuestion: {question}",
    "few_shot": "Example: RAG -> retrieve context, then generate.\nQuestion: {question}",
    "grounded": "Use only the context below. Say you do not know when it is insufficient.\nContext:\n{context}\nQuestion: {question}",
}


def build_prompt(question: str, style: str = "grounded", context: str = "") -> str:
    """Render a named prompt strategy with explicit, inspectable inputs."""
    if style not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt style: {style}")
    template = PROMPT_TEMPLATES[style]
    try:
        prompt_type = import_module("langchain_core.prompts").ChatPromptTemplate
        return prompt_type.from_template(template).invoke({"question": question, "context": context}).to_string()
    except ImportError:
        return template.format(question=question, context=context)


def answer_with_rag(question: str, retriever: Retriever | None = None) -> dict[str, Any]:
    """Return a transparent, local grounded response without hiding model calls."""
    retriever = retriever or Retriever()
    sources = retriever.search(question)
    context = " ".join(document.content for document in sources)
    response = context if context else "I do not know based on the available sources."
    return {"answer": response, "sources": [document.source for document in sources], "prompt": build_prompt(question, context=context)}


SQL_SCHEMA = "customers(id INTEGER, name TEXT, plan TEXT, monthly_spend REAL)"


def run_nl2sql(question: str) -> dict[str, Any]:
    """Translate a narrow set of questions to read-only SQL with validation."""
    normalized = question.lower()
    if "customer" not in normalized and "customers" not in normalized:
        raise ValueError("This demo only supports questions about customers.")
    if "average" in normalized or "avg" in normalized:
        sql = "SELECT plan, ROUND(AVG(monthly_spend), 2) AS average_spend FROM customers GROUP BY plan ORDER BY average_spend DESC"
    elif "how many" in normalized or "count" in normalized:
        sql = "SELECT plan, COUNT(*) AS customer_count FROM customers GROUP BY plan ORDER BY customer_count DESC"
    else:
        sql = "SELECT id, name, plan, monthly_spend FROM customers ORDER BY monthly_spend DESC"
    if not re.fullmatch(r"SELECT [\w ,().=*]+ FROM customers(?: [\w ,().=*]+)*", sql):
        raise ValueError("Generated SQL did not pass the read-only validator.")
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE customers (id INTEGER, name TEXT, plan TEXT, monthly_spend REAL)")
        connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", [(1, "Asha", "pro", 49.0), (2, "Luis", "starter", 19.0), (3, "Mina", "pro", 59.0)])
        cursor = connection.execute(sql)
        rows = [dict(zip([column[0] for column in cursor.description], row)) for row in cursor.fetchall()]
    return {"sql": sql, "schema": SQL_SCHEMA, "rows": rows}
