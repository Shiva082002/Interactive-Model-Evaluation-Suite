"""Smoke tests for the local, no-key learning path."""
from core import answer_with_rag, run_nl2sql
from transformer_demo import demo_summary


def test_rag_returns_sources():
    result = answer_with_rag("What are vector embeddings?")
    assert result["sources"]
    assert "embeddings" in result["answer"].lower()


def test_nl2sql_is_read_only_and_structured():
    result = run_nl2sql("How many customers are on each plan?")
    assert result["sql"].startswith("SELECT")
    assert len(result["rows"]) == 2


def test_transformer_preserves_shape():
    result = demo_summary()
    assert result["input_shape"] == result["output_shape"]
