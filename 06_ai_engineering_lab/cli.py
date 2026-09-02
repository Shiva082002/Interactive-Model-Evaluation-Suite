"""Command-line entry point for the local AI engineering lab."""
from __future__ import annotations

import argparse
import json

try:
    from .core import answer_with_rag, run_nl2sql
    from .graph import QUERY_GRAPH
    from .transformer_demo import demo_summary
except ImportError:
    from core import answer_with_rag, run_nl2sql
    from graph import QUERY_GRAPH
    from transformer_demo import demo_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local AI engineering demonstrations")
    parser.add_argument("question", nargs="?", default="How do vector embeddings help RAG?")
    parser.add_argument("--demo", action="store_true", help="run RAG, NL2SQL, graph, and transformer examples")
    args = parser.parse_args()
    if args.demo:
        result = {"rag": answer_with_rag(args.question), "sql": run_nl2sql("How many customers are on each plan?"), "graph": QUERY_GRAPH.invoke({"question": args.question}), "transformer": demo_summary()}
    else:
        result = QUERY_GRAPH.invoke({"question": args.question})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
