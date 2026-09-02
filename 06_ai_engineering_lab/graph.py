"""LangGraph orchestration for routing questions to local capabilities."""
from __future__ import annotations

from importlib import import_module
from typing import TypedDict

try:
    _langgraph = import_module("langgraph.graph")
    END = _langgraph.END
    START = _langgraph.START
    StateGraph = _langgraph.StateGraph
except ImportError:
    END = START = None
    StateGraph = None

try:
    from .core import answer_with_rag, run_nl2sql
except ImportError:
    from core import answer_with_rag, run_nl2sql


class QueryState(TypedDict, total=False):
    question: str
    route: str
    result: dict


def classify(state: QueryState) -> QueryState:
    question = state.get("question", "").lower()
    route = "sql" if "customer" in question or "spend" in question or "plan" in question else "rag"
    return {"route": route}


def retrieve(state: QueryState) -> QueryState:
    return {"result": answer_with_rag(state.get("question", ""))}


def query_database(state: QueryState) -> QueryState:
    return {"result": run_nl2sql(state.get("question", ""))}


def route(state: QueryState) -> str:
    return state.get("route", "rag")


def build_query_graph():
    if StateGraph is None:
        class LocalGraph:
            def invoke(self, state: QueryState) -> QueryState:
                classified = classify(state)
                handler = query_database if classified.get("route") == "sql" else retrieve
                return {**state, **classified, **handler({**state, **classified})}

        return LocalGraph()
    graph = StateGraph(QueryState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("query_database", query_database)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route, {"rag": "retrieve", "sql": "query_database"})
    graph.add_edge("retrieve", END)
    graph.add_edge("query_database", END)
    return graph.compile()


QUERY_GRAPH = build_query_graph()
