"""Ollama-powered router, RAG answerer, and guarded NL2SQL workflow."""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

try:
    from .database import DATABASE_PATH, initialize_database, schema_description
    from .ollama_client import OllamaError, chat, status as ollama_status
    from .rag_pipeline import search
    from .settings import OLLAMA_CHAT_MODEL
except ImportError:
    from database import DATABASE_PATH, initialize_database, schema_description
    from ollama_client import OllamaError, chat, status as ollama_status
    from rag_pipeline import search
    from settings import OLLAMA_CHAT_MODEL

ROUTER_SYSTEM = """You route user questions for an analytics assistant. Return JSON only with route as one of rag, sql, both, or unsupported. Use rag for documentation/knowledge questions, sql for customer data questions, both when both sources are needed, and unsupported for unrelated requests."""


def route_question(question: str) -> str:
    try:
        raw = chat([{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": question}], OLLAMA_CHAT_MODEL, json_mode=True)
        route = json.loads(raw).get("route", "unsupported").lower()
        if route in {"rag", "sql", "both", "unsupported"}:
            return route
    except (ValueError, json.JSONDecodeError, KeyError):
        pass
    normalized = question.lower()
    if any(word in normalized for word in ("customer", "customers", "revenue", "spend", "churn", "plan", "country")):
        return "sql"
    if any(word in normalized for word in ("what is", "how does", "explain", "rag", "embedding", "transformer", "prompt")):
        return "rag"
    return "unsupported"


def route_question_with_trace(question: str, trace: list[dict[str, Any]]) -> str:
    trace.append({"step": "Route question", "status": "running", "detail": "Asking Ollama to classify the request as RAG, SQL, both, or unsupported."})
    try:
        raw = chat([{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": question}], OLLAMA_CHAT_MODEL, json_mode=True)
        route = json.loads(raw).get("route", "unsupported").lower()
        if route in {"rag", "sql", "both", "unsupported"}:
            trace.append({"step": "Route question", "status": "complete", "detail": f"Ollama selected: {route}", "raw": raw})
            return route
    except (OllamaError, ValueError, json.JSONDecodeError, KeyError) as error:
        trace.append({"step": "Route question", "status": "fallback", "detail": f"Ollama routing unavailable: {error}. Using keyword fallback."})
    route = route_question(question)
    trace.append({"step": "Route question", "status": "complete", "detail": f"Fallback selected: {route}"})
    return route


def validate_sql(sql: str) -> str:
    cleaned = re.sub(r"```(?:sqlite|sql)?", "", sql.strip(), flags=re.IGNORECASE).replace("```", "").strip().rstrip(";")
    if not re.match(r"^SELECT\b", cleaned, re.IGNORECASE) or ";" in cleaned:
        raise ValueError("Only one read-only SELECT statement is allowed.")
    if not re.search(r"\bcustomers\b", cleaned, re.IGNORECASE):
        raise ValueError("SQL may only query the customers table in this demo.")
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|PRAGMA|CREATE|REPLACE)\b", cleaned, re.IGNORECASE):
        raise ValueError("Write operations and database commands are blocked.")
    return cleaned


def generate_sql(question: str) -> str:
    prompt = f"""Convert the question to exactly one SQLite SELECT statement. Use only table customers and its columns. Never use INSERT, UPDATE, DELETE, DROP, or multiple statements. Return SQL only.\nSchema: {schema_description()}\nQuestion: {question}"""
    return validate_sql(chat([{ "role": "user", "content": prompt}], OLLAMA_CHAT_MODEL))


def generate_sql_with_trace(question: str, trace: list[dict[str, Any]]) -> str:
    trace.append({"step": "Generate SQL", "status": "running", "detail": "Sending the database schema and question to Ollama."})
    sql = generate_sql(question)
    trace.append({"step": "Generate SQL", "status": "complete", "detail": "Ollama generated SQL and the read-only validator accepted it.", "sql": sql})
    return sql


def execute_sql(sql: str, limit: int = 100) -> dict[str, Any]:
    safe_sql = validate_sql(sql)
    limited_sql = f"SELECT * FROM ({safe_sql}) LIMIT {limit}"
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(limited_sql)
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()
    return {"sql": safe_sql, "rows": rows, "row_count": len(rows), "limit": limit}


def answer(question: str, forced_route: str = "auto") -> dict[str, Any]:
    trace: list[dict[str, Any]] = [{"step": "Receive input", "status": "complete", "detail": question}]
    result: dict[str, Any] = {"question": question, "model": OLLAMA_CHAT_MODEL, "trace": trace}
    health = ollama_status()
    trace.append({"step": "Check Ollama", "status": "complete" if health.get("connected") else "failed", "detail": "Connected to Ollama." if health.get("connected") else str(health.get("error")), "models": health.get("models", [])})
    try:
        trace.append({"step": "Prepare database", "status": "running", "detail": "Opening the persistent synthetic customer database."})
        database_info = initialize_database(rows=20000)
        trace.append({"step": "Prepare database", "status": "complete", "detail": f"Ready: {database_info['rows']:,} customer rows."})
        route = forced_route if forced_route != "auto" else route_question_with_trace(question, trace)
        result["route"] = route
        if route in {"rag", "both"}:
            trace.append({"step": "Retrieve documents", "status": "running", "detail": "Embedding the question and searching the persistent vector index."})
            sources = search(question)
            trace.append({"step": "Retrieve documents", "status": "complete", "detail": f"Retrieved {len(sources)} chunks.", "sources": sources})
            context = "\n\n".join(f"[{item['source']}#{item['chunk']}] {item['text']}" for item in sources)
            prompt = f"Answer only from the context. If the context is insufficient, say so.\nContext:\n{context}\nQuestion: {question}"
            trace.append({"step": "Build RAG prompt", "status": "complete", "detail": f"Prepared grounded prompt with {len(context)} characters of context."})
            answer_text = chat([{ "role": "user", "content": prompt}], OLLAMA_CHAT_MODEL)
            result["rag"] = {"answer": answer_text, "sources": sources, "prompt": prompt}
            trace.append({"step": "Generate RAG answer", "status": "complete", "detail": "Ollama answered using retrieved context."})
        if route in {"sql", "both"}:
            sql = generate_sql_with_trace(question, trace)
            trace.append({"step": "Execute SQL", "status": "running", "detail": "Executing the validated SELECT with a 100-row safety limit."})
            result["sql"] = execute_sql(sql)
            trace.append({"step": "Execute SQL", "status": "complete", "detail": f"Returned {result['sql']['row_count']} rows."})
            summary_prompt = f"Explain these SQL results in two concise sentences for the user. Question: {question}\nResults: {json.dumps(result['sql']['rows'])}"
            result["sql"]["answer"] = chat([{ "role": "user", "content": summary_prompt}], OLLAMA_CHAT_MODEL)
            trace.append({"step": "Summarize SQL result", "status": "complete", "detail": "Ollama converted the tabular result into a plain-language answer."})
        if route == "both":
            result["answer"] = f"Documentation answer:\n{result['rag']['answer']}\n\nDatabase answer:\n{result['sql']['answer']}"
        elif route == "rag":
            result["answer"] = result["rag"]["answer"]
        elif route == "sql":
            result["answer"] = result["sql"]["answer"]
        else:
            result["answer"] = "I can answer questions about the indexed documents or the customers database."
            trace.append({"step": "Finish", "status": "complete", "detail": "Request was outside the supported domains."})
    except (OllamaError, ValueError, sqlite3.Error) as error:
        trace.append({"step": "Finish", "status": "failed", "detail": str(error)})
        result["error"] = str(error)
    return result
