"""Async FastAPI interface for the AI engineering lab."""
from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ollama_workflow import answer
from .transformer_demo import demo_summary

app = FastAPI(title="AI Engineering Lab API", version="1.0.0", description="Local RAG, NL2SQL, and transformer learning endpoints")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["auto", "rag", "sql"] = "auto"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
async def query(request: QueryRequest) -> dict:
    try:
        return await asyncio.to_thread(answer, request.question, request.mode)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/transformer-demo")
async def transformer_demo() -> dict:
    return await asyncio.to_thread(demo_summary)
