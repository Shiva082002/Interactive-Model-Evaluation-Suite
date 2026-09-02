# Ollama AI Engineering Lab

A local-first AI application powered by Ollama. It routes questions between RAG, NL2SQL, both, and unsupported requests, while exposing every step in the Streamlit UI. The default models are `gemma3:12b` for chat and `nomic-embed-text:latest` for embeddings.

## What this demonstrates

| Area | Implementation | Why it matters |
|---|---|---|
| RAG | Persistent chunk index plus grounded Ollama prompt | Connects generation to inspectable source evidence |
| Vector embeddings | Ollama `nomic-embed-text:latest` stored in NumPy | Makes similarity search measurable and reproducible |
| NL2SQL | Ollama SQL generation against SQLite with validation | Shows schema grounding and read-only safety |
| Transformer architecture | Token positions, scaled dot-product attention, residual MLP block | Makes the core tensor operations concrete |
| Prompting | Zero-shot, few-shot, and grounded templates | Turns prompt design into a testable interface |
| LangChain | Prompt/retrieval components are intentionally small and composable | Encourages provider-independent pipelines |
| LangGraph | Typed state graph routes RAG versus NL2SQL | Makes multi-step orchestration explicit |
| Async/FastAPI | Async endpoints delegate CPU work with `asyncio.to_thread` | Keeps the HTTP event loop responsive |

## Run it with Ollama

From the repository root:

```powershell
ollama serve
ollama pull gemma3:12b
ollama pull nomic-embed-text:latest
python 06_ai_engineering_lab\ingest.py
streamlit run portfolio_app.py
```

Keep Ollama running in one terminal and Streamlit in another. Open `http://localhost:8501`. Select **06 AI engineering lab**, enter a question, and expand each execution step. The first ingestion embeds the sample documents and creates a 20,000-row SQLite database.

To use the API instead:

```powershell
uvicorn 06_ai_engineering_lab.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API contract.

## Add your documents

Put `.md` or `.txt` files in `06_ai_engineering_lab/docs/`, then rebuild the index:

```powershell
python 06_ai_engineering_lab\ingest.py
```

The pipeline chunks documents with overlap, embeds them in batches, and stores vectors in `06_ai_engineering_lab/index/`.

## Useful questions

- `Explain RAG using the indexed documents.`
- `How many customers are on each plan?`
- `What is the average monthly spend by country?`
- `Explain vector embeddings and compare that with the number of pro customers.`

## Test it

```powershell
pytest 06_ai_engineering_lab\test_lab.py
```

## Suggested learning progression

1. Read `ollama_workflow.py` and follow the trace from route to final answer.
2. Add documents and compare retrieval scores as the corpus grows.
3. Extend the database schema and prompt only with matching validation and tests.
4. Add answer grading, retry, caching, conversation memory, and human approval as new graph nodes.
5. Add evaluation datasets for retrieval recall, SQL correctness, groundedness, latency, and cost.
