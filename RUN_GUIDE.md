# Portfolio Run Guide

This guide starts the complete interactive interface for the repository. You can select every folder from the sidebar, run analyses, inspect charts and metrics, and submit live examples for the interactive projects.

## 1. Open the repository

Open a PowerShell terminal in the folder that contains `README.md`:

```powershell
cd D:\Coding\learning
```

## 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Install the dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The AI lab works locally without an API key. `sentence-transformers`, LangChain, and LangGraph are included for the full learning experience; the local fallback keeps the basic demo usable if those optional packages are unavailable.

## 4. Start the portfolio UI

```powershell
streamlit run portfolio_app.py
```

Streamlit opens a browser automatically. If it does not, open the URL shown in the terminal, usually `http://localhost:8501`.

## 5. Use each project

1. Select **01 Regression diagnostics**, **02 Classification metrics**, **04 SVM and tree pruning**, or **05 Statistical inference**. Select **Run analysis** and inspect the latest JSON results and generated charts.
2. Select **03 Ensemble dashboard**. Use the pre-filled wine measurements, select **Predict cultivar**, then change a measurement and run it again to compare model confidence.
3. Select **06 AI engineering lab**. Choose an example question or write your own, select `auto`, `rag`, or `sql`, and select **Get answer**.

## 6. Start the FastAPI version of the AI lab

Stop Streamlit with `Ctrl+C` only if you want to reuse the terminal, then run:

```powershell
uvicorn 06_ai_engineering_lab.api:app --reload
```

Open the interactive API contract at `http://127.0.0.1:8000/docs`.

Useful requests:

- `GET /health`
- `GET /transformer-demo`
- `POST /query` with `{"question":"How do vector embeddings help RAG?","mode":"rag"}`
- `POST /query` with `{"question":"How many customers are on each plan?","mode":"sql"}`

## 7. Run tests and scripts

```powershell
pytest 06_ai_engineering_lab\test_lab.py
python run_all.py
python 06_ai_engineering_lab\cli.py --demo
```

The analysis scripts save JSON and image artifacts under each project's `outputs` folder. The ensemble training script saves its model under `03_ensemble_methods\artifacts`.

## Troubleshooting

- **`streamlit` or `pytest` is not recognized:** activate `.venv`, then run `pip install -r requirements.txt` again.
- **PowerShell activation error:** use the temporary execution-policy command in step 2.
- **Missing wine model:** use the **Train ensemble models** button in the UI or run `python 03_ensemble_methods\train_model.py`.
- **Port already in use:** run `streamlit run portfolio_app.py --server.port 8502` or `uvicorn 06_ai_engineering_lab.api:app --reload --port 8001`.
