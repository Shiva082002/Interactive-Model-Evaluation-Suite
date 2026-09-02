# Applied Machine Learning and AI Engineering Portfolio

A compact, reproducible machine-learning portfolio that demonstrates core modelling, evaluation, and statistical reasoning skills. The centrepiece is an end-to-end **Wine Cultivar Prediction Dashboard**: train a model, save it, enter new measurements, and receive a predicted class with confidence. The remaining modules support and document the ML concepts used in the project.

## Projects

| Folder | Focus |
|---|---|
| `01_regression_diagnostics` | Linear regression, assumptions, loss functions, MSE/RMSE/MAPE, bias-variance trade-off |
| `02_classification_metrics` | Logistic regression, precision, recall, accuracy, F1, ROC-AUC, threshold selection |
| `03_ensemble_methods` | End-to-end Random Forest prediction dashboard (training, saved model, live inference) |
| `04_svm_and_tree_pruning` | Support Vector Machines and decision-tree cost-complexity pruning |
| `05_statistical_inference` | Hypothesis testing, t-test, F-test, p-values, and the Central Limit Theorem |
| `06_ai_engineering_lab` | RAG, vector embeddings, NL2SQL, prompting, transformer concepts, LangChain, LangGraph, async, and FastAPI |

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python 01_regression_diagnostics\run_regression.py
```

## Interactive portfolio UI

To see and test every folder from one browser interface, follow [RUN_GUIDE.md](RUN_GUIDE.md), then run:

```powershell
streamlit run portfolio_app.py
```

The sidebar lets you choose a project. Analysis projects show refreshed metrics and charts; the ensemble and AI projects provide input forms with example values and questions.

## Run the showcase project

```powershell
python 03_ensemble_methods\train_model.py
streamlit run 03_ensemble_methods\app.py
```

The first command trains and evaluates the model, then saves it in `03_ensemble_methods/artifacts/`. The second command opens a browser interface where you enter chemical measurements and receive a prediction.

Run the supporting modules' `run_*.py` files in the same way. Generated charts and JSON summaries are saved under that module's `outputs/` directory.

## AI engineering lab

The sixth project is intentionally runnable without a paid model or API key. It uses a small local knowledge base and SQLite database so the engineering patterns are visible and testable:

```powershell
python 06_ai_engineering_lab\cli.py --demo
uvicorn 06_ai_engineering_lab.api:app --reload
```

It demonstrates grounded RAG, vector embeddings, safe NL2SQL, reusable prompting strategies, transformer self-attention, LangChain-compatible composition, LangGraph routing, asynchronous Python, and FastAPI endpoints. See `06_ai_engineering_lab/README.md` for the learning progression and extension points.

To execute the supporting learning modules at once:

```powershell
python run_all.py
```

## Repository design

- `01_regression_diagnostics` combines linear regression, assumptions, regression losses, and bias-variance because these belong to one modelling workflow.
- `02_classification_metrics` keeps logistic regression and its evaluation metrics together because metric choice depends on classification thresholds and business cost.
- `03_ensemble_methods` is a deployed-style prediction workflow, using Random Forest as the selected production model rather than presenting a comparison table.
- `04_svm_and_tree_pruning` and `05_statistical_inference` are independent techniques, so they remain separate, self-contained modules.
