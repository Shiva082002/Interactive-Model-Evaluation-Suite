"""Interactive launcher for every project in the learning portfolio."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent

PROJECTS = {
    "01 Regression diagnostics": {
        "folder": "01_regression_diagnostics",
        "description": "Regression losses, residual diagnostics, and the Ridge bias-variance trade-off.",
        "script": "run_regression.py",
        "metrics": "metrics.json",
        "images": ["diagnostics.png", "bias_variance.png"],
        "examples": ["Run the analysis to inspect MSE, RMSE, MAPE, R2, and diagnostic charts."],
    },
    "02 Classification metrics": {
        "folder": "02_classification_metrics",
        "description": "Logistic classification with threshold, precision, recall, F1, ROC-AUC, and PR analysis.",
        "script": "run_classification.py",
        "metrics": "metrics.json",
        "images": ["classification_evaluation.png", "threshold_tradeoff.png"],
        "examples": ["Run the analysis to compare classification metrics and threshold behavior."],
    },
    "03 Ensemble dashboard": {
        "folder": "03_ensemble_methods",
        "description": "Live wine cultivar predictions from seven tree-based models.",
        "examples": ["Default wine measurements", "Change alcohol, flavanoids, or proline and compare model confidence."],
    },
    "04 SVM and tree pruning": {
        "folder": "04_svm_and_tree_pruning",
        "description": "RBF SVM comparison and cost-complexity pruning for decision trees.",
        "script": "run_svm_pruning.py",
        "metrics": "results.json",
        "images": ["pruning_curve.png"],
        "examples": ["Run the analysis to compare SVM, unpruned tree, and pruned tree accuracy."],
    },
    "05 Statistical inference": {
        "folder": "05_statistical_inference",
        "description": "A/B testing, p-values, variance testing, and a Central Limit Theorem simulation.",
        "script": "run_inference.py",
        "metrics": "test_results.json",
        "images": ["central_limit_theorem.png"],
        "examples": ["Run the simulation to inspect the t-test, F-test, and CLT results."],
    },
    "06 AI engineering lab": {
        "folder": "06_ai_engineering_lab",
        "description": "Local-first RAG, embeddings, NL2SQL, prompting, transformers, LangGraph, async, and FastAPI.",
        "examples": [
            "How do vector embeddings help RAG?",
            "What does LangGraph do?",
            "How many customers are on each plan?",
            "What is the average customer spend by plan?",
        ],
    },
}


def run_script(project: dict) -> tuple[bool, str]:
    command = [sys.executable, str(ROOT / project["folder"] / project["script"])]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    output = completed.stdout or completed.stderr
    return completed.returncode == 0, output


def show_analysis(project: dict) -> None:
    if st.button("Run analysis", type="primary"):
        with st.spinner("Running the project script..."):
            success, output = run_script(project)
        if success:
            st.success("Analysis completed and artifacts were refreshed.")
        else:
            st.error("The project script returned an error.")
        with st.expander("Console output", expanded=not success):
            st.code(output or "No console output.")

    metrics_path = ROOT / project["folder"] / "outputs" / project["metrics"]
    if metrics_path.exists():
        st.subheader("Latest results")
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
    st.subheader("Visual evidence")
    columns = st.columns(len(project["images"]))
    for column, image_name in zip(columns, project["images"]):
        image_path = ROOT / project["folder"] / "outputs" / image_name
        if image_path.exists():
            column.image(str(image_path), caption=image_name, width="stretch")
        else:
            column.info(f"Run the analysis to create {image_name}.")


def show_ensemble() -> None:
    model_path = ROOT / "03_ensemble_methods" / "artifacts" / "wine_random_forest.joblib"
    if not model_path.exists():
        st.warning("The trained artifact is missing.")
        if st.button("Train ensemble models", type="primary"):
            success, output = run_script({"folder": "03_ensemble_methods", "script": "train_model.py"})
            st.code(output)
            if success:
                st.rerun()
        return

    try:
        artifact = joblib.load(model_path)
    except (ModuleNotFoundError, ImportError, AttributeError, ValueError) as error:
        st.warning("The saved model was created with an incompatible Python or scikit-learn version. Rebuilding it now...")
        success, output = run_script({"folder": "03_ensemble_methods", "script": "train_model.py"})
        if not success:
            st.error("The model could not be rebuilt.")
            st.code(output or str(error))
            return
        st.success("The model was rebuilt for the active environment.")
        st.rerun()
    feature_names = artifact["feature_names"]
    labels = [name.replace("_", " ").title() for name in feature_names]
    defaults = [13.0, 2.3, 2.4, 19.5, 100.0, 2.3, 2.1, 0.36, 1.6, 5.0, 1.05, 2.6, 1000.0]
    st.subheader("Enter wine measurements")
    st.caption("Use the default example first, then change one measurement to see how confidence changes.")
    values = {}
    input_columns = st.columns(2)
    for index, (feature, label, default) in enumerate(zip(feature_names, labels, defaults)):
        values[feature] = input_columns[index % 2].number_input(label, value=float(default), format="%.4f", key=f"portfolio_{feature}")
    if st.button("Predict cultivar", type="primary"):
        frame = pd.DataFrame([values], columns=feature_names)
        rows = []
        for name, model in artifact["models"].items():
            probabilities = model.predict_proba(frame)[0]
            index = int(probabilities.argmax())
            rows.append({"Model": name, "Cultivar": artifact["class_names"][index].title(), "Confidence": float(probabilities[index])})
        result = pd.DataFrame(rows)
        final = result[result["Model"] == "Majority Vote Ensemble"].iloc[0]
        left, right = st.columns(2)
        left.metric("Recommended cultivar", final["Cultivar"])
        left.metric("Ensemble confidence", f"{final['Confidence']:.1%}")
        right.bar_chart(result.set_index("Model")["Confidence"])
        st.dataframe(result.assign(Confidence=result["Confidence"].map(lambda value: f"{value:.1%}")), hide_index=True, width="stretch")


def show_ai_lab(project: dict) -> None:
    st.subheader("Ask the local AI workflow")
    st.caption("Ollama routes each request to RAG, NL2SQL, both, or an unsupported response. Every processing step is shown below.")
    workflow = importlib.import_module("06_ai_engineering_lab.ollama_workflow")
    client = importlib.import_module("06_ai_engineering_lab.ollama_client")
    rag = importlib.import_module("06_ai_engineering_lab.rag_pipeline")
    health = client.status()
    if health.get("connected"):
        st.success(f"Ollama connected: {', '.join(health.get('models', []))}")
    else:
        st.error(f"Ollama is unavailable: {health.get('error')}")
    with st.expander("Knowledge base setup", expanded=not rag.index_exists()):
        st.write("Add `.md` or `.txt` files to `06_ai_engineering_lab/docs/`, then rebuild the database and vector index.")
        if st.button("Rebuild documents and database"):
            ingest = importlib.import_module("06_ai_engineering_lab.ingest")
            with st.spinner("Embedding documents and generating customer rows..."):
                info = ingest.main()
            st.success("Knowledge base rebuilt.")
            st.write(info)
    examples = project["examples"]
    selected_example = st.selectbox("Example questions", examples)
    question = st.text_area("Your question", value=selected_example, height=90)
    mode = st.radio("Workflow", ["auto", "rag", "sql", "both"], horizontal=True, index=0)
    if st.button("Get answer", type="primary"):
        with st.spinner("Running retrieval or SQL workflow..."):
            result = workflow.answer(question, forced_route=mode)
        st.subheader("Execution trace")
        for number, step in enumerate(result.get("trace", []), start=1):
            status = step.get("status", "info")
            icon = {"complete": "✅", "running": "⏳", "failed": "❌", "fallback": "↪️"}.get(status, "•")
            with st.expander(f"{icon} Step {number}: {step['step']}", expanded=status in {"failed", "fallback"}):
                st.write(step.get("detail", ""))
                if "sources" in step:
                    st.dataframe(pd.DataFrame(step["sources"]), hide_index=True, width="stretch")
                if "sql" in step:
                    st.code(step["sql"], language="sql")
                if "raw" in step:
                    st.code(step["raw"], language="json")
        if result.get("error"):
            st.error(result["error"])
        else:
            st.success(f"Workflow completed using route: {result.get('route', mode)}")
            st.subheader("Final output")
            st.write(result.get("answer", "No answer returned."))
            if result.get("rag", {}).get("sources"):
                st.caption("Retrieved sources")
                st.dataframe(pd.DataFrame(result["rag"]["sources"]), hide_index=True, width="stretch")
            if result.get("sql"):
                st.caption("Generated SQL and result rows")
                st.code(result["sql"]["sql"], language="sql")
                st.dataframe(pd.DataFrame(result["sql"]["rows"]), hide_index=True, width="stretch")
    st.info("Try: 'How do vector embeddings help RAG?', 'How many customers are on each plan?', or 'Explain RAG and compare it with the number of pro customers.'")


def main() -> None:
    st.set_page_config(page_title="ML + AI Portfolio Lab", page_icon="🧪", layout="wide")
    st.title("ML + AI Portfolio Lab")
    st.caption("Run, inspect, and interact with every learning project from one place.")
    project_name = st.sidebar.selectbox("Choose a project", list(PROJECTS))
    project = PROJECTS[project_name]
    st.sidebar.markdown("### What you can practice")
    st.sidebar.write(project["description"])
    st.sidebar.markdown("### Project folder")
    st.sidebar.code(project["folder"])
    st.header(project_name)
    st.write(project["description"])
    if project_name == "03 Ensemble dashboard":
        show_ensemble()
    elif project_name == "06 AI engineering lab":
        show_ai_lab(project)
    else:
        show_analysis(project)
    st.divider()
    st.subheader("Examples and next steps")
    for example in project["examples"]:
        st.markdown(f"- {example}")
    st.caption("Source code and project-specific explanations are in the selected folder's README.md.")


if __name__ == "__main__":
    main()
