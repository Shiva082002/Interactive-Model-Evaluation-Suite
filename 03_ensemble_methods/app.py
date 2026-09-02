"""Streamlit user interface for predicting wine cultivar from lab measurements."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "wine_random_forest.joblib"

st.set_page_config(page_title="Wine Cultivar Predictor", page_icon="🍷", layout="wide")
st.title("Wine Cultivar Predictor")
st.caption("Enter laboratory measurements to classify a wine cultivar using a trained Random Forest model.")

if not MODEL_PATH.exists():
    st.error("No trained model was found. First run: python 03_ensemble_methods\\train_model.py")
    st.stop()

try:
    artifact = joblib.load(MODEL_PATH)
except (ModuleNotFoundError, ImportError, AttributeError, ValueError):
    st.warning("The saved model is incompatible with this environment. Rebuilding it now...")
    import subprocess
    import sys

    completed = subprocess.run([sys.executable, str(PROJECT_DIR / "train_model.py")], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        st.error("The model could not be rebuilt.")
        st.code(completed.stderr or completed.stdout)
        st.stop()
    st.success("The model was rebuilt. Reloading the dashboard...")
    st.rerun()
models = artifact["models"]
feature_names = artifact["feature_names"]
class_names = artifact["class_names"]

# Typical values from the source dataset make the initial form meaningful.
defaults = [13.0, 2.3, 2.4, 19.5, 100.0, 2.3, 2.1, 0.36, 1.6, 5.0, 1.05, 2.6, 1000.0]
labels = [name.replace("_", " ").title() for name in feature_names]

st.sidebar.header("Input measurements")
values = {}
for index, (feature, label, default) in enumerate(zip(feature_names, labels, defaults)):
    values[feature] = st.sidebar.number_input(label, value=float(default), format="%.4f", key=f"input_{index}")

if st.sidebar.button("Predict cultivar", type="primary", use_container_width=True):
    input_frame = pd.DataFrame([values], columns=feature_names)
    results = []
    for name, model in models.items():
        probabilities = model.predict_proba(input_frame)[0]
        predicted_index = int(probabilities.argmax())
        results.append({
            "Model": name,
            "Predicted cultivar": class_names[predicted_index].title(),
            "Confidence": probabilities[predicted_index],
        })
    prediction_table = pd.DataFrame(results)
    final_row = prediction_table.loc[prediction_table["Model"] == "Majority Vote Ensemble"].iloc[0]

    left, right = st.columns(2)
    left.metric("Final recommendation", final_row["Predicted cultivar"])
    left.metric("Ensemble confidence", f"{final_row['Confidence']:.1%}")
    right.subheader("All model confidences")
    right.bar_chart(prediction_table.set_index("Model")["Confidence"])

    st.subheader("Predictions from every tree-based model")
    st.dataframe(
        prediction_table.assign(Confidence=prediction_table["Confidence"].map(lambda value: f"{value:.1%}")),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Submitted measurements")
    st.dataframe(input_frame, use_container_width=True, hide_index=True)
else:
    st.info("Adjust the values in the sidebar, then select **Predict cultivar**. Every tree-based model will return a prediction.")
