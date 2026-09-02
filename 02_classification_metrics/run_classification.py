"""Binary classification with metrics, ROC curve, and threshold trade-offs."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent / "outputs"; OUT.mkdir(exist_ok=True)
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=42)).fit(Xtr, ytr)
prob = model.predict_proba(Xte)[:, 1]; pred = (prob >= .5).astype(int)
metrics = {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
           "recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "roc_auc": roc_auc_score(yte, prob),
           "average_precision": average_precision_score(yte, prob)}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2)); print(json.dumps(metrics, indent=2))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_predictions(yte, pred, ax=axes[0]); axes[0].set_title("Threshold = 0.50")
RocCurveDisplay.from_predictions(yte, prob, ax=axes[1]); axes[1].set_title("ROC curve")
fig.tight_layout(); fig.savefig(OUT / "classification_evaluation.png", dpi=160); plt.close(fig)
precision, recall, thresholds = precision_recall_curve(yte, prob)
fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(thresholds, precision[:-1], label="Precision")
ax.plot(thresholds, recall[:-1], label="Recall"); ax.set(xlabel="Decision threshold", ylabel="Score", title="Precision–recall threshold trade-off")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "threshold_tradeoff.png", dpi=160)
