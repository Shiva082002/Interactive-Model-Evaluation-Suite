"""Train, evaluate, and persist multiple tree-based Wine Cultivar models."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "wine_random_forest.joblib"
REPORT_PATH = ARTIFACT_DIR / "evaluation_report.json"
RANDOM_STATE = 42


def train() -> dict:
    """Train reproducible tree models and return their evaluation reports."""
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.25, stratify=wine.target, random_state=RANDOM_STATE
    )
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), n_estimators=150,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=RANDOM_STATE),
    }
    for model in models.values():
        model.fit(X_train, y_train)

    # Majority voting is the final recommendation; individual predictions remain visible in the UI.
    voter = VotingClassifier(estimators=[(f"model_{i}", model) for i, model in enumerate(models.values())], voting="soft")
    voter.fit(X_train, y_train)
    models["Majority Vote Ensemble"] = voter

    evaluations = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        evaluations[name] = {
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "weighted_f1": round(f1_score(y_test, predictions, average="weighted"), 4),
        }

    report = {
        "project": "Wine Cultivar Predictor",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "model_evaluations": evaluations,
        "majority_vote_classification_report": classification_report(
            y_test, models["Majority Vote Ensemble"].predict(X_test),
            target_names=list(wine.target_names), output_dict=True,
        ),
    }
    artifact = {
        "models": models,
        "feature_names": list(wine.feature_names),
        "class_names": list(wine.target_names),
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    evaluation = train()
    print("Training complete.")
    for name, scores in evaluation["model_evaluations"].items():
        print(f"{name}: accuracy={scores['accuracy']:.2%}, weighted F1={scores['weighted_f1']:.4f}")
    print(f"Saved model: {MODEL_PATH}")
