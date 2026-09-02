"""Regression modelling, diagnostics, and loss-function comparison."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split, validation_curve

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
residuals = y_test - pred
metrics = {
    "mse": round(mean_squared_error(y_test, pred), 3),
    "rmse": round(mean_squared_error(y_test, pred) ** 0.5, 3),
    "mape_percent": round(mean_absolute_percentage_error(y_test, pred) * 100, 3),
    "r2": round(model.score(X_test, y_test), 3),
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(pred, residuals, alpha=.7)
axes[0].axhline(0, color="crimson", ls="--")
axes[0].set(title="Residuals vs fitted", xlabel="Predicted value", ylabel="Residual")
axes[1].hist(residuals, bins=18, edgecolor="white")
axes[1].set(title="Residual distribution", xlabel="Residual")
axes[2].scatter(y_test, pred, alpha=.7)
low, high = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
axes[2].plot([low, high], [low, high], "r--")
axes[2].set(title="Actual vs predicted", xlabel="Actual", ylabel="Predicted")
fig.tight_layout(); fig.savefig(OUT / "diagnostics.png", dpi=160); plt.close(fig)

alphas = np.logspace(-4, 5, 25)
train_scores, valid_scores = validation_curve(Ridge(), X, y, param_name="alpha", param_range=alphas,
    cv=5, scoring="neg_mean_squared_error")
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(alphas, -train_scores.mean(1), label="Train MSE")
ax.semilogx(alphas, -valid_scores.mean(1), label="Validation MSE")
ax.set(title="Bias-variance trade-off with Ridge", xlabel="Regularization strength (alpha)", ylabel="MSE")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "bias_variance.png", dpi=160)
print(f"Saved plots and metrics to {OUT}")
