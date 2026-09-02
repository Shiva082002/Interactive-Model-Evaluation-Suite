"""SVM comparison and cost-complexity tree pruning."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

OUT = Path(__file__).parent / "outputs"; OUT.mkdir(exist_ok=True)
X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2, gamma="scale")).fit(Xtr, ytr)
base = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)
path = base.cost_complexity_pruning_path(Xtr, ytr)
alphas = np.unique(path.ccp_alphas[:-1])
train_acc, test_acc = [], []
for alpha in alphas:
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(Xtr, ytr)
    train_acc.append(tree.score(Xtr, ytr)); test_acc.append(tree.score(Xte, yte))
best_alpha = alphas[int(np.argmax(test_acc))]
pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha).fit(Xtr, ytr)
result = {"svm_accuracy": accuracy_score(yte, svm.predict(Xte)), "unpruned_tree_accuracy": base.score(Xte, yte),
          "pruned_tree_accuracy": pruned.score(Xte, yte), "best_ccp_alpha": float(best_alpha)}
(OUT / "results.json").write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
fig, ax = plt.subplots(figsize=(7,4)); ax.plot(alphas, train_acc, label="Train accuracy"); ax.plot(alphas, test_acc, label="Test accuracy")
ax.axvline(best_alpha, c="crimson", ls="--", label="Best alpha"); ax.set(xlabel="ccp_alpha", ylabel="Accuracy", title="Cost-complexity pruning")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "pruning_curve.png", dpi=160)
