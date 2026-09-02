"""Run every portfolio module from the repository root."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parent
SCRIPTS = [
    "01_regression_diagnostics/run_regression.py",
    "02_classification_metrics/run_classification.py",
    "04_svm_and_tree_pruning/run_svm_pruning.py",
    "05_statistical_inference/run_inference.py",
]

for relative_path in SCRIPTS:
    print(f"\n{'=' * 72}\nRunning: {relative_path}\n{'=' * 72}")
    subprocess.run([sys.executable, str(ROOT / relative_path)], check=True)

print("\nSupporting modules completed. Review each module's outputs/ directory for results.")
print("AI engineering lab: run `python 06_ai_engineering_lab/cli.py --demo` or start its API with `uvicorn 06_ai_engineering_lab.api:app --reload`.")
