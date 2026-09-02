# Wine Cultivar Prediction Dashboard

An end-to-end multi-class machine-learning application. It trains several tree-based classifiers on the UCI Wine Recognition dataset, saves them, and exposes live predictions through a Streamlit dashboard.

## Workflow

1. `train_model.py` loads data, splits it into training/test sets, trains every tree-based model, evaluates each one, and stores the model artifact.
2. `app.py` loads the saved models and accepts new chemical measurements from the user.
3. The dashboard returns each model's predicted cultivar and confidence, plus a soft-voting ensemble recommendation.

## Run locally

```powershell
python 03_ensemble_methods\train_model.py
streamlit run 03_ensemble_methods\app.py
```

The model and evaluation report are saved in `artifacts/`. This folder is generated locally and excluded from Git.

## Models included

- Decision Tree
- Bagging classifier
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- Soft-voting ensemble recommendation
