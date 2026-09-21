# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview
AICTE-2026 Problem Statement #39 — Predictive Maintenance of Industrial Machinery.
Python ML classification project: predicts normal operation vs. failure (and failure type) for industrial machinery.

## Folder Layout (enforced — do not reorganize)
```
data/                         raw dataset CSVs only (never modified by code)
model/                        saved .pkl files output by train_model.py
backend/                      Flask REST API (app.py)
frontend/                     Streamlit app (app.py)
docs/                         project report DOCX and related materials
Predictive_Maintenance.ipynb  full ML workflow notebook (project root)
train_model.py                reproducible training script (project root)
README.md                     project documentation (project root)
```

## Key Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Train and save model (script is at project root)
python train_model.py

# Run Flask API (port 5000)
python backend/app.py

# Run Streamlit UI (port 8501)
streamlit run frontend/app.py

# Launch Jupyter notebook (notebook is at project root)
jupyter notebook Predictive_Maintenance.ipynb
```

## Dataset
- Use the **Machine Predictive Maintenance Classification** dataset from Kaggle (Shivam Bansal).
- URL: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
- CSV placed at `data/predictive_maintenance.csv` — never modify the raw file.
- **The CSV is TAB-separated** — always load with `pd.read_csv(..., sep='\t')`, not the default comma.
- Drop `UDI` and `Product ID` at load time (no predictive value).
- Target columns: `Target` (binary 0/1) and `Failure Type` (string).
- Failure Type values (actual, verified): `'No Failure'`, `'Heat Dissipation Failure'`, `'Power Failure'`, `'Overstrain Failure'`, `'Tool Wear Failure'`, `'Random Failures'`.
- Features: `Type` (L/M/H, OrdinalEncoder), plus 5 numeric sensors (StandardScaler).
- Verified shape: 10 000 rows × 10 columns; 0 nulls; 0 duplicates.
- Class imbalance: Target=0 → 9661 (96.61%), Target=1 → 339 (3.39%).

## Critical Non-Obvious Rules

### Model Persistence
- The `.pkl` file saved by `train_model.py` must bundle **both** the preprocessing pipeline and the fitted model as a single `Pipeline` object — the Flask API and Streamlit app both load this one file via `joblib.load`.
- Never save the scaler and model as separate files; the API would need to know the exact preprocessing order.

### Flask API Contract
- POST `/predict` accepts JSON `{"features": [...]}` where the feature array order must exactly match the column order used during training.
- The API returns `{"prediction": 0|1, "failure_type": "None|TWF|HDF|..."}` — Streamlit parses these exact keys.

### Model Selection Policy
- Evaluate at minimum: Logistic Regression, Random Forest, Gradient Boosting (XGBoost/sklearn).
- Select based on **F1-score (macro)** and **ROC-AUC**, not accuracy alone — the dataset is class-imbalanced (~3.4% failure rate).
- Document all metric results in the notebook; do not claim accuracy without showing the confusion matrix.

### Notebook vs. train_model.py
- The notebook (`Predictive_Maintenance.ipynb`) is at the **project root** and contains EDA visualisations.
- `train_model.py` is at the **project root** — clean, standalone, no plots.
- Both must produce identical model outputs when run on the same data.

### Streamlit UI
- Input fields map 1-to-1 to the model features; include units and valid ranges as helper text.
- Display prediction result AND a plain-English explanation (e.g., "High torque combined with low speed may indicate overload").

### No Fabricated Results
- All metrics, confusion matrices, and example predictions shown in the notebook/report must come from actual model runs.
- Do not hard-code example outputs; always load the model and run `.predict()`.
