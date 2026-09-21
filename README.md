# Predictive Maintenance of Industrial Machinery
### AICTE-2026 Problem Statement #39

---

## Problem Statement

> **Problem Statement 39:** Design and implement a Predictive Maintenance system using machine  
> learning that can analyse industrial machinery sensor data and predict equipment failures before  
> they occur. The system should classify whether a machine is operating normally or likely to fail,  
> and where possible identify the failure type.

---

## Project Objective

Build a beginner-friendly, end-to-end machine learning classification system that:

- Analyses real industrial machinery sensor and operational data
- Predicts whether a machine is operating **normally** or experiencing a **failure**
- Classifies the **type of failure** (TWF / HDF / PWF / OSF / RNF) when a failure is detected
- Exposes predictions through a **REST API** (Flask) and an interactive **web UI** (Streamlit)

---

## Dataset

| Field | Details |
|-------|---------|
| **Name** | Machine Predictive Maintenance Classification |
| **Source** | [Kaggle — Shivam Bansal](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) |
| **Rows** | 10 000 |
| **Columns** | 10 |
| **Primary target** | `Target` (0 = No Failure, 1 = Failure) |
| **Secondary target** | `Failure Type` (No Failure / TWF / HDF / PWF / OSF / RNF) |

### Dataset Columns

| Column | Type | Role |
|--------|------|------|
| `UDI` | int | Row ID — dropped at preprocessing |
| `Product ID` | str | Serial number — dropped at preprocessing |
| `Type` | str (L/M/H) | Machine quality variant — encoded |
| `Air temperature [K]` | float | Sensor feature |
| `Process temperature [K]` | float | Sensor feature |
| `Rotational speed [rpm]` | int | Sensor feature |
| `Torque [Nm]` | float | Sensor feature |
| `Tool wear [min]` | int | Sensor feature |
| `Target` | int (0/1) | **Primary target** |
| `Failure Type` | str | **Secondary target** |

> **Download instructions:** Create a Kaggle account, navigate to the dataset page above,  
> and download `predictive_maintenance.csv`. Place it in the `data/` folder.

---

## Technologies

| Layer | Technology |
|-------|------------|
| Language | Python 3.12+ |
| ML framework | scikit-learn |
| Class imbalance | imbalanced-learn (SMOTE) |
| Data manipulation | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Model persistence | joblib |
| Notebook | Jupyter |
| REST API | Flask + flask-cors |
| Frontend | Streamlit |
| Report generation | python-docx |

---

## Project Structure

```
Predictive_Maintenance_Project/
├── data/
│   └── predictive_maintenance.csv    ← place Kaggle CSV here (not tracked by git)
├── model/
│   ├── best_model.pkl                ← saved pipeline (generated after training)
│   └── label_encoder.pkl             ← failure-type encoder (generated after training)
├── backend/
│   └── app.py                        ← Flask REST API (Step 6)
├── frontend/
│   └── app.py                        ← Streamlit UI (Step 7)
├── docs/                             ← project report materials (DOCX etc.)
├── Predictive_Maintenance.ipynb      ← full ML workflow with EDA and evaluation
├── train_model.py                    ← reproducible training script (no plots)
├── requirements.txt
├── README.md
└── AGENTS.md
```

---

## Installation & Setup

### 1. Clone or download the project

```bash
git clone <repository-url>
cd Predictive_Maintenance_Project
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

1. Go to [https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
2. Download `predictive_maintenance.csv`
3. Place it in the `data/` folder

---

## How to Train the Model

```bash
python train_model.py
```

This script:
- Loads `data/predictive_maintenance.csv`
- Cleans and preprocesses the data
- Trains and compares multiple classification models
- Selects the best model by macro F1-score
- Saves `model/best_model.pkl` and `model/label_encoder.pkl`

> **Note:** Run this after placing the dataset in `data/`.

---

## How to Open the Jupyter Notebook

```bash
jupyter notebook Predictive_Maintenance.ipynb
```

The notebook contains the complete ML workflow:
- Data loading and cleaning
- Exploratory Data Analysis (EDA) with visualisations
- Preprocessing pipeline
- Model training and comparison table
- Full evaluation (classification report, confusion matrix, ROC curve)
- Failure-type classification
- Sample predictions

Run all cells from top to bottom on a clean kernel.

---

## How to Start the Flask Backend

> **Requires:** `model/best_model.pkl` to exist (run `train_model.py` first)

```bash
python backend/app.py
```

The API will start on [http://127.0.0.1:5000](http://127.0.0.1:5000).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Submit sensor values, receive failure prediction |
| `/feature-info` | GET | Returns feature names, types, and valid ranges |

---

## How to Start the Streamlit Frontend

> **Requires:** Flask backend running on port 5000

```bash
streamlit run frontend/app.py
```

The UI will open at [http://localhost:8501](http://localhost:8501).

Enter sensor values using the sidebar controls and click **Predict** to receive:
- A colour-coded prediction banner (No Failure / Machine Failure)
- The failure type (if a failure is predicted)
- A confidence score
- A plain-English explanation of key contributing factors

---

## Model Performance

> Results will be filled in after the notebook is run on the actual dataset.  
> See `Predictive_Maintenance.ipynb` Section 6 for the full evaluation.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC |
|-------|----------|-------------------|----------------|------------|---------|
| *(populated after training)* | — | — | — | — | — |

---

## IBM Cloud Lite Deployment

IBM Cloud Lite integration will be completed as a separate, later stage of this project.  
Planned options include deploying the Flask API via IBM Cloud Code Engine, registering the  
trained model with IBM Watson Machine Learning, or using IBM Cloud Foundry.

> This README will be updated with deployment instructions once that stage is implemented.

---

## License

This project is created for educational purposes as part of the AICTE-2026 internship programme.  
Dataset license: see Kaggle dataset page linked above.
