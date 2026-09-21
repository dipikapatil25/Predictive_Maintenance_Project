"""
train_model.py
==============
AICTE-2026 Problem Statement #39 — Predictive Maintenance of Industrial Machinery

Reproducible, standalone training script.

Usage
-----
    python train_model.py

What this script does
---------------------
1. Loads the raw dataset from data/predictive_maintenance.csv
2. Cleans and validates the data
3. Builds a ColumnTransformer preprocessor (OneHotEncoder + StandardScaler)
4. Performs a stratified 80/20 train/test split
5. 5-fold stratified CV on training data to compare 5 classifiers
6. Selects the best model by mean macro F1 across CV folds
7. Fits the best pipeline on the full training set
8. Evaluates once on the untouched test set
9. Saves model/best_model.pkl (preprocessing + classifier in one Pipeline)

No plots are generated here — see Predictive_Maintenance.ipynb for EDA
and visualisations.

NOTE: Run this script only after placing predictive_maintenance.csv
      inside the data/ folder.
"""

# ── Standard library ────────────────────────────────────────────────────────
import sys
from pathlib import Path
from typing import Tuple

# ── Third-party ──────────────────────────────────────────────────────────────
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ── Path constants ───────────────────────────────────────────────────────────
# Script lives at the project root, so __file__.parent IS the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
MODEL_DIR    = PROJECT_ROOT / "model"

RAW_CSV              = DATA_DIR  / "predictive_maintenance.csv"
MODEL_PKL            = MODEL_DIR / "best_model.pkl"
MODEL_FAILURE_TYPE_PKL = MODEL_DIR / "failure_type_model.pkl"

# ── Feature / target column names ────────────────────────────────────────────
# Source: Kaggle — Machine Predictive Maintenance Classification
# https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
#
# Columns dropped at load time (no predictive value):
DROP_COLUMNS = ["UDI", "Product ID"]

# Categorical feature encoded with OneHotEncoder (H/L/M are nominal — no ordering):
CATEGORICAL_FEATURE = "Type"                  # values: H, L, M

# Numeric features (scaled with StandardScaler):
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Final ordered feature list fed into the pipeline.
# IMPORTANT: backend/app.py must define an identical FEATURE_COLUMNS list.
FEATURE_COLUMNS = [CATEGORICAL_FEATURE] + NUMERIC_FEATURES

# Target columns:
TARGET_BINARY       = "Target"          # 0 = No Failure, 1 = Failure
TARGET_FAILURE_TYPE = "Failure Type"    # e.g. "No Failure", "Heat Dissipation Failure", ...

# ── Training constants ────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20    # 80 % train / 20 % test


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load the raw CSV and return it as a DataFrame (no modifications).

    The file is tab-separated — sep='\\t' is mandatory.
    """
    if not RAW_CSV.exists():
        print(
            f"ERROR: Dataset not found at {RAW_CSV}\n"
            f"  Place predictive_maintenance.csv in: {DATA_DIR}\n"
            "  Download from:\n"
            "  https://www.kaggle.com/datasets/shivamb/"
            "machine-predictive-maintenance-classification"
        )
        sys.exit(1)

    df = pd.read_csv(RAW_CSV, sep="\t")
    print(f"Loaded  : {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ── Data cleaning ─────────────────────────────────────────────────────────────
def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the raw DataFrame.

    Steps (mirrors Notebook Section 2):
      1. Assert zero null values.
      2. Assert zero duplicate rows.
      3. Assert Target ↔ Failure Type consistency.
      4. Drop non-predictive columns (UDI, Product ID).

    Returns the cleaned working DataFrame.
    """
    # 1. Null check
    null_total = df_raw.isnull().sum().sum()
    assert null_total == 0, f"Unexpected null values found: {null_total}"
    print(f"Null check  : PASS (0 null cells)")

    # 2. Duplicate check
    n_dup = df_raw.duplicated().sum()
    assert n_dup == 0, f"Unexpected duplicate rows found: {n_dup}"
    print(f"Dup check   : PASS (0 duplicate rows)")

    # 3. Consistency audit — reporting only (known quirks in published dataset):
    #    - 18 rows: Target=0 with Failure Type='Random Failures' (retained)
    #    - 9 rows:  Target=1 with Failure Type='No Failure'      (retained)
    rule_a = df_raw[(df_raw[TARGET_BINARY] == 0) & (df_raw[TARGET_FAILURE_TYPE] != "No Failure")]
    rule_b = df_raw[(df_raw[TARGET_BINARY] == 1) & (df_raw[TARGET_FAILURE_TYPE] == "No Failure")]
    print(f"Consistency : Rule A (Target=0, non-'No Failure' type) = {len(rule_a)} rows")
    print(f"Consistency : Rule B (Target=1, 'No Failure' type)     = {len(rule_b)} rows")
    print(f"Consistency : Known quirks in published dataset — rows retained as-is")

    # 4. Drop non-predictive columns
    df = df_raw.drop(columns=DROP_COLUMNS)
    print(f"Dropped     : {DROP_COLUMNS}")
    print(f"Working df  : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns     : {df.columns.tolist()}")

    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer with OneHotEncoder + StandardScaler.

    Type (nominal categorical) → OneHotEncoder(handle_unknown='ignore')
    Five numeric sensors        → StandardScaler

    The transformer is returned unfitted; it is fit on training data only
    (inside the Pipeline) to prevent data leakage.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [CATEGORICAL_FEATURE],
            ),
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ── Train / test split ────────────────────────────────────────────────────────
def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Extract feature matrix X and target y, then perform a stratified split.

    Returns (X_train, X_test, y_train, y_test).
    Stratification preserves the ~3.4% failure class ratio in both splits.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_BINARY].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"X_train : {X_train.shape}   X_test : {X_test.shape}")
    print(f"y_train : {y_train.shape}   y_test : {y_test.shape}")
    for split_name, split_y in [("y_train", y_train), ("y_test", y_test)]:
        vc = split_y.value_counts().sort_index()
        row = "  ".join(
            f"class {v}={cnt:,} ({cnt / len(split_y) * 100:.2f}%)"
            for v, cnt in vc.items()
        )
        print(f"  {split_name}: {row}")

    return X_train, X_test, y_train, y_test


# ── Candidate models ──────────────────────────────────────────────────────────
def _build_candidates() -> dict:
    """Return a dict of {name: unfitted_classifier}."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4,
            random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    }


# ── Cross-validation comparison ───────────────────────────────────────────────
def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[str, pd.DataFrame]:
    """Run 5-fold stratified CV on each candidate and return (best_name, results_df).

    Selection criterion: highest mean macro F1 across folds.
    The test set is never touched here.
    """
    candidates  = _build_candidates()
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scoring  = ["f1_macro", "roc_auc", "accuracy", "precision_macro", "recall_macro"]

    rows = {}
    print("5-fold stratified CV on training data:")
    for name, clf in candidates.items():
        pipe   = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
        scores = cross_validate(
            pipe, X_train, y_train,
            cv=cv_strategy, scoring=cv_scoring,
            return_train_score=False, n_jobs=1,
        )
        rows[name] = {
            "F1-macro (mean)":  scores["test_f1_macro"].mean(),
            "F1-macro (std)":   scores["test_f1_macro"].std(),
            "ROC-AUC (mean)":   scores["test_roc_auc"].mean(),
            "Accuracy (mean)":  scores["test_accuracy"].mean(),
            "Precision-macro":  scores["test_precision_macro"].mean(),
            "Recall-macro":     scores["test_recall_macro"].mean(),
        }
        print(
            f"  {name:25s}  "
            f"F1-macro={rows[name]['F1-macro (mean)']:.4f}  "
            f"ROC-AUC={rows[name]['ROC-AUC (mean)']:.4f}"
        )

    results_df      = pd.DataFrame(rows).T.sort_values("F1-macro (mean)", ascending=False)
    best_model_name = results_df.index[0]
    print(f"\nSelected  : {best_model_name!r}  "
          f"(F1-macro={results_df.iloc[0]['F1-macro (mean)']:.4f})")
    return best_model_name, results_df


# ── Final fit and evaluation ──────────────────────────────────────────────────
def fit_and_evaluate(
    best_model_name: str,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test:  pd.DataFrame, y_test:  pd.Series,
) -> Pipeline:
    """Fit the best pipeline on the full training set and evaluate on test set.

    Returns the fitted Pipeline.
    """
    candidates = _build_candidates()
    best_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   candidates[best_model_name]),
    ])
    best_pipeline.fit(X_train, y_train)

    y_pred      = best_pipeline.predict(X_test)
    y_pred_prob = best_pipeline.predict_proba(X_test)[:, 1]

    print(f"\nTest-set evaluation  (model: {best_model_name})")
    print("=" * 52)
    print(f"  F1-macro    : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"  ROC-AUC     : {roc_auc_score(y_test, y_pred_prob):.4f}")
    print(f"  Precision   : {precision_score(y_test, y_pred, average='macro'):.4f}  (macro)")
    print(f"  Recall      : {recall_score(y_test, y_pred, average='macro'):.4f}  (macro)")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["No Failure (0)", "Failure (1)"]
    ))
    return best_pipeline


# ── Save model ────────────────────────────────────────────────────────────────
def save_model(pipeline: Pipeline) -> None:
    """Dump the fitted pipeline to MODEL_PKL and verify reload."""
    joblib.dump(pipeline, MODEL_PKL)
    loaded   = joblib.load(MODEL_PKL)
    size_kb  = MODEL_PKL.stat().st_size / 1024
    print(f"Model saved  : {MODEL_PKL}  ({size_kb:.1f} KB)")
    print(f"Reload check : {type(loaded).__name__} — PASS")


# ── Secondary: failure-type classifier ────────────────────────────────────────
def build_failure_type_model(df: pd.DataFrame) -> Pipeline:
    """Train a multiclass failure-type classifier on the failure-only subset.

    Workflow (mirrors Notebook Section 7):
      - Subset: Target==1 AND Failure Type != 'No Failure'  (330 rows)
      - 'Random Failures' is absent from this subset (all 18 rows have Target=0)
      - 9 quirk rows (Target=1, Failure Type='No Failure') are excluded
      - Stratified 80/20 split, 5-fold CV, best model by macro F1
      - Saved to MODEL_FAILURE_TYPE_PKL

    Returns the fitted Pipeline.
    """
    # Build subset
    df_failures = df[
        (df[TARGET_BINARY] == 1) &
        (df[TARGET_FAILURE_TYPE] != "No Failure")
    ].copy()

    X_ft = df_failures[FEATURE_COLUMNS].copy()
    y_ft = df_failures[TARGET_FAILURE_TYPE].copy()

    print(f"Failure-type subset : {len(df_failures)} rows")
    for cls, cnt in y_ft.value_counts().items():
        print(f"  {cls:35s}: {cnt}")

    # Stratified split
    X_ft_train, X_ft_test, y_ft_train, y_ft_test = train_test_split(
        X_ft, y_ft,
        test_size=0.20, random_state=RANDOM_STATE, stratify=y_ft,
    )
    print(f"Split: train={len(X_ft_train)}  test={len(X_ft_test)}")

    def _ft_preprocessor() -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 [CATEGORICAL_FEATURE]),
                ("num", StandardScaler(), NUMERIC_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )

    ft_candidates = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3,
            random_state=RANDOM_STATE),
    }

    cv_ft    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_ft_name  = None
    best_ft_score = -1.0

    print("Failure-type CV:")
    for name, clf in ft_candidates.items():
        pipe   = Pipeline([("preprocessor", _ft_preprocessor()), ("classifier", clf)])
        scores = cross_validate(
            pipe, X_ft_train, y_ft_train,
            cv=cv_ft, scoring=["f1_macro"],
            return_train_score=False, n_jobs=1,
        )
        mean_f1 = scores["test_f1_macro"].mean()
        print(f"  {name:25s}  F1-macro={mean_f1:.4f}")
        if mean_f1 > best_ft_score:
            best_ft_score, best_ft_name = mean_f1, name

    print(f"Selected: {best_ft_name!r}  (F1-macro={best_ft_score:.4f})")

    # Fit on full training subset
    best_ft_pipeline = Pipeline([
        ("preprocessor", _ft_preprocessor()),
        ("classifier",   ft_candidates[best_ft_name]),
    ])
    best_ft_pipeline.fit(X_ft_train, y_ft_train)

    # Evaluate on test subset
    y_ft_pred = best_ft_pipeline.predict(X_ft_test)
    ft_classes = sorted(y_ft.unique())
    print(f"\nFailure-type test-set report:")
    print(classification_report(
        y_ft_test, y_ft_pred,
        labels=ft_classes, target_names=ft_classes, zero_division=0,
    ))
    print("CAUTION: smallest class (Tool Wear Failure, ~9 test samples) has high metric variance.")

    # Save + verify
    joblib.dump(best_ft_pipeline, MODEL_FAILURE_TYPE_PKL)
    loaded_ft = joblib.load(MODEL_FAILURE_TYPE_PKL)
    assert (loaded_ft.predict(X_ft_test) == y_ft_pred).all()
    size_kb = MODEL_FAILURE_TYPE_PKL.stat().st_size / 1024
    print(f"\nSaved  : {MODEL_FAILURE_TYPE_PKL}  ({size_kb:.1f} KB)")
    print(f"Reload : PASS")

    return best_ft_pipeline


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Entry point."""
    print("=" * 60)
    print("Predictive Maintenance — Model Training Script")
    print("AICTE-2026 Problem Statement #39")
    print("=" * 60)
    print()

    df_raw = load_data()
    df     = clean_data(df_raw)
    print()

    X_train, X_test, y_train, y_test = split_data(df)
    print()

    best_name, results_df = compare_models(X_train, y_train)
    print()
    print(results_df.round(4).to_string())
    print()

    best_pipeline = fit_and_evaluate(best_name, X_train, y_train, X_test, y_test)
    save_model(best_pipeline)

    print()
    print("-" * 60)
    print("Secondary: Failure-type classifier")
    print("-" * 60)
    build_failure_type_model(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
