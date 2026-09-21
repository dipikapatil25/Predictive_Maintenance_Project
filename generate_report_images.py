from pathlib import Path
import os
import ast

os.environ["MPLBACKEND"] = "Agg"

import nbformat
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    f1_score,
    roc_auc_score,
)

# --------------------------------------------------
# Project paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "predictive_maintenance.csv"
MODEL_FILE = ROOT / "model" / "best_model.pkl"
NOTEBOOK_FILE = ROOT / "Predictive_Maintenance.ipynb"
OUTPUT_DIR = ROOT / "report_images"

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Read RANDOM_STATE from the notebook
# --------------------------------------------------
random_state = 42

nb = nbformat.read(NOTEBOOK_FILE, as_version=4)

for cell in nb.cells:
    if cell.cell_type != "code":
        continue

    try:
        tree = ast.parse(cell.source)
    except SyntaxError:
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "RANDOM_STATE"
                    and isinstance(node.value, ast.Constant)
                ):
                    random_state = node.value.value

print("Random state:", random_state)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv(DATA_FILE, sep="\t")

print("Dataset shape:", df.shape)

# --------------------------------------------------
# Feature definitions
# --------------------------------------------------
TARGET = "Target"
FAILURE_TYPE = "Failure Type"
CATEGORICAL = "Type"

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FEATURES = [CATEGORICAL] + NUMERIC_FEATURES

sns.set_theme(style="whitegrid")


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path.name)


# ==================================================
# 1. Target class distribution
# ==================================================
target_counts = df[TARGET].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(
    ["No Failure", "Failure"],
    [
        target_counts.get(0, 0),
        target_counts.get(1, 0),
    ],
)
axes[0].set_title("Target Class Distribution")
axes[0].set_ylabel("Number of Records")

axes[1].pie(
    target_counts.values,
    labels=["No Failure", "Failure"],
    autopct="%1.1f%%",
    startangle=90,
)
axes[1].set_title("Target Class Proportion")

fig.tight_layout()

save_figure(fig, "01_target_class_distribution.png")


# ==================================================
# 2. Failure type distribution
# ==================================================
failure_counts = df[FAILURE_TYPE].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))

failure_counts.sort_values().plot(
    kind="barh",
    ax=ax,
)

ax.set_title("Failure Type Distribution")
ax.set_xlabel("Number of Records")
ax.set_ylabel("Failure Type")

fig.tight_layout()

save_figure(fig, "02_failure_type_distribution.png")


# ==================================================
# 3. Machine type distribution and failure rate
# ==================================================
machine_counts = df[CATEGORICAL].value_counts().sort_index()
failure_rate = df.groupby(CATEGORICAL)[TARGET].mean().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(machine_counts.index, machine_counts.values)
axes[0].set_title("Machine Type Distribution")
axes[0].set_xlabel("Machine Type")
axes[0].set_ylabel("Number of Records")

axes[1].bar(
    failure_rate.index,
    failure_rate.values * 100,
)
axes[1].set_title("Failure Rate by Machine Type")
axes[1].set_xlabel("Machine Type")
axes[1].set_ylabel("Failure Rate (%)")

fig.tight_layout()

save_figure(fig, "03_machine_type_distribution.png")


# ==================================================
# 4. Numeric feature distributions
# ==================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, feature in enumerate(NUMERIC_FEATURES):
    sns.histplot(
        data=df,
        x=feature,
        hue=TARGET,
        kde=True,
        stat="density",
        common_norm=False,
        ax=axes[i],
    )
    axes[i].set_title(feature)

# Remove unused sixth axis if necessary
if len(NUMERIC_FEATURES) < len(axes):
    for j in range(len(NUMERIC_FEATURES), len(axes)):
        axes[j].axis("off")

fig.suptitle(
    "Numeric Feature Distributions by Failure Status",
    fontsize=16,
)

fig.tight_layout()

save_figure(fig, "04_sensor_feature_distributions.png")


# ==================================================
# 5. Correlation heatmap
# ==================================================
corr_features = NUMERIC_FEATURES + [TARGET]
corr = df[corr_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ax,
)

ax.set_title("Correlation Heatmap")

fig.tight_layout()

save_figure(fig, "05_correlation_heatmap.png")


# ==================================================
# 6. Box plots by target
# ==================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, feature in enumerate(NUMERIC_FEATURES):
    sns.boxplot(
        data=df,
        x=TARGET,
        y=feature,
        ax=axes[i],
    )
    axes[i].set_title(feature)
    axes[i].set_xlabel("Failure Status")

if len(NUMERIC_FEATURES) < len(axes):
    for j in range(len(NUMERIC_FEATURES), len(axes)):
        axes[j].axis("off")

fig.suptitle(
    "Sensor Features by Failure Status",
    fontsize=16,
)

fig.tight_layout()

save_figure(fig, "06_boxplots_by_failure_status.png")


# ==================================================
# Load the saved primary model
# ==================================================
model = joblib.load(MODEL_FILE)

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=random_state,
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred, average="macro")
roc_auc = roc_auc_score(y_test, y_prob)

print("Test F1-macro:", round(f1, 4))
print("Test ROC-AUC:", round(roc_auc, 4))


# ==================================================
# 7. Confusion matrix
# ==================================================
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Failure", "Failure"],
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)

ax.set_title(
    f"Confusion Matrix\nF1-macro={f1:.4f} | ROC-AUC={roc_auc:.4f}"
)

fig.tight_layout()

save_figure(fig, "07_confusion_matrix.png")

tn, fp, fn, tp = cm.ravel()

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# ==================================================
# 8. ROC curve
# ==================================================
fig, ax = plt.subplots(figsize=(7, 6))

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    ax=ax,
)

ax.set_title(
    f"ROC Curve\nROC-AUC={roc_auc:.4f}"
)

fig.tight_layout()

save_figure(fig, "08_roc_curve.png")


# ==================================================
# 9. Feature importance
# ==================================================
classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

if hasattr(classifier, "feature_importances_"):
    importances = classifier.feature_importances_
    method = "Native feature importance"

else:
    from sklearn.inspection import permutation_importance

    transformed_test = preprocessor.transform(X_test)

    result = permutation_importance(
        classifier,
        transformed_test,
        y_test,
        scoring="f1_macro",
        n_repeats=10,
        random_state=random_state,
    )

    importances = result.importances_mean
    method = "Permutation importance"

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importances,
    }
).sort_values(
    "Importance",
    ascending=False,
)

fig, ax = plt.subplots(figsize=(10, 6))

top_features = importance_df.head(10).sort_values("Importance")

ax.barh(
    top_features["Feature"],
    top_features["Importance"],
)

ax.set_title(f"Top Feature Importance ({method})")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")

fig.tight_layout()

save_figure(fig, "09_feature_importance.png")


# ==================================================
# Final check
# ==================================================
print("\nReport images created successfully.")
print("Files in report_images:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(" -", file.name)