"""
backend/app.py
==============
AICTE-2026 Problem Statement #39 — Predictive Maintenance of Industrial Machinery

Flask REST API backend.

Usage
-----
    python backend/app.py

Endpoints
---------
    GET  /              Health check
    GET  /feature-info  Returns feature names, types, and valid ranges
    POST /predict       Accepts sensor values; returns failure prediction +
                        failure type (when applicable)

Port: 5000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────────────────────
# backend/app.py lives one level inside the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR    = PROJECT_ROOT / "model"

PRIMARY_MODEL_PATH   = MODEL_DIR / "best_model.pkl"
SECONDARY_MODEL_PATH = MODEL_DIR / "failure_type_model.pkl"

# ── Feature contract ──────────────────────────────────────────────────────────
# These must exactly match the FEATURE_COLUMNS used during training.
FEATURE_COLUMNS: list[str] = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Valid values and ranges sourced from the actual dataset (verified in Step 2).
FEATURE_INFO: dict[str, Any] = {
    "Type": {
        "type": "categorical",
        "valid_values": ["L", "M", "H"],
        "description": "Machine quality variant (L=Low, M=Medium, H=High)",
    },
    "Air temperature [K]": {
        "type": "numeric",
        "min": 295.3,
        "max": 304.5,
        "unit": "Kelvin",
        "description": "Ambient air temperature",
    },
    "Process temperature [K]": {
        "type": "numeric",
        "min": 305.7,
        "max": 313.8,
        "unit": "Kelvin",
        "description": "Process temperature",
    },
    "Rotational speed [rpm]": {
        "type": "numeric",
        "min": 1168,
        "max": 2886,
        "unit": "RPM",
        "description": "Rotational speed",
    },
    "Torque [Nm]": {
        "type": "numeric",
        "min": 3.8,
        "max": 76.6,
        "unit": "Newton-metres",
        "description": "Machine torque",
    },
    "Tool wear [min]": {
        "type": "numeric",
        "min": 0,
        "max": 253,
        "unit": "Minutes",
        "description": "Accumulated tool wear time",
    },
}

# ── App initialisation ────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (required for Streamlit → Flask calls)

# Load both models once at startup — not on every request.
print("Loading models...")

if not PRIMARY_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Primary model not found: {PRIMARY_MODEL_PATH}\n"
        "Run `python train_model.py` first."
    )
if not SECONDARY_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Secondary model not found: {SECONDARY_MODEL_PATH}\n"
        "Run `python train_model.py` first."
    )

primary_model   = joblib.load(PRIMARY_MODEL_PATH)
secondary_model = joblib.load(SECONDARY_MODEL_PATH)

print(f"  Primary model   : {PRIMARY_MODEL_PATH.name}  "
      f"steps={[s for s, _ in primary_model.steps]}")
print(f"  Secondary model : {SECONDARY_MODEL_PATH.name}  "
      f"steps={[s for s, _ in secondary_model.steps]}")
print("Models ready.")


# ── Input validation ──────────────────────────────────────────────────────────
def _validate_input(data: dict) -> tuple[dict | None, str | None]:
    """Validate and coerce the incoming JSON payload.

    Returns (coerced_dict, None) on success, (None, error_message) on failure.
    """
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."

    missing = [f for f in FEATURE_COLUMNS if f not in data]
    if missing:
        return None, f"Missing required fields: {missing}"

    coerced: dict[str, Any] = {}

    # Categorical: Type
    type_val = str(data["Type"]).strip().upper()
    if type_val not in ("L", "M", "H"):
        return None, f"'Type' must be one of L, M, H. Got: {data['Type']!r}"
    coerced["Type"] = type_val

    # Numeric fields
    numeric_fields = {
        "Air temperature [K]":     (float, 270.0, 330.0),
        "Process temperature [K]": (float, 270.0, 330.0),
        "Rotational speed [rpm]":  (float, 500.0, 4000.0),
        "Torque [Nm]":             (float, 0.0,   200.0),
        "Tool wear [min]":         (float, 0.0,   500.0),
    }
    for field, (cast, lo, hi) in numeric_fields.items():
        try:
            val = cast(data[field])
        except (TypeError, ValueError):
            return None, f"'{field}' must be a number. Got: {data[field]!r}"
        if not (lo <= val <= hi):
            return None, (
                f"'{field}' value {val} is outside the accepted range "
                f"[{lo}, {hi}]."
            )
        coerced[field] = val

    return coerced, None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "message": "Predictive Maintenance API is running.",
        "models_loaded": {
            "primary":   PRIMARY_MODEL_PATH.name,
            "secondary": SECONDARY_MODEL_PATH.name,
        },
        "endpoints": {
            "health":       "GET  /",
            "feature_info": "GET  /feature-info",
            "predict":      "POST /predict",
        },
    })


@app.get("/feature-info")
def feature_info():
    """Return feature names, types, valid ranges, and units for the UI."""
    return jsonify({
        "feature_columns": FEATURE_COLUMNS,
        "features": FEATURE_INFO,
    })


@app.post("/predict")
def predict():
    """Accept sensor values and return failure prediction.

    Request body (JSON):
    {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50
    }

    Response (JSON):
    {
        "failure_predicted": false,
        "failure_probability": 0.023,
        "failure_type": null,
        "failure_type_probability": null,
        "message": "No failure predicted."
    }

    When failure_predicted is true:
    {
        "failure_predicted": true,
        "failure_probability": 0.872,
        "failure_type": "Overstrain Failure",
        "failure_type_probability": 0.741,
        "message": "Machine failure predicted. Likely type: Overstrain Failure."
    }
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    coerced, error = _validate_input(data)
    if error:
        return jsonify({"error": error}), 400

    # Build a single-row DataFrame in the exact feature column order.
    row_df = pd.DataFrame([{col: coerced[col] for col in FEATURE_COLUMNS}])

    # Stage 1 — Primary binary prediction.
    binary_pred  = int(primary_model.predict(row_df)[0])
    binary_proba = primary_model.predict_proba(row_df)[0]
    failure_prob = float(binary_proba[1])

    if binary_pred == 0:
        return jsonify({
            "failure_predicted":        False,
            "failure_probability":      round(failure_prob, 4),
            "failure_type":             None,
            "failure_type_probability": None,
            "message": "No failure predicted.",
        })

    # Stage 2 — Secondary failure-type prediction (only when binary == 1).
    ft_pred  = secondary_model.predict(row_df)[0]
    ft_proba = secondary_model.predict_proba(row_df)[0]
    ft_conf  = float(ft_proba.max())

    return jsonify({
        "failure_predicted":        True,
        "failure_probability":      round(failure_prob, 4),
        "failure_type":             str(ft_pred),
        "failure_type_probability": round(ft_conf, 4),
        "message": f"Machine failure predicted. Likely type: {ft_pred}.",
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
