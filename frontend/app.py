"""
frontend/app.py
===============
AICTE-2026 Problem Statement #39 — Predictive Maintenance of Industrial Machinery

Streamlit UI frontend.
Sends all prediction requests to the Flask API at http://localhost:5000.
No ML logic lives here.

Usage
-----
    streamlit run frontend/app.py
    (runs on port 8501 by default)
"""

from __future__ import annotations

import requests
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:5000"
PREDICT_URL  = f"{API_BASE_URL}/predict"

# Feature ranges sourced from FEATURE_INFO in backend/app.py.
MACHINE_TYPES = ["L", "M", "H"]


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance — AICTE 2026",
    page_icon="🔧",
    layout="centered",
)

st.title("🔧 Predictive Maintenance of Industrial Machinery")
st.caption("AICTE-2026 | Problem Statement #39")
st.markdown(
    "Enter the current sensor readings below and click **Predict** to check "
    "whether the machine is at risk of failure."
)
st.divider()


# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Sensor Inputs")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        options=MACHINE_TYPES,
        index=1,           # default: M
        help="Quality variant of the machine (L = Low, M = Medium, H = High)",
    )

    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=270.0,
        max_value=330.0,
        value=300.0,
        step=0.1,
        format="%.1f",
        help="Ambient air temperature in Kelvin. Dataset range: 295.3 – 304.5 K",
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=270.0,
        max_value=330.0,
        value=310.0,
        step=0.1,
        format="%.1f",
        help="Process temperature in Kelvin. Dataset range: 305.7 – 313.8 K",
    )

with col2:
    rot_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=500,
        max_value=4000,
        value=1500,
        step=10,
        help="Rotational speed in RPM. Dataset range: 1168 – 2886 rpm",
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=200.0,
        value=40.0,
        step=0.1,
        format="%.1f",
        help="Machine torque in Newton-metres. Dataset range: 3.8 – 76.6 Nm",
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0,
        max_value=500,
        value=50,
        step=1,
        help="Accumulated tool wear time in minutes. Dataset range: 0 – 253 min",
    )

st.divider()


# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

if predict_clicked:
    payload: dict = {
        "Type":                       machine_type,
        "Air temperature [K]":        air_temp,
        "Process temperature [K]":    process_temp,
        "Rotational speed [rpm]":     rot_speed,
        "Torque [Nm]":                torque,
        "Tool wear [min]":            tool_wear,
    }

    # ── Call the Flask API ────────────────────────────────────────────────────
    try:
        response = requests.post(PREDICT_URL, json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        st.error(
            "**Cannot reach the Flask API.**  \n"
            "Make sure the Flask server is running on port 5000:  \n"
            "`python backend/app.py`"
        )
        st.stop()
    except requests.exceptions.Timeout:
        st.error(
            "**The Flask API did not respond within 10 seconds.**  \n"
            "The server may be overloaded. Please try again."
        )
        st.stop()

    # ── Handle HTTP errors ────────────────────────────────────────────────────
    if response.status_code == 400:
        error_msg = response.json().get("error", "Unknown validation error.")
        st.error(f"**Invalid input:** {error_msg}")
        st.stop()

    if response.status_code != 200:
        st.error(
            f"**Unexpected API error (HTTP {response.status_code}).**  \n"
            f"{response.text[:300]}"
        )
        st.stop()

    # ── Parse and display result ──────────────────────────────────────────────
    result: dict = response.json()

    failure_predicted:   bool        = result["failure_predicted"]
    failure_prob:        float       = result["failure_probability"]
    failure_type:        str | None  = result.get("failure_type")
    failure_type_prob:   float | None = result.get("failure_type_probability")

    st.subheader("Prediction Result")

    if not failure_predicted:
        st.success("✅ **No failure predicted.** The machine appears to be operating normally.")
        st.metric(
            label="Failure Probability",
            value=f"{failure_prob * 100:.2f}%",
            delta=None,
        )
        st.caption(
            "A low probability indicates normal operating conditions based on "
            "the current sensor readings."
        )

    else:
        st.error("⚠️ **Machine failure predicted!** Immediate inspection is recommended.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                label="Failure Probability",
                value=f"{failure_prob * 100:.2f}%",
            )
        with col_b:
            if failure_type_prob is not None:
                st.metric(
                    label="Failure Type Confidence",
                    value=f"{failure_type_prob * 100:.2f}%",
                )

        if failure_type:
            st.info(f"**Predicted Failure Type:** {failure_type}")

        # Plain-English explanation based on failure type.
        explanations: dict[str, str] = {
            "Heat Dissipation Failure": (
                "High air or process temperature may be causing inadequate heat "
                "dissipation. Check the cooling system."
            ),
            "Power Failure": (
                "Abnormal torque or rotational speed combination may be drawing "
                "excessive power. Inspect the power supply and drive."
            ),
            "Overstrain Failure": (
                "High torque combined with low rotational speed may indicate "
                "mechanical overload. Reduce load or inspect the drive train."
            ),
            "Tool Wear Failure": (
                "Accumulated tool wear has likely reached a critical level. "
                "Replace or inspect the cutting tool immediately."
            ),
            "Random Failures": (
                "The failure signature does not match a specific pattern. "
                "A general inspection of all machine components is advised."
            ),
        }
        explanation = explanations.get(
            failure_type,
            "Review all sensor readings and inspect the machine thoroughly.",
        )
        st.warning(f"💡 **Possible cause:** {explanation}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AICTE-2026 Problem Statement #39 · Predictive Maintenance of Industrial Machinery · "
    "Model trained on the Kaggle Machine Predictive Maintenance Classification dataset."
)
