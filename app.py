from __future__ import annotations

import json
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌱",
    layout="centered",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "crop_recommendation_model.pkl"
AUDIT_LOG_PATH = APP_DIR / "audit_log.jsonl"

FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

@st.cache_resource
def load_artifact():
    """Load the model and its matching preprocessing artifacts."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise TypeError("Expected the model artifact to be a dictionary.")

    required_keys = {
        "model",
        "scaler",
        "label_encoder",
        "classes",
        "n_classes",
    }
    missing = sorted(required_keys - set(artifact.keys()))
    if missing:
        raise KeyError(
            f"Model artifact is missing {missing}. "
            "Re-export the model with its fitted scaler and label encoder."
        )

    return artifact


def append_audit_log(payload: dict) -> None:
    """Append a prediction event to a JSONL audit file."""
    payload = dict(payload)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def get_model_metadata(artifact: dict) -> dict:
    """Return a friendly model metadata dictionary."""
    metadata = {
        "model_type": type(artifact["model"]).__name__,
        "test_accuracy": artifact.get("test_accuracy", None),
        "classes": list(artifact.get("classes", [])),
        "feature_order": FEATURE_ORDER,
        "model_name": artifact.get("model_name", "CropRecommendationRandomForest"),
        "training_date": artifact.get("training_date", "Not specified"),
    }
    return metadata


def simple_drift_check(values: dict, baseline: dict | None = None) -> str:
    """
    Lightweight drift indicator that is human-readable and consistent with the
    README's monitoring section.

    If the model artifact contains a baseline_stats dictionary, it is used.
    Otherwise, a safe fallback is returned.
    """
    if not baseline:
        return "Stable (baseline unavailable)"

    try:
        # Simple heuristic using mean comparison
        deviations = []
        for key in FEATURE_ORDER:
            if key in values and key in baseline:
                current_value = float(values[key])
                mean_val = float(baseline[key].get("mean", current_value))
                std_val = float(baseline[key].get("std", 1.0))
                if std_val == 0:
                    std_val = 1.0
                z_score = abs(current_value - mean_val) / std_val
                deviations.append(z_score)
        if not deviations:
            return "Stable (baseline unavailable)"
        mean_dev = float(np.mean(deviations))
        if mean_dev > 3.0:
            return "Drift risk detected"
        if mean_dev > 1.5:
            return "Watchlist"
        return "Stable"
    except Exception:
        return "Stable (baseline unavailable)"


with st.spinner("Loading crop recommendation model..."):
    artifact = load_artifact()

model = artifact["model"]
scaler = artifact["scaler"]
label_encoder = artifact["label_encoder"]
model_meta = get_model_metadata(artifact)

# Model compatibility validation
if not (
    len(artifact["classes"])
    == artifact["n_classes"]
    == len(label_encoder.classes_)
    == len(model.classes_)
):
    st.error(
        "Model artifact mismatch: the model, classes, and label encoder "
        "do not have the same number of classes."
    )
    st.stop()

st.title("🌱 Intelligent Crop Recommendation System")
st.caption(
    f"FUSE AI-201 | Group 11 | {model_meta['model_type']} "
    f"({model_meta['test_accuracy']:.2%} test accuracy)"
    if model_meta["test_accuracy"] is not None
    else "FUSE AI-201 | Group 11 | Crop Recommendation App"
)

with st.sidebar:
    st.subheader("Model Details")
    st.write(f"Model: {model_meta['model_type']}")
    if model_meta["test_accuracy"] is not None:
        st.write(f"Test accuracy: {model_meta['test_accuracy']:.2%}")
    st.write(f"Model name: {model_meta['model_name']}")
    st.write(f"Training date: {model_meta['training_date']}")
    st.write("Deployment: https://crop-recomm-app.streamlit.app")
    st.write("Features: " + ", ".join(FEATURE_ORDER))

    # Optional drift baseline if present in model artifact
    baseline = artifact.get("baseline_stats")
    if baseline:
        st.write("Baseline drift check: available")
    else:
        st.write("Baseline drift check: not configured")

    if AUDIT_LOG_PATH.exists():
        st.caption(f"Audit log present: {AUDIT_LOG_PATH.name}")
    else:
        st.caption("No audit log yet; it will be created when predictions run.")

with st.form("crop_form"):
    st.subheader("Soil & Climate Measurements")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N) mg/kg", min_value=0.0, max_value=200.0, value=90.0)
        P = st.number_input("Phosphorus (P) mg/kg", min_value=0.0, max_value=200.0, value=42.0)
        K = st.number_input("Potassium (K) mg/kg", min_value=0.0, max_value=250.0, value=43.0)
        ph = st.slider("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    with col2:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=20.8)
        humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=202.9)

    submitted = st.form_submit_button("🌾 Recommend Crop", use_container_width=True)

if submitted:
    try:
        checks = {
            "N": (N, 0, 200),
            "P": (P, 0, 200),
            "K": (K, 0, 250),
            "temperature": (temperature, -10, 60),
            "humidity": (humidity, 0, 100),
            "ph": (ph, 0, 14),
            "rainfall": (rainfall, 0, 400),
        }

        for name, (value, low, high) in checks.items():
            if not isfinite(value) or not (low <= value <= high):
                st.error(
                    f"Invalid {name}: {value}. "
                    f"Expected range [{low}, {high}]."
                )
                st.stop()

        row = np.array(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            dtype=np.float64,
        )

        scaled_row = scaler.transform(row)
        proba = model.predict_proba(scaled_row)[0]

        if len(proba) != len(model.classes_):
            raise ValueError("Model probability output does not match model classes.")

        idx = int(np.argmax(proba))
        class_id = int(model.classes_[idx])
        crop = label_encoder.inverse_transform([class_id])[0]
        confidence = float(proba[idx])

        current_values = {
            "N": float(N),
            "P": float(P),
            "K": float(K),
            "temperature": float(temperature),
            "humidity": float(humidity),
            "ph": float(ph),
            "rainfall": float(rainfall),
        }
        baseline_stats = artifact.get("baseline_stats")
        drift_status = simple_drift_check(current_values, baseline_stats)

        if confidence < 0.60:
            st.warning(
                f"⚠️ Low confidence ({confidence:.0%}). "
                "Consider confirming with a soil test."
            )

        st.success(f"### Recommended Crop: **{crop.title()}**")
        st.metric("Confidence Score", f"{confidence:.1%}")
        st.progress(confidence)

        st.caption(f"Drift status: {drift_status}")

        top_idx = np.argsort(proba)[::-1][:5]
        top_class_ids = np.asarray(model.classes_)[top_idx].astype(int)
        top_df = {
            "Crop": label_encoder.inverse_transform(top_class_ids),
            "Probability": [f"{p:.2%}" for p in proba[top_idx]],
        }

        st.subheader("Top 5 Alternatives")
        st.dataframe(top_df, width="stretch", hide_index=True)

        # Audit logging
        append_audit_log(
            {
                "input_values": current_values,
                "predicted_crop": crop,
                "confidence": confidence,
                "top_5": [str(x) for x in label_encoder.inverse_transform(top_class_ids)],
                "drift_status": drift_status,
                "model_name": model_meta["model_name"],
            }
        )

    except Exception as e:
        st.error(f"Prediction failed: {type(e).__name__}: {e}")

st.divider()
st.warning(
    "**Responsible AI Notice**\n\n"
    "This recommendation is advisory only and should not replace "
    "local soil testing or professional agronomic guidance."
)