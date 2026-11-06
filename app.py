from math import isfinite
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌱",
    layout="centered",
)

# Use a path relative to this file so the app works from any working directory.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "crop_recommendation_model.pkl"


@st.cache_resource
def load_artifact():
    """Load the trained model artifact and validate its structure."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise TypeError("Expected the model artifact to be a dictionary.")

    required_keys = {"model", "label_encoder", "classes", "n_classes", "test_accuracy"}
    missing = sorted(required_keys - set(artifact.keys()))
    if missing:
        raise KeyError(f"Model artifact missing required keys: {missing}")

    return artifact


with st.spinner("Loading crop recommendation model..."):
    art = load_artifact()

model = art["model"]
le = art["label_encoder"]

# --------------------------------------------------
# Integrity Check
# --------------------------------------------------
if not (len(art["classes"]) == art["n_classes"] == len(le.classes_)):
    st.error(
        f"Artifact mismatch: "
        f"n_classes={art['n_classes']}, "
        f"classes={len(art['classes'])}, "
        f"encoder={len(le.classes_)}"
    )
    st.stop()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🌱 Intelligent Crop Recommendation System")
st.caption(
    f"FUSE AI-201 | Group 11 | "
    f"{type(model).__name__} "
    f"({art['test_accuracy']:.2%} test accuracy)"
)

# --------------------------------------------------
# Inputs
# --------------------------------------------------
with st.form("crop_form"):
    st.subheader("Soil & Climate Measurements")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input(
            "Nitrogen (N) mg/kg",
            min_value=0.0,
            max_value=200.0,
            value=90.0,
        )

        P = st.number_input(
            "Phosphorus (P) mg/kg",
            min_value=0.0,
            max_value=200.0,
            value=42.0,
        )

        K = st.number_input(
            "Potassium (K) mg/kg",
            min_value=0.0,
            max_value=250.0,
            value=43.0,
        )

        ph = st.slider(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1,
        )

    with col2:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=20.8,
        )

        humidity = st.slider(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=82.0,
            step=0.1,
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=400.0,
            value=202.9,
        )

    submitted = st.form_submit_button(
        "🌾 Recommend Crop",
        use_container_width=True,
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
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
                    f"Expected range [{low}, {high}]"
                )
                st.stop()

        row = np.array(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            dtype=np.float32,
        )

        if not hasattr(model, "predict_proba"):
            raise AttributeError("The loaded model does not implement predict_proba().")

        proba = model.predict_proba(row)[0]

        if len(proba) != len(le.classes_):
            raise ValueError(
                f"Probability length mismatch: expected {len(le.classes_)} but got {len(proba)}"
            )

        idx = int(np.argmax(proba))
        crop = le.inverse_transform([idx])[0]
        confidence = float(proba[idx])

        # ------------------------------------------
        # Results
        # ------------------------------------------
        if confidence < 0.60:
            st.warning(
                f"⚠️ Low confidence ({confidence:.0%}). "
                f"Consider confirming with a soil test."
            )

        st.success(f"### Recommended Crop: **{crop.title()}**")
        st.metric("Confidence Score", f"{confidence:.1%}")
        st.progress(confidence)

        # ------------------------------------------
        # Top 5 Alternatives
        # ------------------------------------------
        top_idx = np.argsort(proba)[::-1][:5]
        top_df = {
            "Crop": le.inverse_transform(top_idx),
            "Probability": [f"{p:.2%}" for p in proba[top_idx]],
        }

        st.subheader("Top 5 Alternatives")
        st.dataframe(
            top_df,
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        st.error(f"Prediction failed: {type(e).__name__}: {e}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.warning(
    "**Responsible AI Notice**\n\n"
    "This recommendation is advisory only and should "
    "not replace local soil testing or professional "
    "agronomic guidance."
)