import streamlit as st
import pandas as pd
import joblib
from math import isfinite

st.set_page_config(page_title="Crop Recommendation", page_icon="🌱", layout="centered")

@st.cache_resource
def load_artifact():
    return joblib.load("crop_recommendation_model.pkl")

art      = load_artifact()
model    = art["model"]
le       = art["label_encoder"]
FEATURES = art["features"]

# Integrity check — fail visibly in the browser, not the terminal
if not (len(art["classes"]) == art["n_classes"] == len(le.classes_)):
    st.error(
        f"⚠️ Artifact mismatch — n_classes={art['n_classes']}, "
        f"classes={len(art['classes'])}, encoder={len(le.classes_)}. "
        f"Re-run Cell 10."
    )
    st.stop()

st.title("🌱 Intelligent Crop Recommendation System")
st.caption(f"FUSE AI-201 | Group 11 | {type(model).__name__} "
           f"({art['test_accuracy']:.2%} test accuracy)")

with st.form("inputs"):
    st.subheader("Soil measurements")
    c1, c2 = st.columns(2)
    with c1:
        N  = st.number_input("Nitrogen (N) — mg/kg",   0.0, 200.0, 90.0, 1.0)
        P  = st.number_input("Phosphorus (P) — mg/kg", 0.0, 200.0, 42.0, 1.0)
        K  = st.number_input("Potassium (K) — mg/kg",  0.0, 250.0, 43.0, 1.0)
        ph = st.slider("Soil pH",                      0.0, 14.0,   6.5, 0.1)
    with c2:
        temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 20.8, 0.1)
        humidity    = st.slider("Humidity (%)",           0.0, 100.0, 82.0, 0.1)
        rainfall    = st.number_input("Rainfall (mm)",    0.0, 400.0, 202.9, 0.1)
    submitted = st.form_submit_button("🌾 Recommend Crop", use_container_width=True)

if submitted:
    try:
        # Inline validation (mirrors Cell 9 contract)
        checks = {
            "N":           (N,           0,  200),
            "P":           (P,           0,  200),
            "K":           (K,           0,  250),
            "temperature": (temperature, -10, 60),
            "humidity":    (humidity,    0,  100),
            "ph":          (ph,          0,  14),
            "rainfall":    (rainfall,    0,  400),
        }
        for name, (v, lo, hi) in checks.items():
            if not isfinite(v) or not (lo <= v <= hi):
                st.error(f"Invalid {name}={v} — must be in [{lo}, {hi}]")
                st.stop()

        row   = pd.DataFrame([{"N": N, "P": P, "K": K,
                               "temperature": temperature, "humidity": humidity,
                               "ph": ph, "rainfall": rainfall}])[FEATURES]
        proba = model.predict_proba(row)[0]
        idx   = int(proba.argmax())
        crop  = le.inverse_transform([idx])[0]
        conf  = float(proba[idx])

        # Low-confidence advisory
        if conf < 0.60:
            st.warning(f"⚠️ Low confidence ({conf:.0%}). "
                       f"Consider verifying with a local soil test before acting.")

        st.success(f"### Recommended: **{crop.title()}**")
        st.metric("Confidence", f"{conf:.1%}")
        st.progress(min(conf, 1.0))

        top = proba.argsort()[::-1][:5]
        st.subheader("Top 5 alternatives")
        st.dataframe(pd.DataFrame({"Crop": le.inverse_transform(top),
                                   "Probability": proba[top]})
                       .style.format({"Probability": "{:.2%}"}),
                     use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction failed: {type(e).__name__}: {e}")
        st.stop()

st.divider()
st.warning("**Responsible AI Notice** — Advisory only. "
           "Not a replacement for local soil tests or agronomist review.")