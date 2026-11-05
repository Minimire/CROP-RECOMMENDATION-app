
# 🌱 Intelligent Crop Recommendation System

A Streamlit web app that recommends the optimal crop to plant given soil nutrients (N, P, K) and climate conditions (temperature, humidity, pH, rainfall). Built as the capstone for **FUSE AI-201 — Group 11**.

**Live app:** https://<your-subdomain>.streamlit.app

---

## 📊 Model Summary

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier (200 trees, `max_features="sqrt"`) |
| Test accuracy | **99.55%** (held-out 20% split) |
| Multi-seed CV | 99.47% ± 0.11% across 3 seeds |
| Classes | 22 crops |
| Features | 7 (`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`) |
| Training samples | 1,760 (stratified split from 2,200) |

**Why Random Forest?** Beat XGBoost, KNN, Logistic Regression, and SVM on the same split. Tuned via GridSearchCV across 3 seeds. XGBoost was also tuned via Optuna but lost by ~0.2 pp on the held-out set.

---

## 📂 Repository Contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI + inference code |
| `crop_recommendation_model.pkl` | Deployment bundle (see below) |
| `requirements.txt` | Pinned dependencies for Streamlit Cloud |
| `README.md` | This file |
| `.gitignore` | Python + Jupyter + Streamlit ignore rules |

**What's inside `crop_recommendation_model.pkl`:** a Python dict containing
`model`, `label_encoder`, `scaler`, `features`, `classes`, `n_classes`, `test_accuracy`, and `sklearn_version`. This single file is the entire "model" for deployment — nothing else is needed at runtime.

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py