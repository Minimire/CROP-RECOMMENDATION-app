# 🌾 Intelligent Crop Recommendation System

**Course:** FUSE AI-201 — AI Model Development and Implementation  
**Group Number:** 11  
**Live Application:** [https://crop-recomm-app.streamlit.app](https://crop-recomm-app.streamlit.app)  
**GitHub Repository:** [https://github.com/Minimire/CROP-RECOMMENDATION-app](https://github.com/Minimire/CROP-RECOMMENDATION-app)  

---

## 📌 Project Purpose & Problem Statement

Agriculture is the backbone of Ethiopia's economy, yet smallholder farmers often rely on generalized advice rather than site-specific data, leading to reduced yields and inefficient fertilizer use. 

This project delivers an end-to-end **Intelligent Crop Recommendation System**. It is a 22-class multi-class Machine Learning classification model that takes 7 soil and climatic inputs (Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall) and recommends the most suitable crop along with a confidence score.

---

## 📊 Dataset & Model Performance

* **Dataset:** Kaggle Crop Recommendation Dataset (2,200 observations, 22 crop classes, perfectly balanced).
* **Algorithm:** Random Forest Classifier (Ensemble).
* **Test Accuracy:** **99.55%**
* **Evaluation:** 5-Fold Cross-Validation (mean accuracy 99.50% ± 0.09%). SHAP analysis confirmed Humidity and Rainfall are the top predictive features.

### ⚠️ Simulated Data Statement
This project **does not use simulated or synthetic data**. It utilizes the open-source Kaggle dataset. However, because the dataset originates from India, it is used as a **proxy pilot** for the Ethiopian agricultural context. A localized retraining plan using EthioSIS data is outlined in our future roadmap.

---

## 🧪 Testing & Error Handling Evidence

The deployed application was tested against two boundary cases to verify error handling and confidence calibration:

| Test Case | Inputs | Predicted Crop | Confidence | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Conditions** | N=90, P=42, K=43, Temp=20.8°C, Hum=82.1%, pH=6.5, Rain=202.9mm | **Maize** | 98.0% | ✅ Passed |
| **Extreme Drought** | N=20, P=40, K=20, Temp=48.0°C, Hum=45%, pH=7.0, Rain=5.0mm | **Moth Beans** | 41.5% | ✅ Passed |

**Error Handling & Responsible AI:** 
* The UI enforces bounded input constraints (e.g., pH limited to 0-14) to prevent invalid entries.
* The model demonstrates **calibrated uncertainty**—it returns a lower confidence (41.5%) for out-of-distribution drought data rather than overconfident extrapolation. 
* A UI advisory explicitly states the tool is for decision-support and does not replace physical soil testing.
* **Security:** The app collects no Personally Identifiable Information (PII) and runs purely over HTTPS.

---

## 🚀 Deployment & Experiment Tracking

* **Deployment Approach:** Hosted on Streamlit Community Cloud. The app pulls from the `main` branch of GitHub and auto-redeploys on every push.
* **Runtime:** Pinned to Python 3.12 via `runtime.txt` to avoid source-build delays with `scikit-learn`.
* **Experiment Tracking:** Model training, baseline comparisons (XGBoost, KNN, Logistic Regression), and GridSearchCV tuning (`n_estimators=500`, `max_features="sqrt"`) were tracked and documented within the Jupyter/Colab Notebook environment.

---

## 🛡 Monitoring, Support & Maintenance Plan

**1. Performance Monitoring:**
* **Feature Drift:** Weekly Kolmogorov–Smirnov test against the training baseline (using Evidently AI). Alert if KS > 0.1 or p < 0.05.
* **Confidence Drift:** Track the mean predicted confidence weekly. Alert if it drops by >10% from the baseline.
* **Latency & Uptime:** Continuous p95 latency checks (Alert if > 1s) and uptime availability.

**2. Data & Model Updates:**
* **Retraining Cadence:** Quarterly scheduled retraining, or event-triggered if drift alerts occur.
* **Data Refresh:** Ingestion of EthioSIS soil data and NMA weather data for localized retraining.
* **Versioning:** Model artifacts saved as `model_v<major>.<minor>_<date>.pkl` alongside joblib hashes. Retain the last 3 versions for rollback.

**3. Support & Maintenance:**
* Dependency security patches are applied monthly. 
* Any reported defects are documented via GitHub Issues and addressed in the next patch release. 
* Rollback procedure: Revert `main` branch to the last known-good commit.

---

## ⚠️ Limitations & Future Roadmap

* **Limitations:** The current model lacks soil physical properties (texture, bulk density) and topographic variables (altitude). It is trained on Indian data and requires localized calibration for Ethiopian agro-ecological zones (Dega, Weina Dega, Kolla).
* **Roadmap:** 
  1. Retrain on localized Ethiopian data (EthioSIS).
  2. Integrate live weather APIs (OpenWeather/NMA).
  3. Migrate backend to FastAPI + Docker for scalability (source code already in `backend/`).
  4. Export model to ONNX for offline mobile/USSD access for rural farmers.

---

## ⚙️ Installation & Setup (Local)

### Prerequisites
* Python 3.9 - 3.12
* Git

### Instructions
```bash
# 1. Clone the repository
git clone https://github.com/Minimire/CROP-RECOMMENDATION-app.git
cd CROP-RECOMMENDATION-app

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
# → App will open at http://localhost:8501


## 👨‍👩‍👧‍👦 Team Contributions (Group 11)

| Member | Core Contributions |
| :--- | :--- |
| **Mihret Endeshaw** | Problem identification, literature review, technical reporting |
| **Biniyam Meldeye** | Problem identification, background study, documentation |
| **Eyuel Getachew** | Problem formulation, baseline model evaluation, report drafting |
| **Samson Ambissa** | Dataset acquisition, data cleaning, EDA |
| **Yonas Debole** | Preprocessing pipeline, feature scaling, label encoding |
| **Fikerachen Sisay** | Advanced model training (RF, XGBoost), Optuna tuning, SHAP |
| **Sisay Yeshak** | Streamlit application development, UI layout design |
| **Milkoma Ayele** | Web prototype architecture, input validation layer |
| **Temesgen Teshome** | Streamlit app integration, test execution, deployment |
| **Boniyas Wagari** | QA, boundary testing, drought scenario validation |

