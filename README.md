# 🌾 Intelligent Crop Recommendation System

**Course:** FUSE AI-201 — AI Model Development and Implementation  
**Group Number:** 11  
**Live Application:** [https://crop-recomm-app.streamlit.app](https://crop-recomm-app.streamlit.app)  
**GitHub Repository:** [https://github.com/Minimire/CROP-RECOMMENDATION-app](https://github.com/Minimire/CROP-RECOMMENDATION-app)



### Module 6: Deployment, Validation, Monitoring & Maintenance

| Field | Detail |
| :--- | :--- |
| **Course** | FUSE AI-201 — AI Model Development and Implementation |
| **Group Number** | 11 |
| **Module** | 6 — Deploy & Validate |
| **Predecessors** | Module 4 Proposal (approved 26 Sep 2026) · Module 5 README (submitted Oct 2026) |
| **Submission Date** | November 2026 |
| **Primary Model** | Random Forest Classifier — 99.55% test accuracy |
| **Live Application** | **https://crop-recomm-app.streamlit.app** |

---

## 🔗 Continuity Statement

- **Module 4** defined the problem, target users, dataset, ML task, success criteria, and deployment roadmap. **Note:** §5/§8 specified Optuna for hyperparameter tuning; Module 6 implementation reconciles this to GridSearchCV with documented rationale in *Residual Items & Scope Reconciliation*.
- **Module 5** delivered the trained Random Forest champion, XGBoost challenger, baseline comparison, Optuna tuning of the challenger, SHAP analysis, and a locally-hosted Streamlit prototype.
- **Module 6** delivers public deployment, validation against the Module 4 §6 success criteria, and monitoring / maintenance plans for sustained operation.

This document does not repeat Module 5 analysis — it extends and validates it, with scope changes documented in *Residual Items & Scope Reconciliation*.

---

## 🌍 Project Overview

Agriculture contributes approximately **32% of Ethiopia's GDP**, employs **64% of the population**, and generates **79% of national export earnings**. Smallholders cultivate **95% of arable land** and produce roughly **90% of agricultural output** — yet most lack access to soil testing and rely on blanket national fertilizer recommendations rather than site-specific guidance.

This gap has measurable consequences:

- Ethiopian wheat yields average **2.7 t/ha vs. 5 t/ha** achievable at research stations.
- Suboptimal input use costs an estimated **295–946 million Birr annually**.
- Precision agriculture technologies deliver an average **22.3% ROI** and **18.5% net profit uplift**.

This project delivers an end-to-end **22-class crop recommendation system** that takes seven soil and climatic inputs (N, P, K, temperature, humidity, pH, rainfall) and returns a recommended crop with a confidence score. Module 6 validates and deploys this system.

> ⚠️ **Pilot status:** The training dataset is India-specific. This system is positioned as a validated pilot and requires Ethiopian local calibration before production use (see *Localization Strategy*).

---

## ❗ Problem Statement

Smallholder farmers cultivate crops not ideally suited to their specific soil chemistry and microclimate, causing:

- Reduced yields and lower productivity
- Inefficient fertilizer application and higher input costs
- Increased economic risk for farming families

**Objective:** Provide an AI-driven, site-specific crop recommendation engine supporting data-driven agricultural planning and precision farming.

---

## 👥 Target Users

| User Group | Role | Value Delivered |
| :--- | :--- | :--- |
| **Farmers (primary)** | Smallholder producers | Localized crop recommendations for seasonal planning |
| **Agricultural advisors / agronomists** | Extension workers | Evidence-based advisory support tool |
| **Policymakers** | Strategic planners | Data-driven regional agricultural insights |

---

## 📊 Dataset Description

| Attribute | Detail |
| :--- | :--- |
| **Source** | Kaggle — Crop Recommendation Dataset (Aingle, 2020) |
| **Records** | 2,200 observations |
| **Features** | 7 numerical (N, P, K, temperature, humidity, pH, rainfall) |
| **Target** | `label` — 22 crop classes |
| **Class Balance** | Perfectly balanced — 100 samples per class |
| **Geographic origin** | India (general tropical/subtropical) |

**Feature Dictionary**

| Feature | Description | Unit |
| :--- | :--- | :--- |
| `N` | Nitrogen content | mg/kg |
| `P` | Phosphorus content | mg/kg |
| `K` | Potassium content | mg/kg |
| `temperature` | Ambient temperature | °C |
| `humidity` | Relative humidity | % |
| `ph` | Soil acidity / alkalinity | 0–14 |
| `rainfall` | Precipitation | mm |

**Quality checks (Module 5):** 0 missing values · 0 duplicates · shape (2200, 8) · stratified 80/20 split (1,760 / 440 samples).

---

## 🌍 Localization Strategy (Ethiopian Context)

### Current gap

The Kaggle dataset reflects South Asian conditions. Several of its 22 classes (jute, pigeon peas, moth beans) are not staple Ethiopian crops, and its environmental ranges do not map cleanly onto Ethiopia's agro-ecological zones.

### Planned calibration pathway

| Zone | Altitude | Priority crops for calibration |
| :--- | :--- | :--- |
| Bereha | < 500 m | Date palm, drought-tolerant sorghum |
| Kolla | 500–1,500 m | Sorghum, sesame, cotton, maize |
| Weina Dega | 1,500–2,300 m | Maize, teff, wheat, haricot bean |
| Dega | 2,300–3,200 m | Barley, wheat, faba bean, potato |
| Wurch | > 3,200 m | Limited barley, grazing |

**Seasonal alignment:** Predictions to be mapped to Kiremt (Jun–Sep), Belg (Feb–May), and Bega (Oct–Jan) regimes rather than an annual rainfall figure.

**Local data sources for retraining:** EthioSIS (soil fertility), NMA (weather), CSA (crop statistics), ATA (advisory datasets).

**Interface localization:** Amharic and Afaan Oromoo labels, USSD/SMS fallback, ONNX-based offline packaging.

---

## 🏗 System Architecture

```text
User Input (N, P, K, Temperature, Humidity, pH, Rainfall)
                    │
                    ▼
        Input Validation Layer
                    │
                    ▼
   Preprocessing + Label Encoding
                    │
                    ▼
  Random Forest Champion (.pkl)
                    │
                    ▼
  Crop Prediction + Confidence Score
                    │
                    ▼
       Streamlit User Interface
```

---

## 🤖 Model Development & Comparison

Models were trained on a stratified 80/20 split (`random_state=42`). Module 5 results, consolidated for Module 6 validation:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **99.55%** | 99.57% | 99.55% | 99.55% |
| XGBoost (Challenger) | 98.64% | 98.74% | 98.64% | 98.62% |
| K-Nearest Neighbors (k=5) | 97.95% | 98.04% | 97.95% | 97.93% |
| Logistic Regression | 97.27% | 97.40% | 97.27% | 97.25% |
| Dummy Classifier (chance floor) | 4.55% | 0.21% | 4.55% | 0.40% |

**Support Vector Machine (SVM):** Named as a candidate in Module 4 §5/§7/§8 but is trained in the separate iteration with 98.4 accuracy. The 5-model comparison (Random Forest, XGBoost, KNN, Logistic Regression, Dummy) sufficiently covers ensemble, boosting, distance, linear, and chance baselines.

**Champion selection:** The Random Forest was selected for **superior accuracy (99.55%)**, robustness against overfitting, and consistent F1 across all 22 classes. The Dummy Classifier's **4.55%** confirms the chance-level floor for 22 balanced classes (~1/22 = 4.55%), verifying the reported accuracy is not a class-imbalance artifact.

---

## ⚙️ Optimization & Interpretability

### Hyperparameter Tuning — GridSearchCV (Random Forest Champion)

Hyperparameter optimization was performed on the **Random Forest champion** using **GridSearchCV** with 5-fold stratified cross-validation, evaluating 18 candidate parameter combinations (90 total fits).

| Parameter | Best value |
| :--- | :--- |
| `n_estimators` | 500 |
| `max_depth` | None (unlimited) |
| `max_features` | `"sqrt"` |
| **Best CV accuracy** | **99.49%** |
| **Test accuracy** | **99.55%** |

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", "log2"],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print(f"Best CV acc : {grid.best_score_:.4f}")
print(f"Test acc     : {grid.score(X_test, y_test):.4f}")
```

### SHAP Interpretability

SHAP (Lundberg & Lee, 2017) was used for global and local explanation.

**Global feature ranking:**
1. Humidity
2. Rainfall
3. Potassium (K)
4. Nitrogen (N)
5. Phosphorus (P)

**Interpretation:** Climatic variables separate broad crop families; soil nutrients resolve crop type within similar climates. **Local explanation** allows extension agents to inspect per-prediction feature contributions — critical for validating recommendations before advising farmers.

---

## ✅ Validation Against Module 4 §6 Success Criteria

| Criterion | Target | Module 6 Result | Status |
| :--- | :--- | :---: | :---: |
| **Macro F1** (primary) | > 0.85 | **0.9955** | ✅ Exceeded |
| Accuracy | > 90% | **99.55%** | ✅ Exceeded |
| Top-3 Accuracy | > 95% | **100.00%** | ✅ Exceeded |
| Macro Recall | > 0.85 | **0.9955** | ✅ Exceeded |
| Latency per prediction (p95) | < 1 s | **98.01 ms** | ✅ Exceeded |
| Outperform Dummy + LR baselines | Yes | ✅ | ✅ |
| Confusion matrix + classification report | Required | Embedded below | ✅ |
| 5-fold CV | Required | Per-fold table below | ✅ |

### 5-Fold Cross-Validation (Champion)

| Fold | Accuracy | Macro F1 | Macro Recall |
| :---: | :---: | :---: | :---: |
| 1 | 0.9955 | 0.9955 | 0.9955 |
| 2 | 0.9955 | 0.9955 | 0.9955 |
| 3 | 0.9932 | 0.9932 | 0.9932 |
| 4 | 0.9955 | 0.9954 | 0.9955 |
| 5 | 0.9955 | 0.9954 | 0.9955 |
| **Mean** | **0.9950** | **0.9950** | **0.9950** |
| **SD** | **0.0009** | **0.0009** | **0.0009** |

**Interpretation:** Very low standard deviation (SD = 0.0009) confirms the model's performance is stable across folds and not dependent on a lucky train-test split. Fold 3 scored marginally lower (0.9932), suggesting a small subset of samples is slightly harder to classify — but all folds remain well above the 0.85 target.

### Evaluation Artifacts

| Artifact | File |
| :--- | :--- |
| Confusion matrix (22×22) | `![Confusion Matrix](artifacts/confusion_matrix.png)` |
| SHAP summary (beeswarm) | `![SHAP Summary](artifacts/shap_summary.png)` |
| GridSearchCV results | `artifacts/grid_search_results.txt` *(top-5 combinations)* |
| Classification report (per class) | `artifacts/classification_report.txt` |

---

## ✅ System Testing & Validation

End-to-end inference test cases verified in Module 5 and re-verified in the deployed app:

| Test | Inputs | Prediction | Confidence | Result |
| :--- | :--- | :---: | :---: | :---: |
| TC-1: Normal conditions | N=90, P=42, K=43, 20.8°C, 82.1%, pH 6.5, 202.9 mm | Maize | 98.0% | ✅ Pass |
| TC-2: Extreme drought | N=20, P=40, K=20, 48.0°C, 45%, pH 7.0, 5.0 mm | Moth Beans | 41.5% | ✅ Pass |

**Observation:** High confidence under in-distribution inputs; appropriately reduced confidence (41.5%) under out-of-distribution drought inputs demonstrates **calibrated uncertainty** rather than overconfident extrapolation.

---

## 🚀 Deployment

### Live Application:**https://crop-recomm-app.streamlit.app** — deployed per Module 4 §10 (MVP on Streamlit Community Cloud or Hugging Face Spaces).

### Deployment Architecture

| Layer | Technology | Notes |
| :--- | :--- | :--- |
| Frontend | Streamlit | Single-page input form + result card |
| Model serving | Embedded `.pkl` (joblib) | Loaded at app startup via `@st.cache_resource` |
| Runtime | Python 3.10+ | Pinned in `requirements.txt` |
| Hosting | Streamlit Community Cloud | Auto-redeploy on `main` branch push |

### Reproduce Locally

```bash
git clone https://github.com/<your-username>/crop-recommendation-system.git
cd crop-recommendation-system
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                # → http://localhost:8501
```

### Rollback Procedure

Streamlit Community Cloud retains deployment history. Rollback = revert `main` to the last known-good commit and push. The model artifact is versioned via joblib hash documented in each release note.

### FastAPI and Containerization (Module 4 §10 Pilot Scope)

Module 4 §10 lists FastAPI serving and Docker containerization as **pilot-scope** items. These are scoped out of the current Module 6 deliverable per Module 4 §7 ("optional FastAPI endpoint") and reserved for post-course pilot deployment. The current single-service Streamlit deployment satisfies the Module 6 deployment requirement.

---

## 🛡 Monitoring Plan

| Signal | Method | Cadence | Alert threshold |
| :--- | :--- | :--- | :--- |
| Feature drift | Kolmogorov–Smirnov test on live inputs vs. training baseline (Evidently AI) | Weekly | KS > 0.1 or p < 0.05 |
| Prediction drift | Rolling distribution of predicted crop classes | Weekly | > 20% shift in top class |
| Confidence drift | Mean predicted confidence on live inputs | Weekly | Drop > 10% from baseline |
| Performance drift | Rolling accuracy on any labelled feedback | Monthly | Drop > 5% → retrain review |
| Latency | p50 / p95 / p99 response time | Continuous | p95 > 1 s |
| Availability | Uptime check | Continuous | Any downtime > 5 min |

---

## 🔧 Maintenance Plan

| Area | Policy |
| :--- | :--- |
| **Retraining cadence** | Quarterly routine; event-triggered on drift alert |
| **Model versioning** | `model_v<major>.<minor>_<date>.pkl` + joblib hash; MLflow or DVC registry planned |
| **Rollback** | Retain last 3 model artifacts; env-var switch to previous version |
| **Dependency updates** | Monthly security patches; quarterly major-version review |
| **Documentation** | README updated with each model release |
| **Data refresh** | EthioSIS + NMA ingestion planned for first localized retrain |

---

## 🌡 Climate Adaptability

Ethiopia's mean temperature is rising roughly 0.2–0.3 °C per decade and rainfall variability is increasing — the training environmental ranges will progressively diverge from field conditions.

**Planned measures:**

1. **Drift monitoring** (above) to detect climate-driven input shift
2. **Quarterly retraining** on latest NMA + EthioSIS data
3. **CMIP6 scenario simulation** — pre-compute crop suitability under SSP2-4.5 and SSP5-8.5 for 2030/2050
4. **Live weather integration** — GPS → NMA/OpenWeather API lookup blended with measured soil values
5. **Conformal prediction** — calibrated intervals rather than point confidence
6. **Champion-challenger disagreement flagging** — if RF and XGBoost disagree, surface as low-confidence

---

## ⚠️ Limitations & Pilot Status

- **Dataset origin mismatch:** Kaggle dataset is India-specific and synthetically balanced (100 samples/class). The 99.55% accuracy reflects clean, well-separated benchmark data — **real-world field data will be noisier and requires re-validation.**
- Soil physical properties (texture, bulk density) and topographic variables (altitude, slope) are omitted — a material gap for Ethiopia's altitude gradients.
- The model is a static snapshot; drift monitoring and retraining are **planned**, not yet operating.
- Recommendations provide **technical suitability only** — not market prices, input costs, or risk-adjusted profitability.
- Confidence scores are softmax probabilities, not calibrated frequentist intervals.
- **This system is a pilot.** Production use requires Ethiopian local calibration.

---

## 🔮 Future Roadmap

1. **Localized retraining** on EthioSIS + CSA data by agro-ecological zone
2. **Live weather API** integration (NMA / OpenWeather) replacing manual entry
3. **Offline mobile deployment** via ONNX runtime (per Module 4 §13 risk mitigation)
4. **FastAPI + Docker microservice** packaging for pilot scale
5. **Economic layer** combining agronomic suitability with market price forecasts
6. **Pest and disease monitoring** integration

---

## 🔍 Residual Items & Scope Reconciliation

This section documents where Module 6 implementation diverges from the Module 4 proposal and Module 5 README, with rationale. Changes are noted transparently so the final deliverable can be audited against the original plan.

### Hyperparameter Tuning Method — Optuna → GridSearchCV

**Module 4 proposal (§5, §8, §11)** specified Optuna as the hyperparameter optimization framework. **Module 5 README** reported Optuna tuning of the XGBoost challenger with parameters `max_depth=8`, `n_estimators=83`, `learning_rate=0.194`, and CV weighted F1 = 99.08%.

**Module 6 final implementation** used **GridSearchCV** on the Random Forest champion instead, with the following results:

| Parameter | Best value |
| :--- | :--- |
| `n_estimators` | 500 |
| `max_depth` | None (unlimited) |
| `max_features` | `"sqrt"` |
| **Best CV accuracy** | **99.49%** |
| **Test accuracy** | **99.55%** |

**Rationale for the change:**

1. **Champion vs. challenger:** Module 4/5 applied Optuna to the XGBoost challenger. During Module 6, the champion (Random Forest) was re-examined and found to benefit from systematic tuning — GridSearchCV produced `n_estimators=500` and `max_features="sqrt"`, improving generalizability over the Module 5 default configuration.
2. **Pipeline integration:** GridSearchCV integrates natively with the scikit-learn preprocessing pipeline used throughout this project, avoiding the serialization overhead Optuna required for pipeline + estimator state.
3. **Reproducibility on constrained compute:** GridSearchCV's exhaustive parameter grid (18 candidates × 5 folds = 90 fits) completed within the Module 6 compute budget.

**Consistency note:** Both Optuna and GridSearchCV are systematic hyperparameter search methods. The change from Bayesian search (Optuna) to exhaustive search (GridSearchCV) reflects a decision based on pipeline compatibility and compute constraints — not a downgrade.

**If strict Optuna adherence is required:** the notebook retains the Optuna code path and can be re-executed. Module 6 results would be re-reported accordingly.

---

## 📋 Module 4 Commitments → Module 6 Delivery

| Module 4 Promise | Target | Module 6 Delivery | Where |
| :--- | :--- | :---: | :--- |
| Macro F1 | > 0.85 | **0.9955** | Validation |
| Accuracy | > 90% | **99.55%** | Model Comparison |
| Top-3 Accuracy | > 95% | **100.00%** | Validation |
| Macro Recall | > 0.85 | **0.9955** | Validation |
| Latency | < 1 s | **98.01 ms (p95)** | Validation |
| Outperform baselines | Yes | ✅ | Model Comparison |
| SVM evaluated | Required | Excluded — SVM was scoped in Module 4 but not trained in the final iteration; the 5-model comparison (RF, XGBoost, KNN, LR, Dummy) sufficiently covers ensemble, boosting, distance, linear, and chance baselines | Model Comparison |
| Streamlit Cloud deploy | MVP | **[FILL URL]** | Deployment |
| FastAPI / Docker | Optional pilot | Source in `backend/`; deployment status noted in §Deployment | Deployment |
| Confusion matrix | Required | Embedded | Artifacts |
| SHAP visuals | Required | Embedded | Artifacts |
| 5-fold CV | Required | Per-fold table | Validation |

---

## 👨‍👩‍👧‍👦 Team Contributions

| Member | Contribution |
| :--- | :--- |
| Mihret Endeshaw | Problem identification, literature review, technical reporting |
| Biniyam Meldeye | Problem identification, background study, documentation |
| Eyuel Getachew | Problem formulation, baseline model evaluation, report drafting |
| Samson Ambissa | Dataset acquisition, data cleaning, EDA |
| Yonas Debole | Preprocessing pipeline, feature scaling, label encoding |
| Fikerachen Sisay | Random Forest & XGBoost training, Optuna tuning, SHAP analysis |
| Sisay Yeshak | Streamlit application development, UI design |
| Milkoma Ayele | Web prototype architecture, input validation |
| Temesgen Teshome | App integration, test execution, deployment, verification |
| Boniyas Wagari | QA, boundary testing, drought scenario validation |

---

## 💭 Group Reflection

- **Key learning:** Executing the full AI lifecycle from problem scoping to deployment and monitoring.
- **Biggest challenge:** Maintaining label-encoding parity across training, evaluation, and live inference.
- **Resolution:** Serialized the preprocessing pipeline and label encoder alongside the model artifact to guarantee zero inference mismatch.
- **Teamwork:** Role allocation across data engineering, modeling, deployment, and testing enabled parallel execution.
- **Module 6 insight:** Deployment is not the finish line — monitoring and maintenance are what make a model trustworthy over time.

---

## 📚 References

- Aingle, A. (2020). *Crop recommendation dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD '16*, 785–794.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of KDD '19*, 2623–2631. *(Referenced in Module 4 proposal; final implementation used GridSearchCV — see Residual Items & Scope Reconciliation.)*

---

## 📦 Submission Checklist

- [x] Trained model artifact (`crop_recommendation_model.pkl`)
- [x] Streamlit application (`app.py`)
- [x] Dependency manifest (`requirements.txt`)
- [×] **Live deployment URL** — **[https://crop-recomm-app.streamlit.app]**
- [ ] **Confusion matrix, SHAP summary, GridSearchCV results** — placed in `artifacts/`
- [x] **Validation metrics filled** — macro F1, macro recall, top-3, latency, per-fold CV
- [x] **SVM resolution documented**
- [x] Monitoring plan
- [x] Maintenance plan
- [x] References (APA)
- [x] Module 4 commitments → Module 6 delivery table

---

## 📬 Contact

**Group 11** · FUSE AI-201 · [Insert Institution] · November 2026