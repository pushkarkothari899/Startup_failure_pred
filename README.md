# 🚀 Startup Failure Risk Predictor

A machine learning web application that predicts the probability of a startup failing, built on 42,000+ companies from the Crunchbase dataset.

🔗 **[Live Demo → startup-failure-predictor.streamlit.app](https://startup-failure-predictor.streamlit.app)**

---

## 📌 What it does

- Takes startup details as input (founding year, funding, market, country)
- Returns a **failure probability score** (0–100%)
- Explains **why** using SHAP values — which factors drove the prediction
- Displays top 15 features that historically predict startup failure

---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Model | XGBoost |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Web App | Streamlit |
| Dataset | Crunchbase via Kaggle |

---

## 📊 Model Performance

- **Algorithm:** XGBoost with `scale_pos_weight` + threshold tuning
- **Accuracy:** 90% on held-out test set (8,454 samples)
- **Macro F1:** 0.57 — balances performance across both classes
- **Dataset:** 42,000+ startups, binary classification (failed vs not failed)
- **Challenge:** Severe class imbalance (19:1) handled via SMOTE + scale_pos_weight + threshold tuning

---

## 🔍 Key Findings

- `market_unknown` was the strongest predictor — startups with no identifiable market show highest risk
- `founded_year` ranks #2 — timing correlates strongly with failure patterns
- `funding_duration_days` (engineered feature) ranked in **top 5** — how long a startup kept raising money is a strong failure signal
- `funding_per_round` and `startup_age_years` added as derived features to improve signal quality
- **Bias note:** `market_unknown` shows survivorship bias — successful companies often had incomplete market tags. In a production system this feature would be dropped.

---

## 🧠 How it works
```
Raw Crunchbase Data (42,000+ startups)
↓
Data Cleaning (missing values, encoding)
↓
Feature Engineering (funding_duration_days, funding_per_round, startup_age_years)
↓
One-Hot Encoding (870+ features: market, country)
↓
SMOTE (balance failed vs not-failed)
↓
XGBoost Classifier + Threshold Tuning
↓
SHAP Explainer
↓
Streamlit Web App
```
---

## 🚀 Run Locally

```bash
git clone https://github.com/pushkarkothari899/Startup_failure_pred.git
cd Startup_failure_pred
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Pushkar Kothari**  
CS Undergrad @ WIT Solapur | ML Engineer in progress  
[LinkedIn](https://linkedin.com/in/pushkar-kothari) · [GitHub](https://github.com/pushkarkothari899)