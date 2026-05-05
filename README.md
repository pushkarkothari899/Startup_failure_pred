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

- **Algorithm:** XGBoost with `scale_pos_weight` for class imbalance
- **Class 1 (Failed) Recall:** 0.56 — catches 56% of actual startup failures
- **Dataset:** 42,000+ startups, binary classification (failed vs not failed)
- **Challenge:** Severe class imbalance (95% not failed, 5% failed) handled via SMOTE + scale_pos_weight

---

## 🔍 Key Findings

- `market_unknown` was the strongest predictor — startups with no identifiable market show highest risk
- `founded_year` matters significantly — timing correlates with failure patterns
- `funding_rounds` and funding type (angel, venture, equity) are strong signals
- **Bias note:** `market_unknown` shows survivorship bias in the dataset — successful companies often had incomplete market tags. In a production system this feature would be dropped.

---

## 🧠 How it works
Raw Crunchbase Data (42k startups)
↓
Data Cleaning (missing values, encoding)
↓
One-Hot Encoding (market, country)
↓
SMOTE (balance failed vs not-failed)
↓
XGBoost Classifier
↓
SHAP Explainer
↓
Streamlit Web App

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