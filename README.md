# 🧠 Customer Churn Prediction System

**End-to-end ML system with deployed FastAPI inference pipeline and SHAP-based interpretability.**

Built with a strong focus on feature engineering, model performance, and real-world business impact.

---

## 🚀 Project Overview

Customer churn is a critical business problem. This project focuses on:

- Predicting which customers are likely to churn  
- Understanding *why* they churn  
- Providing actionable insights  

---

## 🏗️ Architecture

```
Raw Data → Data Cleaning → Feature Engineering → Model Training → Evaluation → SHAP Explainability → FastAPI Deployment
```


---

## 📊 Dataset

- Telco Customer Churn dataset  
- ~7,000 customers  
- Features include:
  - Demographics  
  - Services used  
  - Billing & contract information  

---

## ⚙️ Feature Engineering

Key engineered features:

- **Tenure groups** (short / mid / long-term)
- **Service engagement score** (number of active services)
- **Contract signals** (monthly vs long-term)
- **Payment behavior** (electronic check indicator)
- **Support indicators** (tech support, security, backup)

---

## 🤖 Models Used

| Model | Purpose |
|------|--------|
| Logistic Regression | Baseline |
| Random Forest | Capture non-linear patterns |
| XGBoost | Final model |

---

## 📈 Results

### XGBoost Performance:

- Accuracy: **77.6%**
- Precision (churn): **59%**
- Recall (churn): **53%**

> Focused on identifying churners rather than maximizing accuracy.

---

## 🔍 Model Interpretability (SHAP)

Used SHAP to understand feature impact.

### Key Drivers of Churn:

- 📈 Month-to-month contracts → **highest churn risk**
- 📈 High monthly charges  
- 📈 Fiber optic internet users  
- 📉 Long-term contracts (2-year) reduce churn  
- 📉 High tenure reduces churn  
- 📉 Support services reduce churn  

---

## 💡 Business Insights

- Customers on flexible contracts are high-risk  
- Early-tenure users are most likely to churn  
- Support services improve retention  
- Pricing plays a significant role in churn behavior  

---

## 🌐 API Deployment

The model is deployed using **FastAPI** for real-time predictions.

### Run Locally

```bash
python -m uvicorn src.api.main:app --reload
```

### API Endpoint

```
POST /predict
```

### Example Request

```
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 845.5
}
```

### Example Response

```
{
  "churn_prediction": 1,
  "churn_probability": 0.78
}
```

## 🛠️ Tech Stack
- ML/DL: Scikit-learn, XGBoost
- Data: Pandas, NumPy
- Explainability: SHAP
- Deployment: FastAPI
- Tools: Joblib, Matplotlib

## 🎯 Key Takeaways
- Feature engineering significantly improved model performance
- Interpretability is critical for real-world ML systems
- Business understanding matters more than raw accuracy
- Deployment transforms ML models into usable systems
