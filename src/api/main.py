from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

# Load model + preprocessor
model = joblib.load("models/xgb_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Health check
@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Preprocess
        X = preprocessor.transform(df)

        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability)
        }

    except Exception as e:
        return {"error": str(e)}