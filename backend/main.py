from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
PROC_PATH = os.path.join(DATA_DIR, "processed_data.csv")

app = FastAPI(title="Student Performance Analytics API", version="1.0.0")

# Load models and metrics
try:
    lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
    best_model_name = metrics.get("best_model", "Logistic Regression")
    best_model = lr_model if "Logistic" in best_model_name else rf_model
except Exception as e:
    lr_model = rf_model = best_model = None
    metrics = {}
    best_model_name = "Not loaded"

df = pd.read_csv(PROC_PATH) if os.path.exists(PROC_PATH) else pd.DataFrame()

class PredictionRequest(BaseModel):
    gender: str
    race_ethnicity: str
    parental_education: str
    lunch: str
    test_preparation: str

@app.get("/")
def root():
    return {"message": "Student Performance Analytics API", "status": "running"}

@app.get("/api/health")
def health():
    return {"status": "healthy"}

@app.get("/api/analytics")
def analytics() -> Dict[str, Any]:
    if df.empty:
        return {"error": "No data loaded"}
    total = len(df)
    res = {
        "dataset_size": total,
        "average_math": float(df["math_score"].mean()),
        "average_reading": float(df["reading_score"].mean()),
        "average_writing": float(df["writing_score"].mean()),
        "overall_average": float(df["average_score"].mean()),
        "high_performer_count": int((df["performance"] == "High Performer").sum()),
        "normal_performer_count": int((df["performance"] == "Normal Performer").sum()),
        "high_performer_pct": float((df["performance"] == "High Performer").mean() * 100),
        "group_averages": {}
    }
    for cat in ["gender", "race_ethnicity", "parental_education", "lunch", "test_preparation"]:
        g = df.groupby(cat)["average_score"].mean().to_dict()
        res["group_averages"][cat] = {k: float(v) for k, v in g.items()}
    return res

@app.get("/api/models")
def get_models():
    return metrics

@app.post("/api/predict")
def predict(req: PredictionRequest):
    if best_model is None:
        return {"error": "Models not loaded"}
    data = pd.DataFrame([{
        "gender": req.gender,
        "race_ethnicity": req.race_ethnicity,
        "parental_education": req.parental_education,
        "lunch": req.lunch,
        "test_preparation": req.test_preparation
    }])
    pred = best_model.predict(data)[0]
    proba = None
    if hasattr(best_model, "predict_proba"):
        p = best_model.predict_proba(data)[0]
        # get P(High Performer)
        if len(best_model.classes_) == 2:
            idx_hp = list(best_model.classes_).index("High Performer") if "High Performer" in best_model.classes_ else 0
            proba = float(p[idx_hp])
    return {
        "prediction": str(pred),
        "probability": round(float(proba) if proba is not None else 0.0, 4),
        "model": best_model_name
    }
