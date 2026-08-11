"""
FastAPI serving layer for the churn prediction model.

Loads the trained pipeline bundle from models/model.joblib (produced by
src/train.py) and exposes:

  - GET  /health   liveness check + whether a model is loaded
  - POST /predict   churn probability for a single customer payload

Run with:
    uvicorn src.api:app --reload
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import build_feature_frame  # noqa: E402

MODEL_PATH = ROOT / "models" / "model.joblib"

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts the probability that a telecom customer will churn.",
    version="1.0.0",
)

_model_bundle: Optional[dict] = None


def load_model_bundle() -> Optional[dict]:
    global _model_bundle
    if _model_bundle is None and MODEL_PATH.exists():
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., examples=[70.35])
    TotalCharges: float = Field(..., examples=[845.5])


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: str
    model_name: str


@app.get("/health")
def health() -> dict:
    bundle = load_model_bundle()
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_name": bundle["model_name"] if bundle else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    bundle = load_model_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python src/train.py` to train and save a model first.",
        )

    raw_df = pd.DataFrame([customer.model_dump()])
    features_df = build_feature_frame(raw_df)

    # Align to the exact column set the pipeline was fitted on (minus target).
    expected_cols = [c for c in bundle["feature_columns"] if c != "Churn"]
    missing = [c for c in expected_cols if c not in features_df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing derivable features: {missing}")
    features_df = features_df[expected_cols]

    pipeline = bundle["pipeline"]
    proba = float(pipeline.predict_proba(features_df)[0, 1])
    label = "Yes" if proba >= 0.5 else "No"

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=label,
        model_name=bundle["model_name"],
    )
