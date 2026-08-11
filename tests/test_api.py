"""
Tests for the FastAPI serving layer (src/api.py) using FastAPI's TestClient.

Rather than depending on the full training run / real model.joblib, this
trains a tiny throwaway pipeline on a handful of synthetic rows built with
the exact same schema as the real Telco dataset, and points the API module
at that temporary model file via monkeypatching.
"""
from __future__ import annotations

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import get_feature_lists, prepare_training_frame

RAW_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def _make_tiny_raw_dataset(n: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n):
        churn = "Yes" if i % 3 == 0 else "No"
        rows.append(
            {
                "customerID": f"{i:04d}-TEST",
                "gender": "Female" if i % 2 == 0 else "Male",
                "SeniorCitizen": i % 2,
                "Partner": "Yes" if i % 2 == 0 else "No",
                "Dependents": "No",
                "tenure": (i % 60) + 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic" if i % 2 == 0 else "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes" if i % 4 == 0 else "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes" if i % 3 == 0 else "No",
                "StreamingMovies": "No",
                "Contract": ["Month-to-month", "One year", "Two year"][i % 3],
                "PaperlessBilling": "Yes" if i % 2 == 0 else "No",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 20.0 + i,
                "TotalCharges": str((20.0 + i) * ((i % 60) + 1)),
                "Churn": churn,
            }
        )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


@pytest.fixture
def tiny_model_bundle(tmp_path):
    """Train a tiny real (not mocked) sklearn pipeline and save it as a
    joblib bundle in the same format src/train.py produces, so the API
    tests exercise real prediction logic end-to-end.
    """
    raw = _make_tiny_raw_dataset()
    X, y = prepare_training_frame(raw)
    numeric_features, categorical_features = get_feature_lists(X)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    pipeline = Pipeline(
        [("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(X, y)

    bundle = {
        "pipeline": pipeline,
        "model_name": "LogisticRegression-test",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": list(X.columns) + ["Churn"],
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(bundle, model_path)
    return model_path


@pytest.fixture
def client(tiny_model_bundle, monkeypatch):
    import src.api as api_module

    monkeypatch.setattr(api_module, "MODEL_PATH", tiny_model_bundle)
    monkeypatch.setattr(api_module, "_model_bundle", None)
    return TestClient(api_module.app)


SAMPLE_PAYLOAD = {
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
    "TotalCharges": 845.5,
}


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "LogisticRegression-test"


def test_predict_returns_valid_probability(client):
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in {"Yes", "No"}
    assert body["model_name"] == "LogisticRegression-test"


def test_predict_label_consistent_with_probability_threshold(client):
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    body = resp.json()
    expected_label = "Yes" if body["churn_probability"] >= 0.5 else "No"
    assert body["churn_prediction"] == expected_label


def test_predict_rejects_malformed_payload(client):
    bad_payload = dict(SAMPLE_PAYLOAD)
    del bad_payload["tenure"]  # required field missing
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_health_reports_no_model_when_missing(tmp_path, monkeypatch):
    import src.api as api_module

    monkeypatch.setattr(api_module, "MODEL_PATH", tmp_path / "does_not_exist.joblib")
    monkeypatch.setattr(api_module, "_model_bundle", None)
    local_client = TestClient(api_module.app)

    resp = local_client.get("/health")
    body = resp.json()
    assert body["model_loaded"] is False


def test_predict_returns_503_when_model_missing(tmp_path, monkeypatch):
    import src.api as api_module

    monkeypatch.setattr(api_module, "MODEL_PATH", tmp_path / "does_not_exist.joblib")
    monkeypatch.setattr(api_module, "_model_bundle", None)
    local_client = TestClient(api_module.app)

    resp = local_client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 503
