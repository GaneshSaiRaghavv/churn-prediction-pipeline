"""Unit tests for src/features.py using small in-memory fixture DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_engineered_features,
    build_feature_frame,
    clean_raw_data,
    encode_target,
    get_feature_lists,
    normalize_service_columns,
    prepare_training_frame,
)


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customerID": "0001-AAAAA",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No internet service",
                "OnlineBackup": "No internet service",
                "DeviceProtection": "No internet service",
                "TechSupport": "No internet service",
                "StreamingTV": "No internet service",
                "StreamingMovies": "No internet service",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": " ",  # blank string as seen in the raw dataset
                "Churn": "No",
            },
            {
                "customerID": "0002-BBBBB",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 95.5,
                "TotalCharges": "2292.0",
                "Churn": "Yes",
            },
        ]
    )


def test_clean_raw_data_imputes_blank_total_charges(raw_sample):
    cleaned = clean_raw_data(raw_sample)
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert cleaned.loc[1, "TotalCharges"] == 2292.0
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])


def test_clean_raw_data_drops_customer_id(raw_sample):
    cleaned = clean_raw_data(raw_sample)
    assert "customerID" not in cleaned.columns


def test_clean_raw_data_does_not_mutate_input(raw_sample):
    original = raw_sample.copy(deep=True)
    _ = clean_raw_data(raw_sample)
    pd.testing.assert_frame_equal(raw_sample, original)


def test_encode_target_maps_yes_no_to_binary(raw_sample):
    encoded = encode_target(raw_sample)
    assert encoded.loc[0, "Churn"] == 0
    assert encoded.loc[1, "Churn"] == 1
    assert pd.api.types.is_integer_dtype(encoded["Churn"])


def test_encode_target_is_noop_without_target_column():
    df = pd.DataFrame({"a": [1, 2]})
    result = encode_target(df)
    pd.testing.assert_frame_equal(result, df)


def test_normalize_service_columns_collapses_service_dependent_values(raw_sample):
    normalized = normalize_service_columns(raw_sample)
    assert normalized.loc[0, "MultipleLines"] == "No"
    assert normalized.loc[0, "OnlineSecurity"] == "No"
    # Real "Yes"/"No" values on rows that do have the service are untouched.
    assert normalized.loc[1, "OnlineSecurity"] == "Yes"
    assert normalized.loc[1, "OnlineBackup"] == "No"


def test_add_engineered_features_creates_expected_columns(raw_sample):
    cleaned = clean_raw_data(raw_sample)
    engineered = add_engineered_features(cleaned)
    for col in [
        "avg_monthly_spend",
        "tenure_bucket",
        "num_addon_services",
        "is_month_to_month",
    ]:
        assert col in engineered.columns


def test_add_engineered_features_avoids_div_by_zero_at_tenure_zero(raw_sample):
    cleaned = clean_raw_data(raw_sample)
    engineered = add_engineered_features(cleaned)
    # tenure == 0 for row 0; avg_monthly_spend should be finite, not inf/NaN.
    assert np.isfinite(engineered.loc[0, "avg_monthly_spend"])


def test_add_engineered_features_tenure_bucket_values(raw_sample):
    cleaned = clean_raw_data(raw_sample)
    engineered = add_engineered_features(cleaned)
    assert engineered.loc[0, "tenure_bucket"] == "new"  # tenure 0
    assert engineered.loc[1, "tenure_bucket"] == "established"  # tenure 24


def test_add_engineered_features_is_month_to_month_flag(raw_sample):
    cleaned = clean_raw_data(raw_sample)
    engineered = add_engineered_features(cleaned)
    assert engineered.loc[0, "is_month_to_month"] == 1
    assert engineered.loc[1, "is_month_to_month"] == 0


def test_add_engineered_features_num_addon_services_counts_yes(raw_sample):
    cleaned = clean_raw_data(raw_sample)
    normalized = normalize_service_columns(cleaned)
    engineered = add_engineered_features(normalized)
    # Row 1 has OnlineSecurity, DeviceProtection, StreamingTV, StreamingMovies = Yes (4),
    # MultipleLines = Yes (5), OnlineBackup/TechSupport = No.
    assert engineered.loc[1, "num_addon_services"] == 5
    assert engineered.loc[0, "num_addon_services"] == 0


def test_build_feature_frame_full_pipeline_runs(raw_sample):
    engineered = build_feature_frame(raw_sample)
    assert "Churn" in engineered.columns  # target untouched, still Yes/No
    assert engineered.loc[0, "Churn"] == "No"
    assert "avg_monthly_spend" in engineered.columns
    assert "customerID" not in engineered.columns


def test_get_feature_lists_splits_numeric_and_categorical(raw_sample):
    engineered = build_feature_frame(raw_sample)
    engineered = encode_target(engineered)
    numeric, categorical = get_feature_lists(engineered)
    assert "tenure" in numeric
    assert "MonthlyCharges" in numeric
    assert "Contract" in categorical
    assert "Churn" not in numeric
    assert "Churn" not in categorical


def test_prepare_training_frame_returns_X_y_without_target_leak(raw_sample):
    X, y = prepare_training_frame(raw_sample)
    assert "Churn" not in X.columns
    assert list(y) == [0, 1]
    assert len(X) == 2
