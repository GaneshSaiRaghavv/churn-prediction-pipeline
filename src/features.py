"""
Data cleaning and feature engineering for the Telco Customer Churn dataset.

All functions here are pure (no side effects, no I/O) so they can be
unit tested against small, in-memory DataFrames. They are also reused
by both the training pipeline (src/train.py) and the serving layer
(src/api.py) so that offline training and online inference apply
*exactly* the same transformations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that are Yes/No style binary flags in the raw Telco dataset.
BINARY_YES_NO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Columns that use the "No internet service" / "No phone service" pattern
# in addition to Yes/No. We normalize these to a plain 3-way category.
SERVICE_DEPENDENT_COLS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Plain multi-class categorical columns.
CATEGORICAL_COLS = [
    "gender",
    "InternetService",
    "Contract",
    "PaymentMethod",
] + SERVICE_DEPENDENT_COLS

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

ID_COLS = ["customerID"]
TARGET_COL = "Churn"


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco churn dataframe.

    Handles:
      - ``TotalCharges`` being read as a string with blank entries for
        brand-new customers (tenure == 0); these are imputed to 0.0
        since a customer with zero tenure has paid nothing total.
      - Stripping whitespace from object columns.
      - Dropping the customerID identifier column (not predictive).

    Does not mutate the input DataFrame.
    """
    out = df.copy()

    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        out["TotalCharges"] = out["TotalCharges"].fillna(0.0)

    obj_cols = out.select_dtypes(include="object").columns
    for col in obj_cols:
        out[col] = out[col].astype(str).str.strip()

    for col in ID_COLS:
        if col in out.columns:
            out = out.drop(columns=[col])

    return out


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map the ``Churn`` Yes/No column to an integer 0/1 column.

    Leaves the dataframe untouched if the target column is absent
    (useful for inference-time payloads that never carry a label).
    """
    out = df.copy()
    if TARGET_COL in out.columns and out[TARGET_COL].dtype == object:
        out[TARGET_COL] = out[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
    return out


def normalize_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse redundant categories.

    The raw dataset encodes "no internet service" / "no phone service"
    as distinct category values on several columns even though that
    information is already fully captured by ``InternetService`` /
    ``PhoneService``. We fold those into a plain "No" to avoid giving
    tree models redundant, perfectly-correlated splits and to keep the
    one-hot feature space smaller.
    """
    out = df.copy()
    for col in SERVICE_DEPENDENT_COLS:
        if col in out.columns:
            out[col] = out[col].replace(
                {"No internet service": "No", "No phone service": "No"}
            )
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a handful of derived features that are predictive of churn.

    - ``avg_monthly_spend``: TotalCharges / tenure (guards div-by-zero),
      captures whether a customer's historical spend rate differs from
      their current MonthlyCharges (e.g. after a plan change).
    - ``tenure_bucket``: coarse tenure bucket (new / established /
      loyal) — churn risk is famously nonlinear in tenure, concentrated
      in the first year.
    - ``num_addon_services``: count of subscribed value-add services,
      a simple proxy for account "stickiness".
    - ``is_month_to_month``: explicit flag for the single strongest
      known churn driver in this dataset (month-to-month contracts).
    """
    out = df.copy()

    if {"TotalCharges", "tenure"}.issubset(out.columns):
        safe_tenure = out["tenure"].replace(0, 1)
        out["avg_monthly_spend"] = out["TotalCharges"] / safe_tenure

    if "tenure" in out.columns:
        out["tenure_bucket"] = pd.cut(
            out["tenure"],
            bins=[-1, 12, 48, np.inf],
            labels=["new", "established", "loyal"],
        ).astype(str)

    addon_cols = [c for c in SERVICE_DEPENDENT_COLS if c in out.columns]
    if addon_cols:
        out["num_addon_services"] = (out[addon_cols] == "Yes").sum(axis=1)

    if "Contract" in out.columns:
        out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)

    return out


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split the columns of an already-engineered dataframe into
    (numeric_features, categorical_features), excluding the target.
    """
    exclude = {TARGET_COL}
    numeric = [
        c
        for c in df.select_dtypes(include=["int64", "float64", "int32"]).columns
        if c not in exclude
    ]
    categorical = [
        c for c in df.select_dtypes(include=["object"]).columns if c not in exclude
    ]
    return numeric, categorical


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature-engineering pipeline used both at training and at
    inference time: clean -> normalize categories -> engineer features.

    Does NOT encode the target or apply scaling/one-hot encoding —
    those are handled by the sklearn ColumnTransformer in train.py so
    that the fitted encoders/scalers are learned only on the training
    split and persisted inside the model pipeline.
    """
    out = clean_raw_data(df)
    out = normalize_service_columns(out)
    out = add_engineered_features(out)
    return out


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience wrapper for train.py: returns (X, y) ready to hand to
    a scikit-learn pipeline, with the target encoded to 0/1.
    """
    engineered = build_feature_frame(df)
    engineered = encode_target(engineered)
    y = engineered[TARGET_COL]
    X = engineered.drop(columns=[TARGET_COL])
    return X, y
