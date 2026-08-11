"""
Training pipeline for the churn prediction model.

Loads the raw Telco Customer Churn CSV, applies the feature engineering
in src/features.py, cross-validates several classifiers with
hyperparameter search, evaluates the best-of-each on a held-out test
set, selects the overall winner by ROC-AUC (a good fit here since the
classes are imbalanced ~73/27), and persists the winning pipeline
(preprocessing + model bundled together) to models/model.joblib.

Run with:
    python src/train.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import get_feature_lists, prepare_training_frame  # noqa: E402

DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
RANDOM_STATE = 42


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                categorical_features,
            ),
        ]
    )


def get_model_grid(preprocessor: ColumnTransformer) -> dict:
    """Returns {name: (pipeline, param_grid)} for each candidate model."""
    return {
        "LogisticRegression": (
            Pipeline(
                [
                    ("prep", preprocessor),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            Pipeline(
                [
                    ("prep", preprocessor),
                    (
                        "clf",
                        RandomForestClassifier(
                            random_state=RANDOM_STATE, class_weight="balanced"
                        ),
                    ),
                ]
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [6, 10, None],
            },
        ),
        "GradientBoosting": (
            Pipeline(
                [
                    ("prep", preprocessor),
                    ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
        ),
    }


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def main() -> None:
    print(f"Loading data from {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    print(f"Raw shape: {raw.shape}")

    X, y = prepare_training_frame(raw)
    numeric_features, categorical_features = get_feature_lists(X)
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Churn rate: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model_grid = get_model_grid(preprocessor)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    fitted_pipelines = {}

    for name, (pipeline, param_grid) in model_grid.items():
        print(f"\n=== {name}: 5-fold CV grid search (scoring=roc_auc) ===")
        t0 = time.time()
        search = GridSearchCV(
            pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        test_metrics = evaluate(search.best_estimator_, X_test, y_test)
        results[name] = {
            "best_params": search.best_params_,
            "cv_best_roc_auc": round(search.best_score_, 4),
            "test_metrics": test_metrics,
            "train_seconds": round(elapsed, 2),
        }
        fitted_pipelines[name] = search.best_estimator_

        print(f"Best params: {search.best_params_}")
        print(f"CV best ROC-AUC: {search.best_score_:.4f}")
        print(f"Test metrics: {test_metrics}")
        print(f"Fit time: {elapsed:.2f}s")

    best_name = max(results, key=lambda n: results[n]["test_metrics"]["roc_auc"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\n=== Best model: {best_name} (test ROC-AUC={results[best_name]['test_metrics']['roc_auc']}) ===")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best_name,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_columns": list(X.columns),
        },
        MODEL_PATH,
    )
    print(f"Saved best model pipeline to {MODEL_PATH}")

    report = {
        "best_model": best_name,
        "results": results,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved metrics report to {METRICS_PATH}")

    print("\n=== Summary table ===")
    header = f"{'Model':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}{'ROC-AUC':<12}"
    print(header)
    for name, res in results.items():
        m = res["test_metrics"]
        print(
            f"{name:<20}{m['accuracy']:<12}{m['precision']:<12}{m['recall']:<12}{m['f1']:<12}{m['roc_auc']:<12}"
        )


if __name__ == "__main__":
    main()
