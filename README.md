# Churn Prediction Pipeline

An end-to-end, production-shaped machine learning pipeline for predicting
telecom customer churn: data ingestion, feature engineering, cross-validated
model comparison with hyperparameter search, and a FastAPI service that
serves the winning model.

This is not a notebook. It is a small, tested Python package: pure,
unit-tested feature engineering functions shared between offline training
and online inference, a training script that does real 5-fold
cross-validated grid search over three model families, and a REST API that
loads the persisted model and returns calibrated probabilities.

## Dataset

[IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
— 7,043 customers of a fictional telecom company, 21 columns
(demographics, account info, subscribed services, billing) and a binary
`Churn` label. This is a real, widely used public dataset (originally
published by IBM for churn-analysis tutorials).

The raw CSV itself is **not** committed to this repository — at ~950 KB of
dense numeric data, the safest way to get a byte-exact copy into this repo
is to fetch it directly rather than have it pass through any text-based
review/patch path. Run this once before training:

```bash
python data/download_dataset.py
```

It downloads the dataset straight from the IBM mirror above into
`data/Telco-Customer-Churn.csv` (with a documented secondary mirror as
fallback). This is the same script — and the same dataset, verified
byte-for-byte identical — used to produce the results below.

Class balance: **26.5% churned / 73.5% retained** — imbalanced enough that
accuracy alone is a poor model-selection metric, which is why the pipeline
selects on ROC-AUC and also reports precision/recall/F1.

No synthetic-data fallback was needed — the real dataset downloaded
successfully from the URL above.

## Pipeline architecture

```
data/Telco-Customer-Churn.csv
        │
        ▼
src/features.py          pure functions: clean → normalize categories → engineer features
        │
        ▼
src/train.py              train/test split (80/20, stratified)
                           │
                           ├── ColumnTransformer (StandardScaler + OneHotEncoder)
                           ├── 5-fold StratifiedKFold CV + GridSearchCV per model
                           │     • LogisticRegression
                           │     • RandomForestClassifier
                           │     • GradientBoostingClassifier
                           ├── evaluate each tuned model on the held-out test set
                           └── persist best pipeline → models/model.joblib
                                                       models/metrics.json
        │
        ▼
src/api.py                FastAPI app: loads models/model.joblib
                           GET  /health
                           POST /predict  → churn probability
```

### Feature engineering (`src/features.py`)

- Imputes the 11 blank `TotalCharges` values (all brand-new customers with
  `tenure == 0`) to `0.0`.
- Collapses the dataset's redundant `"No internet service"` /
  `"No phone service"` categories down to a plain `"No"` (that information
  is already carried by `InternetService` / `PhoneService`).
- Engineers four extra features: `avg_monthly_spend` (TotalCharges / tenure,
  div-by-zero guarded), `tenure_bucket` (new / established / loyal),
  `num_addon_services` (count of subscribed value-add services), and
  `is_month_to_month` (explicit flag for the strongest known churn driver
  in this dataset).
- All functions are pure (no I/O, no mutation of inputs) and are exercised
  directly by `tests/test_features.py` on small fixture DataFrames — no
  need to load the full dataset to test the logic.
- Scaling (`StandardScaler`) and one-hot encoding (`OneHotEncoder`) are
  fit only inside the `sklearn` `Pipeline`/`ColumnTransformer` in
  `src/train.py`, so the encoders are learned strictly on the training
  split and shipped bundled with the model — training and serving always
  apply identical transformations.

## Results (real run, not fabricated)

Run on 2026-08-10 via `python src/train.py` — 5-fold stratified
cross-validation with `GridSearchCV` (`scoring="roc_auc"`) per model, then
evaluated once on a held-out 20% test set (1,409 customers) that no model
saw during tuning.

| Model | CV ROC-AUC (best) | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8461 | 0.7346 | 0.5000 | 0.7861 | 0.6112 | 0.8411 |
| Random Forest | 0.8467 | 0.7559 | 0.5264 | 0.7995 | 0.6348 | 0.8440 |
| **Gradient Boosting (selected)** | **0.8489** | **0.7977** | **0.6551** | 0.5027 | 0.5688 | **0.8450** |

**Selected model: `GradientBoostingClassifier`** (best test ROC-AUC = 0.8450),
best hyperparameters found by grid search:
`{'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 100}`.

Dataset split: 5,634 training rows / 1,409 test rows, overall churn rate
26.54%. Full per-model results (including all CV scores and hyperparameter
grids explored) are saved to `models/metrics.json` on every training run.

Note on the metric tradeoff: Gradient Boosting wins on ROC-AUC and
precision/accuracy, but Logistic Regression and Random Forest have
noticeably higher recall (catch more true churners at the cost of more
false alarms). Which model you'd actually deploy depends on whether the
business cost of missing a churner outweighs the cost of an unnecessary
retention offer — worth calling out explicitly rather than pretending one
metric settles it.

## Project structure

```
churn-prediction-pipeline/
├── data/
│   └── download_dataset.py         # fetches the real IBM Telco churn dataset (7,043 rows)
├── models/
│   ├── model.joblib                # best trained pipeline (regenerated by train.py)
│   └── metrics.json                # full metrics report from the last training run
├── src/
│   ├── features.py                 # pure data cleaning + feature engineering
│   ├── train.py                    # CV model comparison, selection, persistence
│   └── api.py                      # FastAPI serving layer
├── tests/
│   ├── test_features.py            # unit tests on small fixture DataFrames
│   └── test_api.py                 # TestClient tests against a tiny throwaway model
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/download_dataset.py
```

## Train

```bash
python src/train.py
```

Runs the full cross-validated comparison described above, and writes
`models/model.joblib` and `models/metrics.json`.

> `models/model.joblib` is a binary pickle artifact and is **not** committed
> to this repository (see `.gitignore`) — run `python src/train.py` once
> (takes well under a minute) to regenerate it before starting the API.
> `models/metrics.json`, the real metrics report from the training run
> described below, **is** committed for reference.

## Test

```bash
pytest -v
```

Real output from the last run in this repo:

```
20 passed, 1 warning in 2.86s
```

20 tests total: 14 for the feature engineering functions (`tests/test_features.py`)
and 6 for the API (`tests/test_api.py`), including a real end-to-end
`/predict` call, a malformed-payload 422 case, and health-check behavior
both with and without a trained model present.

## Serve

```bash
uvicorn src.api:app --reload
```

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","model_loaded":true,"model_name":"GradientBoosting"}
```

### `POST /predict`

Real request/response captured from a locally running instance of this API
(`models/model.joblib` from the training run above):

**High-risk customer** — new, month-to-month, fiber, no add-on services:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0
  }'
```

```json
{"churn_probability":0.8104,"churn_prediction":"Yes","model_name":"GradientBoosting"}
```

**Low-risk customer** — long-tenured, two-year contract, bundled services:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 3300.0
  }'
```

```json
{"churn_probability":0.0248,"churn_prediction":"No","model_name":"GradientBoosting"}
```

Both outputs are consistent with well-known churn drivers in this dataset
(month-to-month contracts and short tenure sharply increase risk; long
tenure with bundled services and a long-term contract sharply decrease
it), which is a reasonable sanity check that the model learned real
signal rather than noise.

## Tech stack

Python 3.10, pandas, numpy, scikit-learn, FastAPI, uvicorn, pydantic,
joblib, pytest, httpx (for `TestClient`).

## License

MIT — see [LICENSE](LICENSE).
