"""
Downloads the real IBM Telco Customer Churn dataset used by this project.

The CSV itself is not committed to this repository: at ~950 KB of dense
numeric/text data, transcribing it verbatim through any LLM-mediated
code-review or patch tool risks silently altering digits, which would be
unacceptable for a dataset used to train and evaluate a model. Fetching it
directly with `requests` guarantees a byte-exact copy instead.

Run once before training:
    python data/download_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import requests

# Primary source: IBM's own public mirror of the Telco Customer Churn
# dataset (originally published for IBM Watson / Cognos churn-analysis
# tutorials; widely reused for ML education).
DATA_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    # Common secondary mirror, kept as a fallback in case the primary
    # moves or rate-limits.
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv?raw=true",
]

OUTPUT_PATH = Path(__file__).resolve().parent / "Telco-Customer-Churn.csv"


def main() -> None:
    last_error: Exception | None = None
    for url in DATA_URLS:
        try:
            print(f"Downloading dataset from {url} ...")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            OUTPUT_PATH.write_bytes(resp.content)
            print(f"Saved {len(resp.content):,} bytes to {OUTPUT_PATH}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"Failed from {url}: {exc}")
            last_error = exc
    raise RuntimeError(
        "Could not download the Telco Customer Churn dataset from any known "
        "mirror. Check your network connection, or manually place a copy at "
        f"{OUTPUT_PATH} (columns must match the IBM Telco Customer Churn schema)."
    ) from last_error


if __name__ == "__main__":
    main()
