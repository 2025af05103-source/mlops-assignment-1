"""Download the UCI Heart Disease dataset and clean it into data/heart_disease_clean.csv.

Source: UCI ML Repository, Cleveland subset (303 rows, 14 columns) via a fixed
mirror so the script is deterministic. Raw file is cached in data/raw/.
"""

import io
import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "raw" / "processed.cleveland.data"
CLEAN_PATH = DATA_DIR / "heart_disease_clean.csv"

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]
# ca and thal are read as strings by pandas because they contain '?' markers
NUMERIC_FEATURES = COLUMNS[:-1]


def download_raw() -> Path:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        log.info("Raw data already cached at %s", RAW_PATH)
        return RAW_PATH
    log.info("Downloading %s", RAW_URL)
    resp = requests.get(RAW_URL, timeout=60)
    resp.raise_for_status()
    RAW_PATH.write_bytes(resp.content)
    log.info("Saved raw data to %s", RAW_PATH)
    return RAW_PATH


def clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, header=None, names=COLUMNS, na_values="?")
    n0 = len(df)
    # 6 rows have missing ca/thal -> drop (small fraction, keeps preprocessing simple)
    df = df.dropna().reset_index(drop=True)
    # Target: 0 = no disease, 1-4 = disease -> binarize to 0/1
    df["target"] = (df["target"] > 0).astype(int)
    # Ensure all features are numeric
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    # Basic sanity: cap impossible values
    df = df[(df["age"] >= 20) & (df["age"] <= 100)]
    df = df[df["chol"] > 0]
    log.info("Cleaned %d -> %d rows", n0, len(df))
    return df


def main() -> None:
    download_raw()
    df = clean()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    log.info("Wrote %s (%d rows, %d cols)", CLEAN_PATH, len(df), df.shape[1])
    log.info("Class balance:\n%s", df["target"].value_counts().to_string())


if __name__ == "__main__":
    main()
