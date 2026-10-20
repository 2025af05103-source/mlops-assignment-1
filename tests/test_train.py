"""Unit tests for training utilities and the saved model artifact."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import ALL_FEATURES, TARGET, build_preprocessor
from train import evaluate

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "heart_disease_clean.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "final_pipeline.joblib"


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(DATA_PATH)
    return train_test_split(
        df[ALL_FEATURES],
        df[TARGET],
        test_size=0.2,
        stratify=df[TARGET],
        random_state=42,
    )


def test_evaluate_perfect_predictions():
    y = np.array([0, 1, 0, 1])
    m = evaluate(y, y, np.array([0.1, 0.9, 0.2, 0.8]))
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0


def test_evaluate_worst_predictions():
    y = np.array([0, 1, 0, 1])
    m = evaluate(y, 1 - y, 1 - y.astype(float))
    assert m["accuracy"] == 0.0
    assert m["roc_auc"] == 0.0


def test_dataset_exists_and_clean():
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 250
    assert not df.isnull().any().any()
    assert set(df[TARGET].unique()) == {0, 1}
    assert set(df.columns[:-1]) == set(ALL_FEATURES)


def test_model_artifact_loads_and_meets_quality_threshold(data):
    X_train, X_test, y_train, y_test = data
    pipe = joblib.load(MODEL_PATH)
    proba = pipe.predict_proba(X_test)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    # Guardrail: trained model must beat a coin flip by a comfortable margin
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y_test, proba) > 0.80


def test_model_prediction_output_contract(data):
    X_train, X_test, y_train, y_test = data
    pipe = joblib.load(MODEL_PATH)
    preds = pipe.predict(X_test)
    assert set(np.unique(preds)) <= {0, 1}
