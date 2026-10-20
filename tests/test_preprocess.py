"""Unit tests for the preprocessing pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import (
    ALL_FEATURES,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET,
    build_preprocessor,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [63.0, 45.0, 70.0, 50.0],
            "sex": [1.0, 0.0, 1.0, 0.0],
            "cp": [3.0, 2.0, 1.0, 0.0],
            "trestbps": [145.0, 120.0, 160.0, np.nan],
            "chol": [233.0, 200.0, 286.0, 210.0],
            "fbs": [1.0, 0.0, 1.0, 0.0],
            "restecg": [0.0, 1.0, 0.0, 2.0],
            "thalach": [150.0, 170.0, 100.0, 140.0],
            "exang": [0.0, 1.0, 0.0, 0.0],
            "oldpeak": [2.3, 1.0, 0.0, 3.4],
            "slope": [0.0, 2.0, 1.0, 2.0],
            "ca": [0.0, 1.0, 2.0, 0.0],
            "thal": [1.0, 2.0, np.nan, 3.0],
            "target": [1, 0, 1, 0],
        }
    )


def test_column_groups_are_disjoint_and_cover_features():
    assert not set(NUMERIC_COLS) & set(CATEGORICAL_COLS)
    assert set(NUMERIC_COLS) | set(CATEGORICAL_COLS) == set(ALL_FEATURES)


def test_preprocessor_shape_one_hot(sample_df):
    prep = build_preprocessor(one_hot=True)
    out = prep.fit_transform(sample_df[ALL_FEATURES])
    # one-hot expands well beyond 13 raw columns
    assert out.shape[0] == len(sample_df)
    assert out.shape[1] > len(ALL_FEATURES)


def test_preprocessor_no_nan_in_output(sample_df):
    prep = build_preprocessor(one_hot=True)
    out = prep.fit_transform(sample_df[ALL_FEATURES])
    assert not np.isnan(np.asarray(out)).any()


def test_preprocessor_passthrough_shape(sample_df):
    prep = build_preprocessor(one_hot=False)
    out = prep.fit_transform(sample_df[ALL_FEATURES])
    assert out.shape == (len(sample_df), len(ALL_FEATURES))


def test_preprocessor_handles_unknown_category(sample_df):
    prep = build_preprocessor(one_hot=True)
    prep.fit(sample_df[ALL_FEATURES])
    unseen = sample_df[ALL_FEATURES].copy()
    unseen.loc[0, "thal"] = 7.0  # category never seen in training
    out = prep.transform(unseen.iloc[[0]])  # must not raise
    assert out.shape[0] == 1


def test_preprocessor_fits_in_end_to_end_pipeline(sample_df):
    pipe = Pipeline(
        [("prep", build_preprocessor(True)), ("clf", LogisticRegression(max_iter=1000))]
    )
    pipe.fit(sample_df[ALL_FEATURES], sample_df[TARGET])
    probs = pipe.predict_proba(sample_df[ALL_FEATURES])
    assert probs.shape == (4, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
