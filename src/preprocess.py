"""Reusable preprocessing for the heart-disease dataset.

Defines column groups and builds the sklearn Pipeline used both at training
time and at serving time (the fitted pipeline is saved together with the
model so the API needs no separate preprocessing logic).
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Numeric features kept as continuous measurements
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Categorical/coded features (including binaries) -> one-hot for linear models
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS
TARGET = "target"


def build_preprocessor(one_hot: bool = True) -> ColumnTransformer:
    """Return the feature transformer.

    one_hot=True  -> numeric scaled + categorical one-hot (Logistic Regression)
    one_hot=False -> numeric scaled + categorical passthrough (tree models)
    """
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    if one_hot:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
    else:
        categorical = "passthrough"
    return ColumnTransformer(
        [("num", numeric, NUMERIC_COLS), ("cat", categorical, CATEGORICAL_COLS)],
        remainder="drop",
    )
