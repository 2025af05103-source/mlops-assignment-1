"""Train logistic-regression and random-forest classifiers on the heart-disease
data, tune with grid search + stratified 5-fold CV, log everything to MLflow,
and register the best pipeline as models/final_pipeline.joblib.

Usage: python src/train.py
"""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from preprocess import ALL_FEATURES, TARGET, build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_disease_clean.csv"
MODEL_PATH = ROOT / "models" / "final_pipeline.joblib"
MLRUNS_DIR = ROOT / "mlruns"
RANDOM_STATE = 42

# Personalized tuning grids (kept small so CI training finishes fast)
GRIDS = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "params": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "liblinear"],
        },
        "one_hot": True,
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [4, 6, None],
            "clf__min_samples_leaf": [2, 5],
        },
        "one_hot": False,
    },
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    X, y = df[ALL_FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
    }


def plot_and_log_confusion(y_true, y_pred, run_id):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (holdout)")
    path = ROOT / "artifacts" / f"confusion_{run_id}.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    mlflow.log_artifact(str(path))
    plt.close(fig)


def plot_and_log_roc(y_true, y_proba, run_id):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve (holdout)")
    path = ROOT / "artifacts" / f"roc_{run_id}.png"
    fig.savefig(path, bbox_inches="tight")
    mlflow.log_artifact(str(path))
    plt.close(fig)


def plot_class_balance(y, path):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(x=y, ax=ax)
    ax.set_title("Class Balance (0 = no disease, 1 = disease)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_one(name: str, cfg: dict, X_train, X_test, y_train, y_test) -> dict:
    """Grid-search one model family and log the best run to MLflow."""
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_family", name)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_params({f"grid_{k}": v for k, v in cfg["params"].items()})

        pipe = Pipeline(
            [
                ("prep", build_preprocessor(one_hot=cfg["one_hot"])),
                ("clf", cfg["estimator"]),
            ]
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(pipe, cfg["params"], cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        mlflow.log_params({f"best_{k}": v for k, v in gs.best_params_.items()})

        # Cross-validated metrics of the best estimator (re-scored on folds)
        cv_scores = cross_validate(
            best,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "roc_auc"],
        )
        for m in ["accuracy", "precision", "recall", "roc_auc"]:
            mlflow.log_metric(f"cv_{m}", float(np.mean(cv_scores[f"test_{m}"])))

        # Holdout metrics
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test, y_pred, y_proba)
        for m, v in test_metrics.items():
            mlflow.log_metric(f"test_{m}", float(v))

        plot_and_log_confusion(y_test, y_pred, run.info.run_id[:8])
        plot_and_log_roc(y_test, y_proba, run.info.run_id[:8])
        mlflow.sklearn.log_model(best, "model")

        log.info(
            "%s | CV ROC-AUC %.3f | test acc %.3f | test ROC-AUC %.3f | best %s",
            name,
            np.mean(cv_scores["test_roc_auc"]),
            test_metrics["accuracy"],
            test_metrics["roc_auc"],
            gs.best_params_,
        )
        return {
            "name": name,
            "pipeline": best,
            "test_roc_auc": test_metrics["roc_auc"],
            "test_metrics": test_metrics,
            "cv_roc_auc": float(np.mean(cv_scores["test_roc_auc"])),
        }


from sklearn.pipeline import (
    Pipeline,
)  # noqa: E402  (import after sklearn setup for readability)


def main():
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment("heart-disease-classification")
    X_train, X_test, y_train, y_test = load_data()
    log.info("Train %d rows, test %d rows", len(X_train), len(X_test))

    results = [
        run_one(name, cfg, X_train, X_test, y_train, y_test)
        for name, cfg in GRIDS.items()
    ]
    best = max(results, key=lambda r: r["test_roc_auc"])
    log.info(
        "Selected model: %s (holdout ROC-AUC %.3f)", best["name"], best["test_roc_auc"]
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_PATH)
    log.info("Saved final pipeline to %s", MODEL_PATH)

    summary = {
        "selected_model": best["name"],
        "selection_criterion": "highest holdout ROC-AUC",
        "results": [{k: v for k, v in r.items() if k != "pipeline"} for r in results],
    }
    (ROOT / "artifacts" / "model_selection_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
