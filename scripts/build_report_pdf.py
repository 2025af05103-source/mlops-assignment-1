"""Build the 10-page Assignment 1 report PDF from structured content + images.

Usage: .venv/bin/python scripts/build_report_pdf.py
Output: report/MLOps_Assignment1_Report_2025AF05103.pdf
"""

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)

ACCENT = (13, 71, 161)
GREY = (90, 90, 90)

SECTIONS = [
    (
        "1. Overview",
        [
            (
                "p",
                "This project delivers an end-to-end MLOps pipeline for heart-disease risk "
                "prediction from patient clinical data. The pipeline covers the complete "
                "lifecycle: automated data acquisition, exploratory data analysis (EDA), "
                "training and tuning of two model families with experiment tracking, model "
                "packaging, automated testing with CI/CD, containerization, Kubernetes "
                "deployment, and production monitoring with Prometheus and Grafana.",
            ),
            (
                "p",
                "Every stage is automated and reproducible: a single training script "
                "re-derives the dataset, re-trains both candidates under a fixed random "
                "seed, logs all experiments to MLflow, and registers the winning model as "
                "a single serialized pipeline artifact that the serving API loads directly.",
            ),
            (
                "kv",
                "Final model",
                "Random Forest - holdout ROC-AUC 0.942, accuracy 0.833, "
                "precision 0.875, recall 0.750",
            ),
            (
                "kv",
                "Serving",
                "FastAPI inside Docker, deployed on Kubernetes (2 replicas) "
                "behind a LoadBalancer",
            ),
            (
                "kv",
                "Monitoring",
                "Prometheus scraping /metrics every 15 s; Grafana "
                "dashboard for traffic, latency and prediction mix",
            ),
            (
                "img",
                "screenshots/01_mlflow_ui.png",
                "Figure 1: MLflow UI - experiment comparison of both tuned models",
            ),
        ],
    ),
    (
        "2. Setup and Installation",
        [
            (
                "p",
                "The project runs from a clean checkout with the commands below "
                "(Python 3.12+, all dependencies pinned in requirements.txt):",
            ),
            (
                "code",
                "git clone https://github.com/2025af05103-source/mlops-assignment-1.git\n"
                "cd mlops-assignment-1\n"
                "python3 -m venv .venv && source .venv/bin/activate\n"
                "pip install -r requirements.txt\n"
                "python scripts/download_data.py        # fetch + clean dataset\n"
                "python src/train.py                    # train, tune, log to MLflow\n"
                "uvicorn src.api.app:app --port 8000    # serve locally\n"
                "docker build -t heart-disease-api:latest -f docker/Dockerfile .\n"
                "kubectl apply -f k8s/ -f monitoring/   # deploy to local cluster",
            ),
            (
                "p",
                "Prerequisites: Docker Desktop with Kubernetes enabled (local cluster "
                "only - no cloud resources are used), kubectl, and optionally the MLflow "
                "UI (mlflow ui) for inspecting experiments at http://localhost:5000.",
            ),
        ],
    ),
    (
        "3. Data Acquisition and EDA (Task 1)",
        [
            (
                "p",
                "Dataset: UCI Heart Disease (Cleveland subset). The raw file contains 303 "
                "records; 6 rows with missing ca/thal values are dropped, yielding 297 "
                "clean rows. The target is binarized to 0/1 (presence of heart disease). "
                "The download step is scripted (scripts/download_data.py) so the dataset "
                "is fetched and cleaned deterministically from a fixed URL.",
            ),
            (
                "kv",
                "Rows / features",
                "297 rows x 13 clinical features + binary target",
            ),
            (
                "kv",
                "Class split",
                "160 no-disease (0) vs 137 disease (1) - mild imbalance, "
                "no resampling needed",
            ),
            (
                "img",
                "artifacts/eda/class_balance.png",
                "Figure 2: Class balance - 160 no-disease vs 137 disease patients",
            ),
            (
                "p",
                "Categorical analysis shows expected clinical signal: asymptomatic chest "
                "pain (cp=0), exercise-induced angina (exang=1), reversible thal defect "
                "(thal=7) and higher ca (major-vessel fluoroscopy) counts all carry "
                "markedly higher disease rates (e.g. ca=3 ~ 0.83, thal=7 ~ 0.78).",
            ),
            (
                "img",
                "artifacts/eda/categorical_rates.png",
                "Figure 3: Disease rate by categorical feature level",
            ),
            (
                "p",
                "Correlation analysis of the numeric features identifies thalach (max "
                "heart rate, -0.42) and oldpeak (ST depression, +0.42) as the strongest "
                "numeric signals, with ca (+0.46) and thal (+0.53) strongest overall. "
                "Disease patients show lower thalach and higher oldpeak in the "
                "distribution plots.",
            ),
            (
                "img",
                "artifacts/eda/correlation_heatmap.png",
                "Figure 4: Correlation heatmap - target correlations: thal 0.53, ca 0.46, "
                "oldpeak 0.42, thalach -0.42",
            ),
            (
                "img",
                "artifacts/eda/numeric_distributions.png",
                "Figure 5: Numeric feature distributions by class",
            ),
        ],
    ),
    (
        "4. Model Training and Evaluation (Task 2)",
        [
            (
                "p",
                "Two model families are trained, each wrapped in a full sklearn Pipeline "
                "(median imputation -> scaling -> one-hot/ordinal encoding -> classifier). "
                "Hyperparameters are tuned with GridSearchCV over a small personalized "
                "grid using stratified 5-fold cross-validation scored on ROC-AUC. The "
                "data is split 80/20 with stratification and a fixed seed (42).",
            ),
            (
                "table",
                [
                    "Model",
                    "Best params",
                    "CV ROC-AUC",
                    "Test ROC-AUC",
                    "Acc",
                    "Prec",
                    "Rec",
                ],
                [
                    [
                        "Logistic Regression",
                        "C=1.0, liblinear",
                        "0.903",
                        "0.941",
                        "0.817",
                        "0.870",
                        "0.714",
                    ],
                    [
                        "Random Forest",
                        "200 trees, depth 6, leaf 2",
                        "0.895",
                        "0.942",
                        "0.833",
                        "0.875",
                        "0.750",
                    ],
                ],
            ),
            (
                "p",
                "Selection criterion: highest holdout ROC-AUC -> Random Forest. The two "
                "families perform nearly identically; RF wins marginally and additionally "
                "provides feature importances for interpretation. Both tuned models and "
                "their full metric sets are preserved as MLflow runs (Section 5).",
            ),
            (
                "img",
                "artifacts/confusion_7e266856.png",
                "Figure 6: Random Forest confusion matrix on the holdout set",
            ),
            (
                "img",
                "artifacts/roc_7e266856.png",
                "Figure 7: Random Forest ROC curve on the holdout set (AUC 0.942)",
            ),
        ],
    ),
    (
        "5. Experiment Tracking with MLflow (Task 3)",
        [
            (
                "p",
                "All training runs are tracked in a local MLflow store (mlruns/, "
                "experiment heart-disease-classification). Each run logs the model "
                "family, the full hyperparameter grid, the best hyperparameters found, "
                "cross-validated metrics (accuracy, precision, recall, ROC-AUC) and "
                "holdout metrics, plus confusion-matrix and ROC-curve artifacts and the "
                "pickled pipeline itself.",
            ),
            ("kv", "Tracking URI", "file://mlruns (local filesystem store)"),
            ("kv", "Experiment", "heart-disease-classification"),
            ("kv", "Runs", "logistic_regression and random_forest, one tuned run each"),
            (
                "p",
                "The UI (mlflow ui -> http://localhost:5000) enables side-by-side "
                "comparison of the two candidates, which drove the model selection "
                "decision in Section 4. A programmatic summary is also written to "
                "artifacts/model_selection_summary.json.",
            ),
            (
                "img",
                "screenshots/01_mlflow_ui.png",
                "Figure 8: MLflow experiment view - both runs with metrics",
            ),
            (
                "img",
                "screenshots/01b_mlflow_run_detail.png",
                "Figure 9: MLflow run detail - parameters, metrics and artifacts",
            ),
        ],
    ),
    (
        "6. Packaging and Reproducibility (Task 4)",
        [
            (
                "p",
                "The winning pipeline - preprocessing and classifier serialized together "
                "as one joblib object (models/final_pipeline.joblib) - is the single "
                "deployment artifact. The serving API loads exactly this object, so "
                "there is no duplication of preprocessing logic between training and "
                "serving, eliminating train/serve skew.",
            ),
            (
                "ul",
                [
                    "final_pipeline.joblib - fitted preprocessing + model in one artifact",
                    "requirements.txt - pinned dependency versions",
                    "Fixed random seed (42) everywhere: split, CV folds, estimators",
                    "scripts/download_data.py - deterministic dataset fetch from fixed URL",
                    "Full pipeline re-runnable from a clean checkout in minutes",
                ],
            ),
            (
                "p",
                "Reproducibility was verified by re-running the complete flow "
                "(download -> train -> serve) after a fresh clone: metrics reproduce to "
                "within identical values thanks to the fixed seeds.",
            ),
        ],
    ),
    (
        "7. Testing and CI/CD (Task 5)",
        [
            (
                "p",
                "The project carries 16 pytest tests across three suites: preprocessing "
                "(column groups disjoint, one-hot expansion, no NaNs in output, unknown "
                "category handling, end-to-end pipeline fit), training utilities plus a "
                "saved-model quality guardrail (test ROC-AUC must exceed 0.80), and the "
                "API contract (health, successful prediction, validation errors on "
                "out-of-range fields, metrics endpoint).",
            ),
            (
                "img",
                "screenshots/10_test_run.png",
                "Figure 10: pytest run - all 16 tests passing",
            ),
            (
                "p",
                "GitHub Actions (.github/workflows/ci.yml) implements the CD pipeline as "
                "four chained jobs: lint (flake8 + black), test (pytest with uploaded "
                "results), train (trains both models with MLflow and uploads the model "
                "and run artifacts), and docker (builds the image and smoke-tests /health "
                "and /predict inside the container). Any job failure fails the pipeline "
                "with full logs.",
            ),
            (
                "img",
                "screenshots/08_github_actions_run.png",
                "Figure 11: GitHub Actions - green pipeline run",
            ),
            (
                "img",
                "screenshots/09_github_actions_commit_checks.png",
                "Figure 12: Commit checks - all four jobs passed",
            ),
        ],
    ),
    (
        "8. Containerization (Task 6)",
        [
            (
                "p",
                "The API ships as a Docker image built from docker/Dockerfile: a slim "
                "python:3.12 base, layer-cached dependency install, non-root-friendly "
                "uvicorn entrypoint, and a container healthcheck hitting /health.",
            ),
            (
                "code",
                "docker build -t heart-disease-api:latest -f docker/Dockerfile .\n"
                "docker run -p 8000:80 heart-disease-api:latest\n"
                "curl localhost:8000/health\n"
                "curl localhost:8000/predict -H 'Content-Type: application/json' \\\n"
                "     -d @data/sample_input.json",
            ),
            (
                "p",
                "Container predictions were verified identical to the locally-run API "
                "(evidence in artifacts/docker_proof.txt), confirming the packaged "
                "artifact serves exactly the trained model.",
            ),
        ],
    ),
    (
        "9. Deployment on Kubernetes (Task 7)",
        [
            (
                "p",
                "The container is deployed on a local Docker Desktop Kubernetes cluster. "
                "k8s/deployment.yaml runs 2 replicas with liveness and readiness probes "
                "on /health and resource limits; k8s/service.yaml exposes the API "
                "through a LoadBalancer on port 80, so predictions are available at "
                "http://localhost with traffic spread across both replicas.",
            ),
            (
                "img",
                "screenshots/05_k8s_pod.png",
                "Figure 13: kubectl get pods,svc - 2/2 replicas running, LoadBalancer mapped",
            ),
            (
                "img",
                "screenshots/06_api_predict_lb.png",
                "Figure 14: Prediction served through the LoadBalancer at localhost:80",
            ),
        ],
    ),
    (
        "10. Monitoring and Logging (Task 8)",
        [
            (
                "p",
                "The API emits structured request logs (method, path, status, latency) "
                "via middleware and exposes Prometheus metrics on /metrics: request "
                "count by endpoint/status, a latency histogram, prediction counts by "
                "class, latest prediction confidence, and a model-loaded flag.",
            ),
            (
                "ul",
                [
                    "api_requests_total - requests by method/endpoint/status",
                    "api_request_latency_seconds - latency histogram per endpoint",
                    "predictions_total - predictions by class (0/1)",
                    "prediction_confidence - gauge of latest confidence",
                    "model_loaded - 1 when the model artifact loaded successfully",
                ],
            ),
            (
                "p",
                "Prometheus runs in-cluster and scrapes the API every 15 seconds; the "
                "target is verified up on the Prometheus targets page. Grafana "
                "(localhost:3000, Prometheus datasource auto-provisioned) visualizes "
                "request rate, latency and the prediction class mix over time.",
            ),
            (
                "img",
                "screenshots/08_prometheus_targets.png",
                "Figure 15: Prometheus targets - heart-disease-api endpoint UP",
            ),
            (
                "img",
                "screenshots/07_grafana_dashboard.png",
                "Figure 16: Grafana dashboard - request rate, latency, prediction mix",
            ),
        ],
    ),
    (
        "11. Architecture",
        [
            (
                "p",
                "The end-to-end flow: a push to main triggers CI which lints, tests, "
                "trains (logging to MLflow) and builds the container image. Locally, the "
                "same training flow produces the pipeline artifact that the container "
                "serves; the Kubernetes deployment exposes it behind a LoadBalancer and "
                "feeds metrics to Prometheus/Grafana.",
            ),
            (
                "code",
                "GitHub --push--> GitHub Actions (lint -> test -> train -> docker build)\n"
                "                     | uploads: test results, MLflow runs, model\n"
                "                     v\n"
                "scripts/download_data.py --> data/heart_disease_clean.csv\n"
                "       |\n"
                "       v\n"
                "src/train.py --MLflow--> mlruns/   +   models/final_pipeline.joblib\n"
                "       |                                     |\n"
                "       v                                     v\n"
                "docker/Dockerfile --> image --> k8s Deployment (2 replicas, probes)\n"
                "                                | Service (LoadBalancer :80)\n"
                "                                |--> /health /predict (FastAPI)\n"
                "                                |--> /metrics --scrape--> Prometheus --> Grafana",
            ),
        ],
    ),
    (
        "12. How to Verify",
        [
            (
                "ul",
                [
                    "API: curl localhost/predict -H 'Content-Type: application/json' "
                    "-d @data/sample_input.json",
                    "MLflow UI: mlflow ui -> http://localhost:5000",
                    "Tests: pytest tests/ -v  (16 tests)",
                    "Metrics: curl localhost/metrics; Grafana at localhost:3000 (admin/admin)",
                    "CI: github.com/2025af05103-source/mlops-assignment-1/actions - green runs",
                ],
            ),
        ],
    ),
]


class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, "Heart Disease Risk Prediction - MLOps Assignment 1", align="L")
        self.cell(
            0, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT"
        )
        self.ln(2)

    def footer(self):
        self.set_y(-14)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(
            0,
            8,
            f"Shubham Joshi | 2025AF05103 | Page {self.page_no()} of " f"{{nb}}",
            align="C",
        )


pdf = Report(orientation="P", unit="mm", format="A4")
pdf.set_margins(18, 16, 18)
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()
pdf.alias_nb_pages()

# ---- Cover header block (first page top) ----
pdf.set_font("helvetica", "B", 20)
pdf.set_text_color(*ACCENT)
pdf.cell(0, 12, "Heart Disease Risk Prediction", new_x="LMARGIN", new_y="NEXT")
pdf.cell(
    0, 10, "End-to-End MLOps Pipeline - Assignment 1", new_x="LMARGIN", new_y="NEXT"
)
pdf.ln(2)
pdf.set_font("helvetica", "", 11)
pdf.set_text_color(0, 0, 0)
pdf.cell(
    0,
    7,
    "Course: MLOps (AIMLCZG523)   |   Student: Shubham Joshi   |   " "ID: 2025AF05103",
    new_x="LMARGIN",
    new_y="NEXT",
)
pdf.cell(0, 7, "Date: 2026-10-20", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.set_draw_color(*ACCENT)
pdf.set_line_width(0.6)
pdf.line(18, pdf.get_y(), 192, pdf.get_y())
pdf.ln(4)

for title, blocks in SECTIONS:
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    if pdf.get_y() > 250:
        pdf.add_page()
    # Section heading with accent bar
    pdf.set_fill_color(*ACCENT)
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(2, 8, "")
    pdf.cell(0, 8, title, fill=False, new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y() - 8
    pdf.rect(18, y, 1.4, 8, style="F")
    pdf.ln(2)

    for kind, *args in blocks:
        if kind == "p":
            pdf.set_font("helvetica", "", 10)
            pdf.multi_cell(0, 5.2, args[0])
            pdf.ln(1.5)
        elif kind == "kv":
            pdf.set_font("helvetica", "B", 10)
            pdf.write(5.2, f"{args[0]}: ")
            pdf.set_font("helvetica", "", 10)
            pdf.multi_cell(0, 5.2, args[1])
            pdf.ln(1.2)
        elif kind == "ul":
            pdf.set_font("helvetica", "", 10)
            for item in args[0]:
                pdf.set_x(24)
                pdf.cell(4, 5.2, "-")
                pdf.multi_cell(0, 5.2, item)
            pdf.ln(1.5)
        elif kind == "code":
            pdf.set_font("courier", "", 8.5)
            pdf.set_fill_color(243, 244, 246)
            pdf.multi_cell(0, 4.6, args[0], fill=True)
            pdf.ln(2)
        elif kind == "table":
            headers, rows = args
            pdf.set_font("helvetica", "B", 8.5)
            pdf.set_fill_color(*ACCENT)
            pdf.set_text_color(255, 255, 255)
            widths = [38, 44, 20, 22, 15, 15, 15]
            for h, w in zip(headers, widths):
                pdf.cell(w, 7, h, border=1, fill=True, align="C")
            pdf.ln()
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("helvetica", "", 8.5)
            for i, row in enumerate(rows):
                if i == 1:
                    pdf.set_font("helvetica", "B", 8.5)
                for cell, w in zip(row, widths):
                    pdf.cell(w, 7, cell, border=1, align="C")
                pdf.ln()
            pdf.ln(2)
        elif kind == "img":
            rel, caption = args
            path = ROOT / rel
            if not path.exists():
                continue
            # fit width 150mm, keep aspect; add page if not enough room
            from PIL import Image

            with Image.open(path) as im:
                w_px, h_px = im.size
            w_mm = 150
            h_mm = w_mm * h_px / w_px
            max_h = 95
            if h_mm > max_h:
                w_mm = max_h * w_px / h_px
                h_mm = max_h
            if pdf.get_y() + h_mm + 10 > 277:
                pdf.add_page()
            x = (210 - w_mm) / 2
            pdf.image(str(path), x=x, w=w_mm)
            pdf.set_font("helvetica", "I", 8.5)
            pdf.set_text_color(*GREY)
            pdf.multi_cell(0, 4.5, caption, align="C")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

out_path = OUT / "MLOps_Assignment1_Report_2025AF05103.pdf"
pdf.output(out_path)
print(f"Written: {out_path}")
print(f"Pages: {pdf.page_no()}")
