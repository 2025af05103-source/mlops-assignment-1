# Screenshots — capture checklist

Take these while the local cluster is running (deployed state) and save here:

1. `01_mlflow_ui.png` — MLflow UI run comparison (`mlflow ui` → localhost:5000)
2. `02_eda_plots.png` — EDA plots from `artifacts/eda/` (or the notebook open in Jupyter)
3. `03_docker_build.png` — `docker build` output (rerun if needed)
4. `04_docker_run_predict.png` — `docker run` + curl /predict output
5. `05_k8s_pods.png` — `kubectl get pods,svc` output
6. `06_api_predict_lb.png` — curl through localhost (LoadBalancer)
7. `07_grafana_dashboard.png` — Grafana (localhost:3000, admin/admin) with API dashboard
8. `08_prometheus_targets.png` — Prometheus targets page (port-forward: `kubectl port-forward deploy/prometheus 9090:9090`)
9. `09_github_actions.png` — CI workflow green run on GitHub
10. `10_test_run.png` — `pytest tests/ -v` output
