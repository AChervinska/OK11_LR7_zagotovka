"""Тести app/tracking.py — крок 3."""

from __future__ import annotations

import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import TfidfLogRegClassifier
from app.tracking import log_training_run
from scripts.evaluate import compute_metrics, evaluation_report


def _train_and_log(splits, tracking_uri, data_version="v2"):
    clf = TfidfLogRegClassifier()
    clf.train(splits[f"train_{data_version}"])
    predicted = clf.predict_labels(splits["eval"]["text"])
    metrics = compute_metrics(splits["eval"], predicted)
    report = evaluation_report(splits["eval"], predicted)
    run_id = log_training_run(
        run_name=f"train-{data_version}",
        classifier=clf,
        train_df=splits[f"train_{data_version}"],
        eval_df=splits["eval"],
        metrics=metrics,
        report=report,
        data_version=data_version,
        tracking_uri=tracking_uri,
        experiment_name="pytest-exp",
    )
    return run_id


def test_run_logs_params_metrics_artifact_model(splits, tracking_uri):
    run_id = _train_and_log(splits, tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    assert run.data.params["data_version"] == "v2"
    assert int(run.data.params["n_train"]) == len(splits["train_v2"])
    for metric in ("accuracy", "f1", "accuracy_hard"):
        assert metric in run.data.metrics

    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert "eval_report.json" in artifacts
    assert "model" in artifacts


def test_two_runs_are_independent(splits, tracking_uri):
    r1 = _train_and_log(splits, tracking_uri, "v1")
    r2 = _train_and_log(splits, tracking_uri, "v2")
    client = MlflowClient(tracking_uri=tracking_uri)
    acc_hard_v1 = client.get_run(r1).data.metrics["accuracy_hard"]
    acc_hard_v2 = client.get_run(r2).data.metrics["accuracy_hard"]
    assert acc_hard_v2 > acc_hard_v1
