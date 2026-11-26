"""
Обгортка над MLflow Tracking: логування одного прогону навчання
(параметри, метрики, артефакт-звіт, сама модель).

TODO: log_training_run() — крок 3.
"""


from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature

from app.config import Settings
from app.model import TfidfLogRegClassifier

settings = Settings()


def log_training_run(
    *,
    run_name: str,
    classifier: TfidfLogRegClassifier,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    metrics: dict,
    report: dict,
    data_version: str,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> str:

    uri = tracking_uri or settings.tracking_uri

    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)

    mlflow.set_experiment(
        experiment_name or settings.experiment_name
    )

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_params({
            "ngram_max": classifier.ngram_max,
            "C": classifier.C,
            "n_train": len(train_df),
            "data_version": data_version,
        })

        mlflow.log_metrics({
            key: float(value)
            for key, value in metrics.items()
            if value is not None
        })

        with tempfile.TemporaryDirectory() as temp_dir:

            report_path = (
                Path(temp_dir) / "eval_report.json"
            )

            report_path.write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            mlflow.log_artifact(str(report_path))

        sample = eval_df["text"].head(5).tolist()

        signature = infer_signature(
            sample,
            classifier.predict_labels(sample),
        )

        mlflow.sklearn.log_model(
            classifier.pipeline,
            artifact_path="model",
            signature=signature,
        )

        return run.info.run_id
