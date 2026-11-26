"""
Навчання й логування одного прогону в MLflow.

    python scripts/run_experiment.py v1     # навчити на train_v1.csv
    python scripts/run_experiment.py v2     # навчити на train_v2.csv

Скрипт викликає ВАШ код: TfidfLogRegClassifier.train/predict (крок 2) і
log_training_run (крок 3). Готовий (не редагувати).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.model import TfidfLogRegClassifier
from app.tracking import log_training_run
from scripts.evaluate import compute_metrics, evaluation_report


def main(data_version: str) -> None:
    if data_version not in {"v1", "v2"}:
        raise SystemExit("Використання: python scripts/run_experiment.py v1|v2")

    settings = Settings()
    data_dir = Path(settings.data_dir)
    train_df = pd.read_csv(data_dir / f"train_{data_version}.csv")
    eval_df = pd.read_csv(data_dir / "eval.csv")

    classifier = TfidfLogRegClassifier()
    classifier.train(train_df)

    predicted = classifier.predict_labels(eval_df["text"])
    metrics = compute_metrics(eval_df, predicted)
    report = evaluation_report(eval_df, predicted)

    run_id = log_training_run(
        run_name=f"train-{data_version}",
        classifier=classifier,
        train_df=train_df,
        eval_df=eval_df,
        metrics=metrics,
        report=report,
        data_version=data_version,
    )

    print(f"run_id={run_id}  data_version={data_version}")
    print(
        "  accuracy={accuracy}  f1={f1}  accuracy_hard={accuracy_hard}".format(**metrics)
    )
    print(f"  Перегляд: mlflow ui --backend-store-uri {settings.tracking_uri}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
