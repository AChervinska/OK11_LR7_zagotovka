"""Тести app/registry.py — крок 5 (реєстрація, просування, відкат)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import TfidfLogRegClassifier
from app.registry import (
    get_production_info,
    promote_to_production,
    register_model,
    rollback_to_previous,
)
from app.tracking import log_training_run
from scripts.evaluate import compute_metrics, evaluation_report

MODEL_NAME = "pytest-sentiment"


def _log(splits, tracking_uri, data_version):
    clf = TfidfLogRegClassifier()
    clf.train(splits[f"train_{data_version}"])
    predicted = clf.predict_labels(splits["eval"]["text"])
    return log_training_run(
        run_name=f"train-{data_version}",
        classifier=clf,
        train_df=splits[f"train_{data_version}"],
        eval_df=splits["eval"],
        metrics=compute_metrics(splits["eval"], predicted),
        report=evaluation_report(splits["eval"], predicted),
        data_version=data_version,
        tracking_uri=tracking_uri,
        experiment_name="pytest-registry",
    )


def test_register_creates_incrementing_versions(splits, tracking_uri):
    v1 = register_model(_log(splits, tracking_uri, "v1"), MODEL_NAME, tracking_uri)
    v2 = register_model(_log(splits, tracking_uri, "v2"), MODEL_NAME, tracking_uri)
    assert (v1, v2) == (1, 2)


def test_promote_then_rollback_swaps_production_pointer(splits, tracking_uri):
    v1 = register_model(_log(splits, tracking_uri, "v1"), MODEL_NAME, tracking_uri)
    v2 = register_model(_log(splits, tracking_uri, "v2"), MODEL_NAME, tracking_uri)

    promote_to_production(v1, MODEL_NAME, tracking_uri)
    assert get_production_info(MODEL_NAME, tracking_uri)["version"] == 1

    promote_to_production(v2, MODEL_NAME, tracking_uri)
    info = get_production_info(MODEL_NAME, tracking_uri)
    assert info["version"] == 2
    assert info["accuracy_hard"] is not None

    restored = rollback_to_previous(MODEL_NAME, tracking_uri)
    assert restored == 1
    assert get_production_info(MODEL_NAME, tracking_uri)["version"] == 1

    # повторний відкат повертає назад на 2 (аліаси просто міняються місцями)
    assert rollback_to_previous(MODEL_NAME, tracking_uri) == 2
