"""Тести app/model.py — крок 2."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import TfidfLogRegClassifier
from scripts.evaluate import compute_metrics


def test_predict_returns_label_and_score(splits):
    clf = TfidfLogRegClassifier()
    clf.train(splits["train_v1"])
    label, score = clf.predict("Пристрій чудовий, ось моє враження.")
    assert label in {"POSITIVE", "NEGATIVE"}
    assert 0.5 <= score <= 1.0


def test_v1_learns_direct_sentiment(splits):
    clf = TfidfLogRegClassifier()
    clf.train(splits["train_v1"])
    assert clf.predict("Товар жахливий.")[0] == "NEGATIVE"
    assert clf.predict("Товар чудовий.")[0] == "POSITIVE"


def test_adding_negation_data_fixes_the_systematic_error(splits):
    """Ключова властивість роботи: v1 (лише прямі приклади) систематично
    помиляється на відгуках із запереченням; додавання цих прикладів у
    train_v2 виправляє помилку — і це видно на метриці accuracy_hard."""
    hard = splits["eval"][splits["eval"]["kind"] == "negation"]

    v1 = TfidfLogRegClassifier()
    v1.train(splits["train_v1"])
    v2 = TfidfLogRegClassifier()
    v2.train(splits["train_v2"])

    acc_hard_v1 = compute_metrics(hard, v1.predict_labels(hard["text"]))["accuracy"]
    acc_hard_v2 = compute_metrics(hard, v2.predict_labels(hard["text"]))["accuracy"]

    assert acc_hard_v1 <= 0.25
    assert acc_hard_v2 >= 0.9
