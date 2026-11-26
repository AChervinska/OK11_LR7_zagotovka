"""
Обчислення метрик якості моделі на holdout-наборі.

    accuracy       — частка правильних на всьому eval.csv;
    f1             — macro-F1 по двох класах;
    accuracy_hard  — accuracy ЛИШЕ на прикладах із запереченням (kind == "negation");
                     саме ця метрика показує різницю між v1 і v2.

Готовий модуль (не редагувати) — використовується в scripts/run_experiment.py,
scripts/promote_best.py і в DVC-пайплайні (поглиблений рівень).
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(eval_df: pd.DataFrame, predicted_labels: pd.Series) -> dict:
    y_true = eval_df["label"].to_numpy()
    y_pred = pd.Series(predicted_labels).to_numpy()

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }

    hard_mask = (eval_df["kind"] == "negation").to_numpy()
    if hard_mask.any():
        metrics["accuracy_hard"] = round(
            float(accuracy_score(y_true[hard_mask], y_pred[hard_mask])), 4
        )
    else:
        metrics["accuracy_hard"] = None

    return metrics


def evaluation_report(eval_df: pd.DataFrame, predicted_labels: pd.Series) -> dict:
    """Розгорнутий звіт для збереження як MLflow-артефакт (крок 3)."""
    labels = ["NEGATIVE", "POSITIVE"]
    cm = confusion_matrix(eval_df["label"], pd.Series(predicted_labels), labels=labels)
    return {
        "metrics": compute_metrics(eval_df, predicted_labels),
        "n_eval": int(len(eval_df)),
        "n_hard": int((eval_df["kind"] == "negation").sum()),
        "confusion_matrix": {"labels": labels, "rows": cm.tolist()},
    }
