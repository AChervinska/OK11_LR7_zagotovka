"""
Стадія `evaluate` DVC-пайплайну (поглиблений рівень): оцінити збережену
модель на data/eval.csv і записати metrics.json (це DVC-метрика —
`dvc metrics show` / `dvc metrics diff` працюють саме з нею).

Готовий скрипт (не редагувати).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import load_pipeline_from_path
from scripts.evaluate import compute_metrics


def main() -> None:
    classifier = load_pipeline_from_path("models/pipeline")
    eval_df = pd.read_csv("data/eval.csv")
    metrics = compute_metrics(eval_df, classifier.predict_labels(eval_df["text"]))
    Path("metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("metrics.json:", metrics)


if __name__ == "__main__":
    main()
