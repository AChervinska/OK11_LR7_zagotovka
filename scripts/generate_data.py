"""
Записує навчальні набори на диск: data/train_v1.csv, data/train_v2.csv,
data/eval.csv. Готовий скрипт (не редагувати) — крок 1 методички.

    python scripts/generate_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.dataset import build_splits


def main() -> None:
    settings = Settings()
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    splits = build_splits()
    for name, examples in splits.items():
        df = pd.DataFrame(
            [{"text": e.text, "label": e.label, "kind": e.kind} for e in examples]
        )
        path = data_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        n_hard = int((df["kind"] == "negation").sum())
        n_pos = int((df["label"] == "POSITIVE").sum())
        print(
            f"{path}: {len(df)} рядків "
            f"(POSITIVE={n_pos}, NEGATIVE={len(df) - n_pos}, із них із запереченням={n_hard})"
        )


if __name__ == "__main__":
    main()
