"""
Стадія `train` DVC-пайплайну (поглиблений рівень): навчити модель за
параметрами з params.yaml і зберегти pipeline у models/pipeline/.

Це окремий від MLflow механізм відтворюваності: DVC фіксує в dvc.lock
хеші коду, даних, параметрів і результату, тому `dvc repro` або
відтворює результат з кешу, або перебудовує рівно ті стадії, чиї входи
змінились.

Готовий скрипт (не редагувати).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import TfidfLogRegClassifier


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["train"]
    train_df = pd.read_csv(f"data/train_{params['data_version']}.csv")

    classifier = TfidfLogRegClassifier(ngram_max=params["ngram_max"], C=params["C"])
    classifier.train(train_df)

    out = Path("models/pipeline")
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(classifier.pipeline, str(out))
    print(f"models/pipeline збережено (data_version={params['data_version']}, C={params['C']})")


if __name__ == "__main__":
    main()
