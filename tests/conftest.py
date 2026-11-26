"""
Фікстури тестів ЛР7.

Тести НЕ вимагають запущеного mlflow-сервера: трекінг і реєстр пишуться
в тимчасовий SQLite-файл (фікстура tracking_uri), який живе рівно один
тест. Дані — детерміновані (app.dataset), тому числа стабільні.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dataset import build_splits


@pytest.fixture(scope="session")
def splits() -> dict[str, pd.DataFrame]:
    raw = build_splits()
    return {
        name: pd.DataFrame(
            [{"text": e.text, "label": e.label, "kind": e.kind} for e in examples]
        )
        for name, examples in raw.items()
    }


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
