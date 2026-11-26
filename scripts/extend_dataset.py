"""
Дописує до data/train_v2.csv додаткову партію прикладів із запереченням
(інший шаблон формулювання) — імітація «надійшли нові розмічені дані».

Використовується в кроці 9 (поглиблений рівень) як ЗМІНА даних, після
якої потрібно повторно `dvc add data/train_v2.csv` і побачити нове md5.

Готовий скрипт (не редагувати).

    python scripts/extend_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dataset import NEG_WORDS, POS_WORDS

EXTRA_TEMPLATES = [
    "Не сказав би, що {w} — радше навпаки.",
    "Аж ніяк не {w}, як писали в інших відгуках.",
]


def main() -> None:
    path = Path("data/train_v2.csv")
    df = pd.read_csv(path)

    extra = []
    for i in range(12):
        tmpl = EXTRA_TEMPLATES[i % len(EXTRA_TEMPLATES)]
        extra.append({"text": tmpl.format(w=NEG_WORDS[i % len(NEG_WORDS)]),
                      "label": "POSITIVE", "kind": "negation"})
        extra.append({"text": tmpl.format(w=POS_WORDS[i % len(POS_WORDS)]),
                      "label": "NEGATIVE", "kind": "negation"})

    out = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"{path}: було {len(df)} рядків, стало {len(out)} (+{len(extra)} із запереченням)")


if __name__ == "__main__":
    main()
