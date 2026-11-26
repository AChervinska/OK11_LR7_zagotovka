"""
Відкат активної версії моделі до попередньої (перестановка alias
'production' <-> 'previous').

Викликає ВАШ код: rollback_to_previous / get_production_info (крок 5).
Готовий (не редагувати).

    python scripts/rollback.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.registry import get_production_info, rollback_to_previous


def main() -> None:
    settings = Settings()
    before = get_production_info(settings.registry_model_name)
    restored = rollback_to_previous(settings.registry_model_name)
    after = get_production_info(settings.registry_model_name)

    print(f"Було в проді:   версія {before['version']} (accuracy_hard={before['accuracy_hard']})")
    print(f"Стало в проді:  версія {after['version']} (accuracy_hard={after['accuracy_hard']})")
    print(f"rollback_to_previous повернув: {restored}")
    print("\nПерезапустіть сервіс (uvicorn app.main:app), щоб він перечитав реєстр.")


if __name__ == "__main__":
    main()
