"""
Реєстрація обох прогонів як версій моделі і просування кращого в
'production'.

«Кращий» визначається за accuracy_hard (accuracy на прикладах із
запереченням) — саме ці приклади були предметом суперечки в ЛР6, тому
загальна accuracy тут менш інформативна (підрозділ 3.4 методички).

Викликає ВАШ код: register_model / promote_to_production (крок 5).
Готовий (не редагувати).

    python scripts/promote_best.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.registry import promote_to_production, register_model


def main() -> None:
    settings = Settings()
    mlflow.set_tracking_uri(settings.tracking_uri)

    experiment = mlflow.get_experiment_by_name(settings.experiment_name)
    if experiment is None:
        raise SystemExit("Немає експерименту — спершу запустіть run_experiment.py v1 і v2")

    runs = mlflow.search_runs(
        [experiment.experiment_id], order_by=["attributes.start_time ASC"]
    )
    by_version = {}
    for _, row in runs.iterrows():
        by_version.setdefault(row["params.data_version"], row)

    if {"v1", "v2"} - by_version.keys():
        raise SystemExit("Потрібні обидва прогони: v1 і v2 (scripts/run_experiment.py)")

    registered = {}
    for data_version in ("v1", "v2"):
        row = by_version[data_version]
        version = register_model(row["run_id"], settings.registry_model_name)
        registered[data_version] = (version, float(row["metrics.accuracy_hard"]))
        print(
            f"{data_version}: зареєстровано версію {version}, "
            f"accuracy_hard={row['metrics.accuracy_hard']}"
        )

    # v1 — це модель, що прийшла в ЛР7 з ЛР1-6: спершу робимо її поточним
    # 'production' (базовою лінією), а вже потім просуваємо переможця. Так
    # alias 'previous' отримує коректне значення й відкат стає можливим.
    promote_to_production(registered["v1"][0], settings.registry_model_name)

    best_dv = max(registered, key=lambda dv: registered[dv][1])
    best_version = registered[best_dv][0]
    promote_to_production(best_version, settings.registry_model_name)
    print(
        f"\nБазова лінія 'production' — версія {registered['v1'][0]} (v1); "
        f"просунуто версію {best_version} (з {best_dv}, "
        f"accuracy_hard={registered[best_dv][1]}). Попередня версія збережена під alias 'previous'."
    )


if __name__ == "__main__":
    main()
