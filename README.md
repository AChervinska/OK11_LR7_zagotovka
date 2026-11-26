# ЛР7 · Версіонування експериментів і автоматизація перенавчання

Повний текст інструкції (теорія, покроковий хід, контрольні питання,
критерії оцінювання) — у файлі `OK11_LR7_Metodychni_vkazivky.docx`
поруч із цим репозиторієм. Цей README — лише технічна навігація по коду.

Робота має **два рівні**:

- **Базовий (обов'язковий)** — кроки 1–8: навчання v1/v2, трекінг у
  MLflow, реєстрація й просування версії, підключення сервісу до
  реєстру, відкат. Перевіряється `pytest -m "not advanced"` і `bash verify.sh`.
- **Поглиблений (додатково)** — кроки 9–13: версіонування даних DVC,
  DVC-пайплайн, GitHub Actions, політика перенавчання. Перевіряється
  `pytest -m advanced` і `bash verify.sh advanced`.

## Структура

```
app/
  __init__.py
  config.py            — ГОТОВО: налаштування (tracking_uri, ім'я моделі в реєстрі)
  schemas.py           — ГОТОВО: контракт /predict (з ЛР1) + /model/info
  dataset.py           — ГОТОВО: детермінований генератор наборів v1/v2/eval
  model.py             — TODO: TfidfLogRegClassifier.train/predict; load_from_registry
  tracking.py          — TODO: log_training_run (обгортка MLflow)
  registry.py          — TODO: register_model / promote_to_production / rollback_to_previous / get_production_info
  retrain_policy.py    — TODO [поглиблений]: should_retrain
  main.py              — TODO: підключити load_from_registry у lifespan
scripts/
  generate_data.py     — ГОТОВО: пише data/train_v1.csv, train_v2.csv, eval.csv
  run_experiment.py    — ГОТОВО: навчає й логує один прогін (кличе ваш код)
  promote_best.py      — ГОТОВО: реєструє обидва прогони, просуває кращий
  rollback.py          — ГОТОВО: відкат активної версії
  evaluate.py          — ГОТОВО: compute_metrics (accuracy, f1, accuracy_hard)
  extend_dataset.py    — ГОТОВО [поглиблений]: дописує нові приклади в train_v2
  dvc_train.py         — ГОТОВО [поглиблений]: стадія train DVC-пайплайну
  dvc_evaluate.py      — ГОТОВО [поглиблений]: стадія evaluate DVC-пайплайну
params.yaml            — ГОТОВО [поглиблений]: параметри пайплайну
dvc.yaml               — TODO [поглиблений]: опис стадій prepare/train/evaluate
.github/workflows/ci.yml — TODO [поглиблений]: конвеєр CI
tests/                 — контрактні тести (conftest.py не редагувати)
verify.sh
requirements.txt
pytest.ini
```

## Встановлення

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_data.py
```

## Порядок виконання

**Базовий рівень**

1. Прочитати методичку, розділ «Хід роботи», кроки 1–8.
2. `pytest -m "not advanced" -q` — тести падають з `NotImplementedError`.
3. Реалізувати TODO: `app/model.py` → `app/tracking.py` → `app/registry.py` → `app/main.py`.
4. `python scripts/run_experiment.py v1` і `v2`, потім `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
5. `python scripts/promote_best.py`, `uvicorn app.main:app`, перевірити `/model/info` і `/predict`.
6. `python scripts/rollback.py` — і ще раз `/model/info`.
7. `bash verify.sh` — має завершитись «УСІ АВТОМАТИЧНІ ПЕРЕВІРКИ ПРОЙДЕНО».

**Поглиблений рівень (додатково)**

8. Реалізувати `app/retrain_policy.py`, `dvc.yaml`, `.github/workflows/ci.yml`.
9. `dvc init`, версіонування `data/train_v2.csv`, `dvc repro`, `dvc metrics diff`.
10. Запушити у власний репозиторій, показати зелений workflow.
11. `bash verify.sh advanced`.

## Типові проблеми

Див. розділ «Типові помилки» методички. Коротко: якщо `MlflowClient`
не бачить реєстру — перевірте, що `tracking_uri` вказує на `sqlite:///`,
а не на теку `./mlruns` (файловий бекенд реєстр моделей не підтримує).
