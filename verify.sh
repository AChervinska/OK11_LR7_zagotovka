#!/usr/bin/env bash
# Наскрізна перевірка ЛР7. Запускати з кореня репозиторію.
#
#   bash verify.sh            # базовий рівень
#   bash verify.sh advanced   # додатково: поглиблені тести + dvc repro

set -euo pipefail
FAILED=0
pass() { echo "OK: $1"; }
fail() { echo "FAIL: $1"; FAILED=1; }

echo "== Генерація даних =="
python scripts/generate_data.py

echo "== Тести (базовий рівень) =="
if pytest -m "not advanced" -q; then pass "базові тести"; else fail "базові тести"; fi

echo "== Наскрізний цикл: навчання v1/v2, реєстрація, просування =="
rm -f mlflow.db
python scripts/run_experiment.py v1
python scripts/run_experiment.py v2
python scripts/promote_best.py

echo "== Активна версія після просування =="
if python -c "
from app.registry import get_production_info
info = get_production_info()
print(info)
assert info['version'] == 2, info
"; then pass "у 'production' версія 2 (v2 — краща за accuracy_hard)"; else fail "очікувалась версія 2"; fi

echo "== Відкат =="
python scripts/rollback.py
if python -c "
from app.registry import get_production_info
assert get_production_info()['version'] == 1
"; then pass "відкат повернув версію 1"; else fail "відкат не спрацював"; fi

if [ "${1:-}" = "advanced" ]; then
    echo "== Тести (поглиблений рівень) =="
    if pytest -m advanced -q; then pass "поглиблені тести"; else fail "поглиблені тести"; fi

    echo "== DVC-пайплайн =="
    if command -v dvc >/dev/null 2>&1; then
        [ -d .dvc ] || dvc init -q --no-scm
        if dvc repro; then
            pass "dvc repro відпрацював"
            echo "metrics.json:"; cat metrics.json; echo
        else
            fail "dvc repro завершився помилкою"
        fi
    else
        fail "dvc не встановлено"
    fi
fi

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "УСІ АВТОМАТИЧНІ ПЕРЕВІРКИ ПРОЙДЕНО"
    exit 0
else
    echo "Є ПОМИЛКИ — див. FAIL вище"
    exit 1
fi
