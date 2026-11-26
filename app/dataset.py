"""
Детермінований генератор навчальних наборів для ЛР7.

Сюжет продовжує ЛР6. Модель — лінійний класифікатор тональності
(TF-IDF з біграмами + логістична регресія), який навчається за секунди
на CPU, тому його МОЖНА реально перенавчати на занятті (donavchyty
distilbert із ЛР1 за 90 хв неможливо — це свідоме спрощення).

Два джерела прикладів:

* ПРЯМІ відгуки — шаблон + один полярний прикметник; мітка збігається
  з полярністю слова ("Пристрій жахливий" → NEGATIVE).
* Відгуки із ЗАПЕРЕЧЕННЯМ — "зовсім не <негативне>" насправді POSITIVE,
  "зовсім не <позитивне>" насправді NEGATIVE. Це ті самі складні
  приклади, що людина розмічала в черзі ЛР6.

train_v1.csv — лише прямі приклади. Модель v1 не бачила жодної біграми
виду "не <слово>", тому орієнтується на полярний прикметник і ВПЕВНЕНО
помиляється на кожному відгуку із запереченням (випадок "confidently
wrong" із ЛР6, крок 7).

train_v2.csv = train_v1 + великий блок прикладів із запереченням.
Тепер біграми "не жахливий", "не якісний" тощо мають достатньо
спостережень, щоб переважити відповідну уніграму — і v2 класифікує
складні приклади правильно.

eval.csv — спільний, збалансований holdout: 40 прямих (20/20) + 20 із
запереченням (10/10). Метрика accuracy_hard рахується лише на других 20.

Генерація повністю детермінована — числа у всіх студентів однакові.
Цей файл видається готовим, редагувати не потрібно.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SEED = 20260903

POS_WORDS = [
    "чудовий", "приємний", "якісний", "надійний",
    "зручний", "швидкий", "акуратний", "практичний",
]
NEG_WORDS = [
    "жахливий", "неприємний", "поганий", "ненадійний",
    "незручний", "повільний", "неохайний", "непридатний",
]

DIRECT_TEMPLATES = [
    "Товар {w}.",
    "Пристрій {w}, ось моє враження.",
    "Отриманий ноутбук {w}.",
    "Загалом пристрій {w}, це головне.",
    "Замовлення прийшло, товар {w}.",
]

# Ключовий сигнал для біграмної моделі — біграма "не <слово>".
NEGATION_TEMPLATES = [
    "Товар зовсім не {w}.",
    "Пристрій зовсім не {w}, здивований результатом.",
    "Всупереч відгукам, ноутбук не {w}.",
    "Чесно кажучи, зовсім не {w}, брав би ще.",
]

POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"


@dataclass
class Example:
    text: str
    label: str
    kind: str  # "direct" | "negation"


def _direct(word: str, label: str, i: int) -> Example:
    return Example(DIRECT_TEMPLATES[i % len(DIRECT_TEMPLATES)].format(w=word), label, "direct")


def _negation(word: str, label: str, i: int) -> Example:
    return Example(NEGATION_TEMPLATES[i % len(NEGATION_TEMPLATES)].format(w=word), label, "negation")


def _direct_pool(n_per_class: int) -> list[Example]:
    rows: list[Example] = []
    for i in range(n_per_class):
        rows.append(_direct(POS_WORDS[i % len(POS_WORDS)], POSITIVE, i))
        rows.append(_direct(NEG_WORDS[i % len(NEG_WORDS)], NEGATIVE, i))
    return rows


def _negation_pool(n_per_class: int) -> list[Example]:
    rows: list[Example] = []
    for i in range(n_per_class):
        # "не <негативне>" => POSITIVE
        rows.append(_negation(NEG_WORDS[i % len(NEG_WORDS)], POSITIVE, i))
        # "не <позитивне>" => NEGATIVE
        rows.append(_negation(POS_WORDS[i % len(POS_WORDS)], NEGATIVE, i))
    return rows


def build_splits() -> dict[str, list[Example]]:
    """
    train_v1 — 200 прямих прикладів;
    train_v2 — ті самі 200 прямих + 240 із запереченням;
    eval     — 40 прямих (20/20) + 20 із запереченням (10/10).
    """
    rng = random.Random(SEED)

    direct = _direct_pool(130)      # 260 прямих
    negation = _negation_pool(130)  # 260 із запереченням

    # Holdout формуємо ДО перемішування train-частини й строго збалансовано.
    eval_direct = direct[240:260]        # 20: чергуються POS/NEG => 10/10
    eval_negation = negation[240:260]    # 20: чергуються POS/NEG => 10/10

    train_direct = direct[:200]
    train_negation = negation[:240]
    rng.shuffle(train_direct)
    rng.shuffle(train_negation)

    return {
        "train_v1": train_direct,
        "train_v2": train_direct + train_negation,
        "eval": eval_direct + eval_negation,
    }
