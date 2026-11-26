"""
Політика автоматичного перенавчання — поглиблений рівень, крок 12.

Три незалежні тригери (лекція, тема 9): за розкладом, за подією
(наприклад, стійкий дрейф із ЛР5), за деградацією якості (падіння
метрики нижче порогу — вимагає свіжих міток, тобто циклу ЛР6).

TODO: should_retrain() — крок 12.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrainTriggers:
    scheduled_due: bool = False           # тригер за розкладом
    # тригер за подією (is_persistent_drift з ЛР5)
    drift_persistent: bool = False
    # падіння accuracy відносно базової (0..1), None якщо міток немає
    quality_drop: float | None = None
    min_quality_drop: float = 0.05        # поріг реагування на деградацію


def should_retrain(
    triggers: RetrainTriggers,
) -> tuple[bool, str]:

    if (
        triggers.quality_drop is not None
        and triggers.quality_drop
        >= triggers.min_quality_drop
    ):
        return (
            True,
            f"деградація якості: "
            f"{triggers.quality_drop:.2%}",
        )

    if triggers.drift_persistent:

        return (
            True,
            "стійкий дрейф вхідних даних",
        )

    if triggers.scheduled_due:

        return (
            True,
            "планове перенавчання за розкладом",
        )

    return (
        False,
        "жоден тригер не спрацював",
    )
