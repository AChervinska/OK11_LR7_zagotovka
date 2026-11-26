"""
Pydantic-схеми контракту сервісу інференсу ЛР7.

Контракт /predict успадкований із ЛР1 без змін (POSITIVE/NEGATIVE + score).
Новий ендпоінт /model/info повертає, ЯКА САМЕ версія моделі зараз
обслуговує запити — це і є видима ззовні ознака того, що сервіс
підключений до реєстру моделей, а не до жорстко зашитого файлу.

Цей файл видається готовим — редагувати не потрібно.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Вхідні дані запиту класифікації тональності (як у ЛР1)."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Текст природною мовою для класифікації тональності.",
        examples=["Товар зовсім не жахливий, приємно здивований."],
    )


class PredictResponse(BaseModel):
    """Відповідь сервісу інференсу (як у ЛР1)."""

    label: str = Field(..., description="Мітка класу: 'POSITIVE' або 'NEGATIVE'.")
    score: float = Field(..., ge=0.0, le=1.0, description="Впевненість моделі у мітці, [0, 1].")
    model_version: int = Field(..., ge=1, description="Версія моделі з реєстру, що обробила запит.")


class HealthResponse(BaseModel):
    """Відповідь healthcheck-ендпоінта."""

    status: str = Field(..., description="'ok', якщо модель завантажена з реєстру і готова.")


class ModelInfoResponse(BaseModel):
    """
    Метадані активної версії моделі — джерело правди для питання
    «що саме зараз у проді».
    """

    name: str = Field(..., description="Ім'я зареєстрованої моделі в Model Registry.")
    version: int = Field(..., ge=1, description="Номер версії, на яку вказує alias 'production'.")
    run_id: str = Field(..., description="MLflow run_id експерименту, з якого походить ця версія.")
    accuracy: float | None = Field(None, description="Загальна accuracy на holdout (з метрик run).")
    accuracy_hard: float | None = Field(
        None, description="Accuracy на прикладах із запереченням (з метрик run)."
    )
