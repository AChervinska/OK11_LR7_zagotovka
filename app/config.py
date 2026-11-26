"""
Конфігурація через змінні середовища — той самий підхід, що в ЛР1-ЛР6
(pydantic-settings, префікс OK11_, файл .env).

Цей файл видається готовим — редагувати не потрібно.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OK11_", env_file=".env", extra="ignore")

    # --- Дані ---
    data_dir: str = "data"

    # --- MLflow ---
    # Бекенд зберігання трекінгу й реєстру моделей. sqlite обрано навмисно:
    # Model Registry (реєстр версій) вимагає реляційного бекенду, файловий
    # (./mlruns) для реєстру не підходить. Файл mlflow.db створюється в корені
    # репозиторію при першому запуску.
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "ok11-lr7-sentiment"

    # Ім'я зареєстрованої моделі в Model Registry. Сервіс інференсу
    # завантажує модель за псевдонімом (alias) "production" цього імені,
    # а не за шляхом до файлу (крок 6, підрозділ 3.5 методички).
    registry_model_name: str = "ok11-sentiment"
    production_alias: str = "production"

    # --- Гіперпараметри моделі (дублюються в params.yaml для DVC-пайплайну,
    #     поглиблений рівень; тут — значення за замовчуванням для базового) ---
    tfidf_ngram_max: int = 2
    logreg_c: float = 1.0

    # --- Назви класів — узгоджені з сервісом із ЛР1 (POSITIVE/NEGATIVE) ---
    positive_label: str = "POSITIVE"
    negative_label: str = "NEGATIVE"
