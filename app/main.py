"""
Сервіс інференсу ЛР7 — той самий контракт /predict, що в ЛР1, але
модель береться НЕ з жорстко зашитого шляху, а з MLflow Model Registry
за псевдонімом 'production'. Ендпоінт /model/info показує, яка саме
версія зараз активна.

TODO: підключення load_from_registry у lifespan — крок 6.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from app.config import Settings
from app.model import TfidfLogRegClassifier, load_from_registry
from app.schemas import HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse

settings = Settings()
logging.basicConfig(level="INFO")
logger = logging.getLogger("ok11.lr7")

# Стан процесу: активний класифікатор і метадані його версії.
_state: dict = {"classifier": None, "info": None}


def get_classifier() -> TfidfLogRegClassifier:
    clf = _state["classifier"]
    if clf is None or not clf.is_ready:
        raise HTTPException(
            status_code=503, detail="Модель не завантажена з реєстру")
    return clf


@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        classifier, info = load_from_registry(
            settings.registry_model_name,
            settings.tracking_uri,
            settings.production_alias,
        )

        _state["classifier"] = classifier
        _state["info"] = info

        logger.info(
            "Завантажено модель %s, версія %s",
            info["name"],
            info["version"],
        )

    except Exception as e:

        logger.warning(
            "Не вдалося завантажити модель: %s",
            e,
        )

    yield


app = FastAPI(
    title="Сервіс класифікації тональності — ЛР7 (модель з Model Registry)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _state["classifier"] is None:
        raise HTTPException(
            status_code=503, detail="Модель не завантажена з реєстру")
    return HealthResponse(status="ok")


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    if _state["info"] is None:
        raise HTTPException(
            status_code=503, detail="Модель не завантажена з реєстру")
    return ModelInfoResponse(**_state["info"])


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest, clf: TfidfLogRegClassifier = Depends(get_classifier)
) -> PredictResponse:
    label, score = clf.predict(request.text)
    return PredictResponse(
        label=label, score=score, model_version=_state["info"]["version"]
    )
