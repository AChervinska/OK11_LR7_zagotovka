"""Тести app/main.py — крок 6 (сервіс віддає версію з реєстру)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.main as main_module
from app.model import TfidfLogRegClassifier


def _no_registry(*args, **kwargs):
    raise RuntimeError("у тесті реєстр не піднімається")


@pytest.fixture
def client(splits, monkeypatch):
    # lifespan не має ходити в реальний реєстр під час тесту
    monkeypatch.setattr(main_module, "load_from_registry", _no_registry)

    clf = TfidfLogRegClassifier()
    clf.train(splits["train_v2"])
    monkeypatch.setitem(main_module._state, "classifier", clf)
    monkeypatch.setitem(
        main_module._state,
        "info",
        {"name": "ok11-sentiment", "version": 2, "run_id": "deadbeef",
         "accuracy": 1.0, "accuracy_hard": 1.0},
    )
    with TestClient(main_module.app) as c:
        yield c


def test_health_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info_reports_active_version(client):
    body = client.get("/model/info").json()
    assert body["version"] == 2
    assert body["name"] == "ok11-sentiment"
    assert body["accuracy_hard"] == 1.0


def test_predict_carries_model_version(client):
    resp = client.post("/predict", json={"text": "Товар зовсім не жахливий."})
    body = resp.json()
    assert resp.status_code == 200
    assert body["label"] == "POSITIVE"
    assert body["model_version"] == 2


def test_predict_validates_empty_text(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
