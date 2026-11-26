"""
Робота з MLflow Model Registry: реєстрація версії, просування в
'production', відкат до попередньої версії, читання метаданих активної
версії.

Просування й відкат — через ДВА псевдоніми (alias):
    production — версія, що зараз обслуговує трафік;
    previous   — версія, що була в проді до останнього просування.
Відкат = перестановка цих двох вказівників (підрозділ 3.4 методички).

TODO: усі чотири функції — крок 5.
"""


from __future__ import annotations

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from app.config import Settings

settings = Settings()

PREVIOUS_ALIAS = "previous"


def _client(
    tracking_uri: str | None = None,
) -> MlflowClient:

    uri = tracking_uri or settings.tracking_uri

    return MlflowClient(
        tracking_uri=uri,
        registry_uri=uri,
    )


def register_model(
    run_id: str,
    model_name: str | None = None,
    tracking_uri: str | None = None,
) -> int:

    name = model_name or settings.registry_model_name
    client = _client(tracking_uri)

    try:
        client.create_registered_model(name)

    except MlflowException as e:
        if e.error_code != "RESOURCE_ALREADY_EXISTS":
            raise

    model_version = client.create_model_version(
        name=name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )

    return int(model_version.version)


def promote_to_production(
    version: int,
    model_name: str | None = None,
    tracking_uri: str | None = None,
    production_alias: str | None = None,
) -> None:

    name = model_name or settings.registry_model_name
    alias = production_alias or settings.production_alias

    client = _client(tracking_uri)

    try:
        current = client.get_model_version_by_alias(
            name,
            alias,
        )

    except MlflowException as e:

        if e.error_code not in (
            "RESOURCE_DOES_NOT_EXIST",
            "INVALID_PARAMETER_VALUE",
        ):
            raise

    else:
        client.set_registered_model_alias(
            name,
            PREVIOUS_ALIAS,
            current.version,
        )

    client.set_registered_model_alias(
        name,
        alias,
        str(version),
    )


def rollback_to_previous(
    model_name: str | None = None,
    tracking_uri: str | None = None,
    production_alias: str | None = None,
) -> int:

    name = model_name or settings.registry_model_name
    alias = production_alias or settings.production_alias

    client = _client(tracking_uri)

    current = client.get_model_version_by_alias(
        name,
        alias,
    )

    previous = client.get_model_version_by_alias(
        name,
        PREVIOUS_ALIAS,
    )

    client.set_registered_model_alias(
        name,
        alias,
        previous.version,
    )

    client.set_registered_model_alias(
        name,
        PREVIOUS_ALIAS,
        current.version,
    )

    return int(previous.version)


def get_production_info(
    model_name: str | None = None,
    tracking_uri: str | None = None,
    production_alias: str | None = None,
) -> dict:

    name = model_name or settings.registry_model_name
    alias = production_alias or settings.production_alias

    client = _client(tracking_uri)

    version = client.get_model_version_by_alias(
        name,
        alias,
    )

    run = client.get_run(version.run_id)

    return {
        "name": name,
        "version": int(version.version),
        "run_id": version.run_id,
        "accuracy": run.data.metrics.get("accuracy"),
        "accuracy_hard": run.data.metrics.get(
            "accuracy_hard"
        ),
    }
