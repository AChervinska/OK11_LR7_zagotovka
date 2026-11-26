
from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import Settings

settings = Settings()


class TfidfLogRegClassifier:

    def __init__(
        self,
        ngram_max: int | None = None,
        C: float | None = None,
    ) -> None:

        self.ngram_max = (
            ngram_max
            if ngram_max is not None
            else settings.tfidf_ngram_max
        )

        self.C = (
            C if C is not None else settings.logreg_c
        )

        self.pipeline: Pipeline | None = None

    def train(self, train_df: pd.DataFrame) -> None:

        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, self.ngram_max)
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=self.C,
                    max_iter=1000,
                    random_state=0,
                ),
            ),
        ])

        self.pipeline.fit(
            train_df["text"],
            train_df["label"],
        )

    def predict(self, text: str) -> tuple[str, float]:

        if self.pipeline is None:
            raise RuntimeError("Модель ще не навчена")

        label = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]

        return str(label), float(max(probabilities))

    def predict_labels(self, texts) -> pd.Series:

        return pd.Series(
            self.pipeline.predict(list(texts))
        )

    @property
    def is_ready(self) -> bool:

        return self.pipeline is not None

    @classmethod
    def from_pipeline(
        cls,
        pipeline: Pipeline,
    ) -> "TfidfLogRegClassifier":

        obj = cls()
        obj.pipeline = pipeline

        return obj


def load_from_registry(
    model_name: str | None = None,
    tracking_uri: str | None = None,
    alias: str | None = None,
) -> tuple[TfidfLogRegClassifier, dict]:

    from app.registry import get_production_info

    name = model_name or settings.registry_model_name
    uri = tracking_uri or settings.tracking_uri
    alias = alias or settings.production_alias

    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)

    pipeline = mlflow.sklearn.load_model(
        f"models:/{name}@{alias}"
    )

    info = get_production_info(
        name,
        uri,
        alias,
    )

    return TfidfLogRegClassifier.from_pipeline(pipeline), info


def load_pipeline_from_path(
    path: str | Path,
) -> TfidfLogRegClassifier:

    pipeline = mlflow.sklearn.load_model(str(path))

    return TfidfLogRegClassifier.from_pipeline(pipeline)
