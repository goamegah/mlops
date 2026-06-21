"""Fixtures partagees de la suite de tests (donnees, modele, API).

Les tests sont hermetiques : aucun service externe (MLflow, MySQL) n'est requis.
Le client d'API injecte un modele factice et neutralise la base de donnees.
"""

from __future__ import annotations

import numpy as np
import pytest

from bank_marketing.config import DATA_PATH
from bank_marketing.data import load_data, split
from bank_marketing.features import build_preprocessor


@pytest.fixture(scope="session")
def dataset():
    """Jeu de donnees prepare (data/dataset.csv) ; le prepare si absent."""
    if not DATA_PATH.exists():
        from bank_marketing.prepare_data import prepare

        prepare()
    return load_data()


@pytest.fixture(scope="session")
def trained_pipeline(dataset):
    """Pipeline baseline (preprocessing + LogisticRegression) entraine une fois.

    Retourne ``(pipeline, x_test, y_test)`` pour les tests modele.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    x_train, x_test, y_train, y_test = split(dataset)
    pipe = Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000))]
    )
    pipe.fit(x_train, y_train)
    return pipe, x_test, y_test


class _DummyModel:
    """Modele factice deterministe : proba=0.7 pour la classe positive."""

    def predict_proba(self, x):
        return np.tile([0.3, 0.7], (len(x), 1))


@pytest.fixture
def api_client(monkeypatch):
    """Client de test FastAPI hermetique (modele factice, sans MLflow ni MySQL)."""
    from fastapi.testclient import TestClient

    from bank_marketing import api

    dummy = _DummyModel()
    # Neutralise toute connexion externe au demarrage (lifespan) et en /predict.
    monkeypatch.setattr(api, "_load_model_from_mlflow", lambda: (dummy, "test", "test"))
    monkeypatch.setattr(api, "init_db", lambda: None)
    monkeypatch.setattr(api, "save_prediction", lambda *a, **k: "test-id")

    with TestClient(api.app) as client:
        yield client
