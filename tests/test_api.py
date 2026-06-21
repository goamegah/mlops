"""Tests de l'API FastAPI (Seance 15 - tests d'API).

Couvrent /health, /model-info, /predict (cas nominal, validation, modele absent).
Le client est hermetique (modele factice injecte, base neutralisee) : voir
``tests/conftest.py``.
"""

from __future__ import annotations

from bank_marketing import api

# Profil client valide (memes colonnes que le schema Features de l'API).
VALID_PAYLOAD = {
    "age": 42,
    "balance": 1500,
    "day": 5,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "poutcome": "unknown",
}


def test_health(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info(api_client):
    response = api_client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"]
    assert "model_source" in body
    assert "model_version" in body


def test_predict_returns_valid_prediction(api_client):
    response = api_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_invalid_payload(api_client):
    # age < 18 viole la contrainte Field(ge=18) -> erreur de validation 422.
    bad_payload = {**VALID_PAYLOAD, "age": 5}
    response = api_client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_without_model_returns_503(api_client):
    # Simule une API demarree sans modele charge : /predict doit renvoyer 503.
    api.ml["model"] = None
    response = api_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
