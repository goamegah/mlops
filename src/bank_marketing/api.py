"""API d'inference du modele de classification (FastAPI).

Seance 12 - TP FastAPI (+ journal de predictions en base)
    L'API charge le modele au demarrage (lifespan) et expose :
      - /health, /model-info
      - POST /predict      : inference + journalisation en base (renvoie un id)
      - POST /feedback     : verite terrain rattachee a une prediction
      - GET  /predictions  : journal des dernieres predictions

    Chargement du modele (best practice SaaS) :
    - Alias MLflow @prod (production model)
    - Fallback sur @staging si @prod inexistant
    - Fallback sur fichier joblib local si MLflow indisponible

    Lancement : `uvicorn bank_marketing.api:app --reload`  (ou `make api`)
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bank_marketing.config import MODEL_DIR, MODEL_NAME, MLFLOW_TRACKING_URI
from bank_marketing.db import init_db, list_predictions, save_feedback, save_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


def _load_model_from_mlflow() -> object | None:
    """Charge le modele depuis MLflow par alias (@prod, fallback @staging).
    Retourne None si MLflow est indisponible ou alias inexistant.
    Timeout court pour eviter les blocages au startup."""
    try:
        import mlflow

        # Timeout court + pas de retry (deja configure en env vars)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Essayer @prod, puis @staging
        for alias in ["prod", "staging"]:
            try:
                model_uri = f"models:/{MODEL_NAME}@{alias}"
                # Timeout court : fail vite si MLflow indisponible
                model = mlflow.sklearn.load_model(model_uri)
                logger.info("Modele charge depuis MLflow : %s", model_uri)
                return model
            except Exception as e:
                logger.debug("Alias @%s indisponible : %s", alias, e)
                continue
        logger.warning("Aucun alias MLflow (@prod/@staging) disponible")
        return None
    except Exception as e:
        logger.warning("MLflow indisponible, fallback sur joblib : %s", e)
        return None


def _load_model_fallback() -> object:
    """Charge le modele depuis le fichier joblib local (fallback)."""
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modele non trouve : {model_path}")
    model = joblib.load(model_path)
    logger.info("Modele charge depuis fichier (fallback) : %s", model_path)
    return model


# S12-3 : charge le modele une seule fois au demarrage (best practice SaaS).
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Charger le modele : MLflow (prod/staging) > fallback joblib > None
    model = _load_model_from_mlflow()
    model_source = "mlflow" if model is not None else "joblib"

    if model is None:
        logger.warning("MLflow indisponible, tentative fallback sur modele local")
        try:
            model = _load_model_fallback()
            logger.info("Modele charge depuis fichier local")
        except FileNotFoundError:
            logger.warning("Modele local inexistant : l'API demarre sans modele (en attente du DAG training)")
            model = None
            model_source = "unavailable"

    ml["model"] = model
    ml["model_source"] = model_source

    # Journal de predictions : best-effort, l'API demarre meme sans la base.
    try:
        init_db()
        logger.info("Base de donnees prete (journal de predictions)")
    except Exception:
        logger.warning(
            "Base indisponible : le journal de predictions sera desactive", exc_info=True
        )
    yield
    ml.clear()


app = FastAPI(title="Bank Marketing API", version="0.1.0", lifespan=lifespan)


class Features(BaseModel):
    """Profil d'un client a scorer (colonnes du dataset Bank Marketing)."""

    # Numeriques
    age: int = Field(..., ge=18, le=120, description="Age du client")
    balance: int = Field(..., description="Solde annuel moyen en euros (peut etre negatif)")
    day: int = Field(..., ge=1, le=31, description="Jour du mois du dernier contact")
    campaign: int = Field(..., ge=1, description="Nombre de contacts durant cette campagne")
    pdays: int = Field(..., description="Jours depuis le dernier contact (-1 = jamais contacte)")
    previous: int = Field(..., ge=0, description="Nombre de contacts avant cette campagne")
    # Categorielles
    job: str = Field(..., description="Type d'emploi")
    marital: str = Field(..., description="Statut marital")
    education: str = Field(..., description="Niveau d'etudes")
    default: str = Field(..., description="Defaut de credit (yes/no)")
    housing: str = Field(..., description="Pret immobilier (yes/no)")
    loan: str = Field(..., description="Pret personnel (yes/no)")
    contact: str = Field(..., description="Type de contact")
    month: str = Field(..., description="Mois du dernier contact")
    poutcome: str = Field(..., description="Issue de la campagne precedente")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionOut(BaseModel):
    """Resultat de prediction (l'id permet de rattacher un feedback)."""

    id: str = Field(..., description="Identifiant de la prediction (vide si non journalisee)")
    prediction: int = Field(..., description="Classe predite : 1 = souscrit, 0 = ne souscrit pas")
    probability: float = Field(..., description="Probabilite de souscription (classe 1)")


class FeedbackIn(BaseModel):
    """Verite terrain pour une prediction passee."""

    prediction_id: str = Field(..., description="Identifiant renvoye par /predict")
    actual: int = Field(..., ge=0, le=1, description="Resultat reel : 1 = a souscrit, 0 = non")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    """Infos sur le modele charge (source, version, alias)."""
    model_version = os.environ.get("MODEL_VERSION", "unknown")
    model_source = ml.get("model_source", "unknown")
    return {
        "model_name": MODEL_NAME,
        "model_source": model_source,  # "mlflow" ou "joblib"
        "model_version": model_version,
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    row = pd.DataFrame([features.model_dump()])
    proba = round(float(model.predict_proba(row)[0, 1]), 4)
    pred = int(proba >= 0.5)
    model_version = os.environ.get("MODEL_VERSION", "unknown")

    # Journalisation best-effort : une base indisponible ne casse pas /predict.
    pred_id = ""
    try:
        pred_id = save_prediction(features.model_dump(), pred, proba, model_version)
    except Exception:
        logger.warning(
            "Echec d'enregistrement de la prediction (base indisponible ?)", exc_info=True
        )

    return PredictionOut(id=pred_id, prediction=pred, probability=proba)


@app.post("/feedback")
def feedback(fb: FeedbackIn) -> dict:
    try:
        feedback_id = save_feedback(fb.prediction_id, fb.actual)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Feedback non enregistre : {exc}") from exc
    return {"status": "ok", "feedback_id": feedback_id}


@app.get("/predictions")
def predictions(limit: int = 50) -> list[dict]:
    try:
        return list_predictions(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Journal indisponible : {exc}") from exc
