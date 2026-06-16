"""API d'inference du modele de classification (FastAPI).

Seance 12 - TP FastAPI
    L'API charge le modele une seule fois au demarrage (lifespan) et expose
    /health, /predict (inference) et /model-info. Le schema d'entree reprend
    les colonnes du dataset Bank Marketing.
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

from bank_marketing.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


# S12-3 : charge le modele une seule fois au demarrage (pas a chaque requete).
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODEL_DIR / "model.joblib"
    ml["model"] = joblib.load(model_path)
    logger.info("Modele charge depuis %s", model_path)
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
    """Resultat de prediction."""

    prediction: int = Field(..., description="Classe predite : 1 = souscrit, 0 = ne souscrit pas")
    probability: float = Field(..., description="Probabilite de souscription (classe 1)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


@app.get("/model-info")
def model_info() -> dict:
    """Version du modele servie (lue depuis la variable d'environnement MODEL_VERSION)."""
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}
