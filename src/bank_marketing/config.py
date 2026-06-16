"""Configuration centrale du projet de classification.

C'est le SEUL fichier a adapter pour brancher votre propre jeu de donnees :
data.py, features.py et les scripts d'entrainement lisent toutes leurs
colonnes via ces constantes (TP S0).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Layout src/ : ce fichier est src/bank_marketing/config.py, donc la racine du
# projet est parents[2] (bank_marketing/ -> src/ -> racine).
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# S0-1 : chemin vers le fichier de donnees genere par bank_marketing.prepare_data
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_DIR = ROOT / "models"

# S0-2 : cible binaire Bank Marketing -> le client souscrit un depot a terme (1) ou non (0)
TARGET = "y"

# S0-3 : colonnes numeriques (la fuite `duration` est retiree en preparation)
NUMERIC_FEATURES: list[str] = ["age", "balance", "day", "campaign", "pdays", "previous"]

# S0-4 : colonnes categorielles
CATEGORICAL_FEATURES: list[str] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

RANDOM_STATE = 42

# Surcouche via variables d'environnement (principe 12-factor)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "bank-marketing-baseline")
MODEL_NAME = os.getenv("MODEL_NAME", "bank-marketing-classifier")

# URL de l'API FastAPI (utilisee par le client de test scripts/predict_client.py)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Metadonnees de l'experience MLflow (lues par bank_marketing.tracking pour
# documenter l'experience dans l'UI : description + tags).
MLFLOW_EXPERIMENT_DESCRIPTION = os.getenv(
    "MLFLOW_EXPERIMENT_DESCRIPTION", "Bank Marketing - cours MLOps"
)


def _parse_tags(raw: str) -> dict[str, str]:
    """Parser une chaine "cle=valeur,cle2=valeur2" en dictionnaire de tags."""
    tags: dict[str, str] = {}
    for item in raw.split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            tags[key.strip()] = value.strip()
    return tags


MLFLOW_EXPERIMENT_TAGS = _parse_tags(os.getenv("MLFLOW_EXPERIMENT_TAGS", ""))

# Seuils de la porte qualite (bank_marketing.evaluate) : le modele est rejete si
# une metrique passe sous son seuil. f1 volontairement bas car les classes sont
# desequilibrees (~11,7% de positifs) -> au seuil de decision 0.5 le f1 tourne
# autour de 0.33, bien en dessous d'un 0.55 generique.
EVAL_ROC_AUC_MIN = float(os.getenv("EVAL_ROC_AUC_MIN", "0.65"))
EVAL_F1_MIN = float(os.getenv("EVAL_F1_MIN", "0.30"))

# Alias de modèles pour les differents stades de production (dev, staging, prod)
MODEL_STAGES = ["dev", "staging", "prod"]
