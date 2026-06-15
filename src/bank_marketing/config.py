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

# Alias de modèles pour les differents stades de production (dev, staging, prod)
MODEL_STAGES = ["dev", "staging", "prod"]
