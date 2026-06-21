"""Client de test pour l'API FastAPI du modele.

Seance 15 - TP Tests de l'API
    Envoie quelques payloads de test a une instance locale de l'API
    (`make api`) et affiche les reponses de `/health`, `/predict` et
    `/model-info`.

    Les payloads sont echantillonnes dans votre jeu de donnees, donc valides
    quel que soit votre dataset (les colonnes envoyees sont vos features).

Lancement (depuis la racine du projet, l'API doit tourner via `make api`) :
    make predict-client
    PYTHONPATH=src python scripts/predict_client.py --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import logging

import httpx

from bank_marketing.config import API_URL, TARGET
from bank_marketing.data import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Nombre de clients de test envoyes a l'API.
N_SAMPLES = 3


def build_payloads(n: int = N_SAMPLES) -> list[dict]:
    """Construire n payloads de test a partir du jeu de donnees.

    On retire la colonne cible et on convertit chaque ligne en dict JSON natif.
    """
    features = load_data().drop(columns=[TARGET])
    sample = features.sample(n=n)
    return [json.loads(row.to_json()) for _, row in sample.iterrows()]


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=API_URL, help="URL de base de l'API (defaut: %(default)s)")
    args = parser.parse_args()

    payloads = build_payloads()

    try:
        with httpx.Client(base_url=args.url, timeout=10.0) as client:
            health = client.get("/health")
            logger.info("GET /health -> %s %s", health.status_code, health.json())

            for i, payload in enumerate(payloads):
                response = client.post("/predict", json=payload)
                logger.info(
                    "POST /predict (#%d) -> %s %s", i, response.status_code, response.json()
                )

            info = client.get("/model-info")
            logger.info("GET /model-info -> %s %s", info.status_code, info.json())
    except httpx.ConnectError:
        logger.error(
            "API injoignable sur %s -- demarre-la d'abord dans un autre terminal : make api",
            args.url,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
