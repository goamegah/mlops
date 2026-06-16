"""Preparation des donnees Bank Marketing (UCI) -> data/dataset.csv.

Telecharge le dataset Bank Marketing depuis le UCI ML Repository (sans
authentification), en extrait ``bank-full.csv``, mappe la cible ``y``
(yes/no -> 1/0) et retire la colonne ``duration`` (fuite de donnees : connue
seulement APRES l'appel, donc inutilisable pour predire avant d'appeler). Le
resultat est un CSV propre, separe par des virgules, ecrit dans ``data/``.

Ce module sert a la fois au ``make data``, a la baseline (``bank_marketing.train``
lit ``data/dataset.csv``) et au DAG Airflow de re-entrainement (Seance 17).

Usage:
    python -m bank_marketing.prepare_data    # depuis la racine (PYTHONPATH=.)
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from bank_marketing.config import DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Archive publique UCI (dataset 222). Elle imbrique un `bank.zip` qui contient
# `bank-full.csv` (45 211 lignes, 16 variables + la cible `y`).
UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

# `duration` n'est connue qu'apres l'appel -> fuite de donnees, on l'exclut.
LEAKAGE_COLUMNS = ["duration"]


def _download_bank_full() -> pd.DataFrame:
    """Telecharger et lire ``bank-full.csv`` depuis l'archive imbriquee UCI."""
    logger.info("Telechargement du dataset UCI : %s", UCI_URL)
    with urlopen(UCI_URL, timeout=60) as response:  # noqa: S310 (URL UCI figee)
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    with outer.open("bank.zip") as inner_bytes:
        inner = zipfile.ZipFile(io.BytesIO(inner_bytes.read()))
    with inner.open("bank-full.csv") as csv_file:
        # Le CSV d'origine est separe par des points-virgules.
        return pd.read_csv(csv_file, sep=";")


def prepare(output_path: Path = DATA_PATH) -> pd.DataFrame:
    """Construire le jeu de donnees nettoye et l'ecrire dans ``output_path``.

    Returns
    -------
    pandas.DataFrame
        Le jeu de donnees prepare (cible `y` en 0/1, sans `duration`).
    """
    df = _download_bank_full()
    logger.info("Donnees brutes : %d lignes, %d colonnes", *df.shape)

    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])
    df["y"] = (df["y"].astype(str).str.strip().str.lower() == "yes").astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        "Ecrit %s (%d lignes, taux de positifs : %.1f%%)",
        output_path,
        len(df),
        100 * df["y"].mean(),
    )
    return df


def main() -> None:
    """Point d'entree en ligne de commande."""
    prepare()


if __name__ == "__main__":
    main()
