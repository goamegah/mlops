"""Configuration partagee du suivi MLflow (squelette).

Seance 5 - TP MLflow Tracking (suite)
    Centralise la configuration du tracking pour eviter de la dupliquer dans
    chaque script d'entrainement, et ajoute la tracabilite des donnees
    (dataset lineage). Completez les TODO (S5-8, S5-9).

    Une fois ces fonctions implementees, les scripts (train, train_models,
    train_optuna, evaluate) peuvent appeler `setup_experiment()` au lieu de
    repeter `set_tracking_uri` + `set_experiment`.
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.data
import pandas as pd

from bank_marketing.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    TARGET,
)

logger = logging.getLogger(__name__)


def setup_experiment() -> None:
    """Configurer le tracking MLflow et les metadonnees de l'experience.

    A implementer :
    - pointer vers `MLFLOW_TRACKING_URI` (mlflow.set_tracking_uri)
    - selectionner (ou creer) l'experience `MLFLOW_EXPERIMENT`
      (mlflow.set_experiment, qui renvoie un objet Experiment)
    - appliquer la description (tag special `mlflow.note.content`) et les tags
      `MLFLOW_EXPERIMENT_TAGS` a l'experience, via un `mlflow.MlflowClient()`
      et `client.set_experiment_tag(experiment_id, key, value)`.

    L'operation doit etre idempotente (re-appelable sans erreur).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    client = mlflow.MlflowClient()
    if MLFLOW_EXPERIMENT_DESCRIPTION:
        client.set_experiment_tag(
            experiment.experiment_id, "mlflow.note.content", MLFLOW_EXPERIMENT_DESCRIPTION
        )
    for key, value in MLFLOW_EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment.experiment_id, key, value)
    logger.info("Suivi MLflow : %s (experience: %s)", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT)


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    """Logger un dataset MLflow dans le run courant (tracabilite donnees -> modele).

    Rattache au run la source des donnees, le schema et un profil, visibles
    dans l'onglet "Datasets" de l'UI MLflow.

    A implementer :
    - construire un dataset MLflow a partir du DataFrame :
      `mlflow.data.from_pandas(df, source=str(DATA_PATH), targets=TARGET, name=name)`
    - le logger dans le run courant avec `mlflow.log_input(dataset, context=context)`

    Parameters
    ----------
    df : pandas.DataFrame
        Donnees a referencer (features + cible).
    context : str
        Role du dataset dans le run, par exemple "training" ou "evaluation".
    name : str, optional
        Nom logique du dataset, par defaut "dataset".
    """
    # mlflow.data.from_pandas est resolu dynamiquement -> invisible pour mypy.
    dataset = mlflow.data.from_pandas(  # type: ignore[attr-defined]
        df, source=str(DATA_PATH), targets=TARGET, name=name
    )
    mlflow.log_input(dataset, context=context)
