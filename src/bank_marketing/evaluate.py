"""Evaluation automatisee et validation du modele.

Seance 11 - TP Tests Donnees & Modele
    `mlflow.models.evaluate` calcule en une passe un ensemble de metriques et
    d'artefacts (matrice de confusion, courbes ROC / precision-rappel) sur un
    jeu d'evaluation. `mlflow.validate_evaluation_results` applique ensuite une
    porte qualite : le modele est rejete (exception) si une metrique passe sous
    son seuil.
    Pre-requis : un modele enregistre au Model Registry (Seances 5-6).

Le jeu d'evaluation est logue comme dataset MLflow (tracabilite).

Lancement :
    python -m bank_marketing.evaluate                       # derniere version du registry
    python -m bank_marketing.evaluate --model-uri models:/classifier/1
    python -m bank_marketing.evaluate --no-validate         # evalue sans porte qualite
"""

from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.models
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold

from bank_marketing.config import (
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    MODEL_NAME,
    TARGET,
)
from bank_marketing.data import load_data, split
from bank_marketing.tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def latest_model_uri() -> str:
    """Resoudre l'URI de la derniere version enregistree de ``MODEL_NAME``.

    Returns
    -------
    str
        URI MLflow de la forme ``models:/<MODEL_NAME>/<version>``.
    """
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"Aucune version enregistree pour '{MODEL_NAME}'. "
            "Lancez d'abord un entrainement (make train)."
        )
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{MODEL_NAME}/{latest.version}"


def build_thresholds() -> dict[str, MetricThreshold]:
    """Construire les seuils de validation a partir de la configuration.

    A implementer : retourner un dictionnaire {nom_metrique: MetricThreshold}.
    Les noms de metriques produits par l'evaluateur "classifier" sont
    notamment `roc_auc` et `f1_score`. Exemple de seuil :
        MetricThreshold(threshold=EVAL_ROC_AUC_MIN, greater_is_better=True)

    Returns
    -------
    dict of str to MetricThreshold
        Seuils minimaux sur ``roc_auc`` et ``f1_score``.
    """
    return {
        "roc_auc": MetricThreshold(threshold=EVAL_ROC_AUC_MIN, greater_is_better=True),
        "f1_score": MetricThreshold(threshold=EVAL_F1_MIN, greater_is_better=True),
    }


def evaluate_model(model_uri: str | None = None, validate: bool = True):
    """Evaluer un modele du registry et, optionnellement, valider les seuils.

    Parameters
    ----------
    model_uri : str, optional
        URI MLflow du modele a evaluer. Par defaut, la derniere version
        enregistree de ``MODEL_NAME`` (via ``latest_model_uri``).
    validate : bool, optional
        Applique la porte qualite, par defaut True. Leve une exception si un
        seuil n'est pas atteint.

    Returns
    -------
    mlflow.models.EvaluationResult
        Resultat de l'evaluation (metriques et artefacts).
    """
    df = load_data()
    _, x_test, _, y_test = split(df)
    # mlflow.evaluate attend un seul DataFrame contenant features + cible.
    eval_df = x_test.copy()
    eval_df[TARGET] = y_test.values

    # Configuration du tracking centralisee (comme train_models / train_optuna).
    setup_experiment()
    model_uri = model_uri or latest_model_uri()
    logger.info("Evaluation de %s", model_uri)

    with mlflow.start_run(run_name="evaluate"):
        # a) tracabilite : logger le jeu d'evaluation comme dataset MLflow
        log_dataset(eval_df, context="evaluation", name="eval")

        # b) evaluation en une passe (metriques + artefacts)
        result = mlflow.models.evaluate(
            model_uri,
            data=eval_df,
            targets=TARGET,
            model_type="classifier",
            evaluators=["default"],
        )
        logger.info(
            "f1_score=%.3f roc_auc=%.3f",
            result.metrics["f1_score"],
            result.metrics["roc_auc"],
        )

        # c) porte qualite : leve une exception si un seuil n'est pas atteint
        if validate:
            mlflow.validate_evaluation_results(build_thresholds(), result)

        return result


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-uri",
        default=None,
        help="URI du modele a evaluer (defaut: derniere version de MODEL_NAME)",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Evalue sans appliquer la porte qualite (seuils)",
    )
    args = parser.parse_args()

    model_uri = args.model_uri or None
    try:
        evaluate_model(model_uri=model_uri, validate=args.validate)
    except MlflowException as exc:
        logger.error("Validation echouee : %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
