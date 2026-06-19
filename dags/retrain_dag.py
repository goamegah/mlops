"""DAG Airflow - pipeline de re-entrainement du modele.

Seance 17 - TP Airflow
    Pipeline simple : preparation des donnees -> entrainement -> controle
    qualite. Le DAG importe le package `bank_marketing` (rendu importable via
    PYTHONPATH=/opt/airflow/src dans le conteneur Airflow) et reutilise les
    memes fonctions que la ligne de commande (`make data`, `make train`).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# f1 minimal du modele entraine pour que le pipeline soit considere comme reussi.
QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    # S17-1 : (re)telecharge et prepare data/dataset.csv (meme code que `make data`).
    from bank_marketing.prepare_data import prepare

    df = prepare()
    logger.info("Donnees pretes : %d lignes", len(df))


def task_train(**context) -> None:
    # S17-2 : entraine avec Optuna (hyperparameter tuning) et pousse le f1 dans XCom pour la tache suivante.
    from bank_marketing.train_optuna import train_optuna

    metrics = train_optuna()
    logger.info("Entrainement Optuna termine : f1=%.3f roc_auc=%.3f", metrics["f1"], metrics["roc_auc"])
    context["ti"].xcom_push(key="f1", value=metrics["f1"])


def task_check_quality(**context) -> None:
    # S17-3 : porte qualite - echoue (et marque le DAG en echec) si f1 trop bas.
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 is None:
        raise ValueError("Aucun f1 dans XCom : la tache 'train' a-t-elle reussi ?")
    if f1 < QUALITY_THRESHOLD:
        raise ValueError(
            f"Qualite insuffisante : f1={f1:.3f} < seuil={QUALITY_THRESHOLD} -> modele rejete."
        )
    logger.info("Controle qualite OK : f1=%.3f >= seuil=%.3f", f1, QUALITY_THRESHOLD)


with DAG(
    dag_id="model_retraining",
    description="Prepare les donnees, reentraine le modele et controle sa qualite",
    # S17-4 : planning hebdomadaire (tous les lundis a 3h du matin).
    schedule="0 3 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    # S17-5 : ordre d'execution lineaire.
    prepare >> train_task >> check
