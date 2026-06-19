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

# Les seuils de la porte qualite (roc_auc ET f1) viennent de bank_marketing.config
# (EVAL_ROC_AUC_MIN=0.65, EVAL_F1_MIN=0.30) : MEME source que la porte qualite du
# frontend (bank_marketing.evaluate), pour rester coherent dans tout le projet.
# Le f1 est volontairement bas (0.30) car le dataset est desequilibre (~12% de
# positifs) : au seuil 0.5 le modele fait ~0.34. Le roc_auc (robuste au
# desequilibre) fait ~0.80. Un bon modele passe les deux -> DAG VERT ; une
# regression sous l'un des seuils bloque -> DAG ROUGE. Importes dans la tache
# (et non au niveau module) pour ne pas casser le parsing du DAG si le package
# n'est pas importable au moment du chargement.

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
    # S17-2 : entraine avec Optuna (hyperparameter tuning) et pousse f1 + roc_auc dans XCom pour la tache suivante.
    from bank_marketing.train_optuna import train_optuna

    metrics = train_optuna()
    logger.info("Entrainement Optuna termine : f1=%.3f roc_auc=%.3f", metrics["f1"], metrics["roc_auc"])
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    context["ti"].xcom_push(key="roc_auc", value=metrics["roc_auc"])


def task_check_quality(**context) -> None:
    # S17-3 : porte qualite - echoue (et marque le DAG en echec) si roc_auc OU f1 passe sous son seuil.
    # Seuils lus depuis bank_marketing.config (meme source que la porte du frontend).
    from bank_marketing.config import EVAL_F1_MIN, EVAL_ROC_AUC_MIN

    roc_auc = context["ti"].xcom_pull(task_ids="train", key="roc_auc")
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if roc_auc is None or f1 is None:
        raise ValueError("Metriques absentes dans XCom : la tache 'train' a-t-elle reussi ?")
    if roc_auc < EVAL_ROC_AUC_MIN:
        raise ValueError(
            f"Qualite insuffisante : roc_auc={roc_auc:.3f} < seuil={EVAL_ROC_AUC_MIN} -> modele rejete."
        )
    if f1 < EVAL_F1_MIN:
        raise ValueError(
            f"Qualite insuffisante : f1={f1:.3f} < seuil={EVAL_F1_MIN} -> modele rejete."
        )
    logger.info(
        "Controle qualite OK : roc_auc=%.3f (>=%.2f) et f1=%.3f (>=%.2f)",
        roc_auc, EVAL_ROC_AUC_MIN, f1, EVAL_F1_MIN,
    )


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
