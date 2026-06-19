"""DAG Airflow - pipeline de prediction par lots via l'API de serving.

Seance 17 - TP Airflow (prediction)
    Pipeline : echantillonne des clients du dataset -> appelle l'API FastAPI
    POST /predict pour chacun -> resume les resultats. Demontre l'orchestration
    d'un service de serving deja deploye (service `api`, port 8000). L'API
    journalise elle-meme chaque prediction en base (table `predictions`), donc
    ce DAG n'ecrit rien directement : il consomme le service comme un client.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# Nombre de clients echantillonnes et envoyes a l'API par execution.
N_SAMPLES = 5

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_build_batch(**context) -> None:
    # Echantillonne N_SAMPLES lignes du dataset (features sans la cible) et pousse les payloads dans XCom.
    # Si le dataset n'existe pas encore (ex. retrain jamais lance), on le prepare.
    from bank_marketing.config import DATA_PATH, TARGET
    from bank_marketing.data import load_data

    if not DATA_PATH.exists():
        from bank_marketing.prepare_data import prepare

        prepare()

    features = load_data().drop(columns=[TARGET])
    sample = features.sample(n=N_SAMPLES)
    payloads = [json.loads(row.to_json()) for _, row in sample.iterrows()]
    logger.info("Batch construit : %d payloads", len(payloads))
    context["ti"].xcom_push(key="payloads", value=payloads)


def task_predict(**context) -> None:
    # Appelle l'API /predict pour chaque payload (l'API journalise en base) et pousse les resultats dans XCom.
    # API_URL doit pointer vers le service Docker `api` (http://api:8000), defini en variable d'env du
    # service airflow : le defaut config.py (127.0.0.1:8000) viserait le conteneur Airflow lui-meme.
    import requests

    from bank_marketing.config import API_URL

    payloads = context["ti"].xcom_pull(task_ids="build_batch", key="payloads")
    if not payloads:
        raise ValueError("Aucun payload dans XCom : la tache 'build_batch' a-t-elle reussi ?")

    # Verifie que l'API repond avant d'envoyer le lot (message clair si injoignable).
    try:
        health = requests.get(f"{API_URL}/health", timeout=10)
        health.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"API injoignable sur {API_URL} : {exc}") from exc
    logger.info("API /health -> %s", health.json())

    results = []
    for i, payload in enumerate(payloads):
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        body = response.json()
        results.append(body)
        logger.info(
            "POST /predict (#%d) -> prediction=%s proba=%.4f id=%s",
            i, body["prediction"], body["probability"], body["id"],
        )

    context["ti"].xcom_push(key="results", value=results)


def task_summarize(**context) -> None:
    # Resume le lot : nombre de souscripteurs predits et probabilite moyenne.
    results = context["ti"].xcom_pull(task_ids="predict", key="results")
    if not results:
        raise ValueError("Aucun resultat dans XCom : la tache 'predict' a-t-elle reussi ?")

    n = len(results)
    n_pos = sum(1 for r in results if r["prediction"] == 1)
    avg_proba = sum(r["probability"] for r in results) / n
    logger.info(
        "Resume : %d predictions, %d souscripteurs predits (%.0f%%), proba moyenne=%.3f",
        n, n_pos, 100 * n_pos / n, avg_proba,
    )


with DAG(
    dag_id="batch_prediction",
    description="Echantillonne des clients et appelle l'API /predict (serving + journal en base)",
    # Quotidien a 6h du matin (apres un eventuel re-entrainement nocturne).
    schedule="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "prediction"],
) as dag:
    build = PythonOperator(task_id="build_batch", python_callable=task_build_batch)
    predict = PythonOperator(task_id="predict", python_callable=task_predict)
    summarize = PythonOperator(task_id="summarize", python_callable=task_summarize)

    # Ordre lineaire : echantillonner -> predire via l'API -> resumer.
    build >> predict >> summarize
