"""Entrainement du modele de classification (baseline) + suivi MLflow.

Seance 5 - TP MLflow Tracking
    L'entrainement de la baseline (LogisticRegression) est instrumente avec
    MLflow : chaque execution ouvre un run qui logge les parametres, les
    metriques, le modele et la matrice de confusion. Le suivi peut etre
    desactive avec --no-mlflow (utile sans serveur de tracking, ex. tests).

    Pre-requis : un serveur MLflow accessible (voir MLFLOW_TRACKING_URI), par
    exemple `make mlflow` qui demarre le service `mlflow` du docker-compose.
"""
from __future__ import annotations

import argparse
import logging

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from mlproject.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
)
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )


def log_to_mlflow(
    model: Pipeline,
    metrics: dict,
    c: float,
    max_iter: int,
    x_test,
    y_test,
    preds,
) -> None:
    """Logger parametres, metriques, modele et matrice de confusion dans le run courant."""
    # S5-4 : parametres de l'entrainement
    mlflow.log_params({"c": c, "max_iter": max_iter})
    # S5-5 : metriques d'evaluation
    mlflow.log_metrics(metrics)

    # S5-6 : modele (signature + exemple d'entree pour enrichir la doc du modele)
    signature = infer_signature(x_test, model.predict(x_test))
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=x_test.iloc[:5],
    )

    # S5-7 (bonus) : matrice de confusion loggee comme artefact image
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    ax.set_title("Matrice de confusion : baseline LogisticRegression")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


def train(c: float = 1.0, max_iter: int = 1000, use_mlflow: bool = True) -> dict:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    model = build_model(c=c, max_iter=max_iter)
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

    if use_mlflow:
        # S5-2 : URI de tracking + experience cibles
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        logger.info("Suivi MLflow : %s (experience: %s)", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT)
        # S5-3 : un run englobe parametres, metriques, modele et artefacts
        with mlflow.start_run(run_name="baseline-logreg"):
            log_to_mlflow(model, metrics, c, max_iter, x_test, y_test, preds)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Desactive le suivi MLflow (utile sans serveur de tracking)",
    )
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
