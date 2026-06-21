"""Entrainement du modele de classification (baseline LogisticRegression).

Seance 5 - TP MLflow Tracking
    Entraine une baseline (regression logistique) et l'instrumente avec MLflow :
    parametres, metriques (f1, roc_auc), matrice de confusion en artefact et
    modele loggue. Le suivi MLflow peut etre desactive (--no-mlflow) pour
    s'executer sans serveur de tracking (ex. integration continue).

Lancement :
    python -m bank_marketing.train
    python -m bank_marketing.train --c 0.5 --max-iter 2000
    python -m bank_marketing.train --no-mlflow   # sans serveur MLflow
"""

from __future__ import annotations

import argparse

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

from bank_marketing.config import MODEL_DIR
from bank_marketing.data import load_data, split
from bank_marketing.features import build_preprocessor
from bank_marketing.tracking import setup_experiment


def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )


def train(c: float = 1.0, max_iter: int = 1000, use_mlflow: bool = True) -> dict:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    # S5-2 : configuration centralisee du tracking (URI + experience), comme
    # train_optuna / train_models. Voir bank_marketing.tracking.setup_experiment.
    if use_mlflow:
        setup_experiment()

    model = build_model(c=c, max_iter=max_iter)
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

    # S5-3 : un run englobe le logging des parametres, metriques, artefacts et modele.
    if use_mlflow:
        with mlflow.start_run(run_name="baseline"):
            mlflow.set_tag("model_family", "logistic_regression")
            # S5-4 : parametres.
            mlflow.log_params({"c": c, "max_iter": max_iter})
            # S5-5 : metriques.
            mlflow.log_metrics(metrics)

            # S5-7 (bonus) : matrice de confusion sauvegardee comme artefact image.
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots(figsize=(5, 5))
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            ax.set_title("Matrice de confusion : baseline")
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            # S5-6 : modele loggue avec signature et exemple d'entree (rechargeable).
            signature = infer_signature(x_test, model.predict(x_test))
            mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=x_test.iloc[:5],
            )

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
        help="Desactive le suivi MLflow (utile sans serveur de tracking, ex. CI)",
    )
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
