"""Entrainement et optimisation de plusieurs modeles de classification (AutoML + SHAP).

Seance 7 - TP AutoML & SHAP
    Ce module compare trois familles de modeles (Random Forest, XGBoost,
    LightGBM), chacune optimisee par recherche d'hyperparametres en grille
    (GridSearchCV), et persiste la meilleure dans `models/model.joblib`.
    Completez les TODO (S7-n).

Chaque modele est suivi dans MLflow (un run par modele, imbrique sous un run
parent ``compare-models``) et le meilleur est enregistre dans le Model
Registry, avec une description et des tags (TODO S7-5, bonus) et un summary
plot SHAP loggue comme artefact (`bank_marketing.evaluation.log_shap_summary`, deja
fourni).

Lancement :
    python -m bank_marketing.train_models
    python -m bank_marketing.train_models --cv 3 --scoring roc_auc
    python -m bank_marketing.train_models --no-mlflow   # desactive le suivi MLflow
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from bank_marketing.config import (
    MODEL_DIR,
    MODEL_NAME,
    RANDOM_STATE,
)
from bank_marketing.data import load_data, split
from bank_marketing.evaluation import log_shap_summary
from bank_marketing.features import build_preprocessor
from bank_marketing.tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Le ColumnTransformer renvoie un tableau numpy sans noms de colonnes lors du
# scoring interne de la validation croisee : on neutralise l'avertissement
# correspondant, sans incidence sur les predictions.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


@dataclass
class ModelSpec:
    """Specification d'un modele a optimiser.

    Parameters
    ----------
    name : str
        Identifiant lisible du modele.
    estimator : ClassifierMixin
        Instance du classifieur scikit-learn compatible.
    param_grid : dict
        Grille d'hyperparametres pour ``GridSearchCV``. Les cles sont
        prefixees par ``clf__`` car le classifieur est la derniere etape
        du pipeline.
    """

    name: str
    estimator: ClassifierMixin
    param_grid: dict


def build_model_specs() -> list[ModelSpec]:
    """Construire la liste des modeles a optimiser.

    Returns
    -------
    list of ModelSpec
        Random Forest, XGBoost et LightGBM avec leurs grilles respectives.
    """
    return [
        ModelSpec(
            name="random_forest",
            estimator=cast(ClassifierMixin, RandomForestClassifier(random_state=RANDOM_STATE)),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        ModelSpec(
            name="xgboost",
            estimator=cast(
                ClassifierMixin,
                XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
        ModelSpec(
            name="lightgbm",
            estimator=cast(ClassifierMixin, LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [31, 63],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Assembler le preprocessing et un classifieur dans un pipeline.

    Parameters
    ----------
    estimator : ClassifierMixin
        Classifieur place en derniere etape (``clf``).

    Returns
    -------
    Pipeline
        Pipeline scikit-learn pret a etre optimise.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", estimator),
        ]
    )


@dataclass
class FitResult:
    """Resultat d'optimisation d'un modele.

    Parameters
    ----------
    name : str
        Identifiant du modele.
    best_estimator : Pipeline
        Pipeline reentraine avec les meilleurs hyperparametres.
    best_params : dict
        Meilleurs hyperparametres trouves par la recherche.
    cv_score : float
        Meilleur score moyen de validation croisee.
    f1 : float
        F1-score sur le jeu de test.
    roc_auc : float
        ROC AUC sur le jeu de test.
    preds : np.ndarray
        Predictions (classes) sur le jeu de test.
    """

    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    f1: float
    roc_auc: float
    preds: np.ndarray


def optimize_model(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> FitResult:
    """Optimiser un modele par GridSearchCV et l'evaluer sur le test.

    Parameters
    ----------
    spec : ModelSpec
        Modele et grille d'hyperparametres.
    x_train, y_train : array-like
        Donnees d'entrainement.
    x_test, y_test : array-like
        Donnees de test pour l'evaluation finale.
    cv : int, optional
        Nombre de plis de validation croisee, par defaut 5.
    scoring : str, optional
        Metrique optimisee par la recherche, par defaut "roc_auc".

    Returns
    -------
    FitResult
        Meilleur estimateur et metriques associees.
    """
    logger.info("Optimisation de %s (cv=%d, scoring=%s)", spec.name, cv, scoring)

    search = GridSearchCV(
        estimator=build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)

    best = search.best_estimator_
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return FitResult(
        name=spec.name,
        best_estimator=best,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
        f1=float(f1_score(y_test, preds)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        preds=preds,
    )


def log_run_to_mlflow(
    result: FitResult,
    x_test,
    y_test,
    cv: int,
    scoring: str,
    register_as: str | None = None,
) -> None:
    """Logger un resultat d'optimisation dans le run MLflow parent (pas imbrique).

    Parameters
    ----------
    result : FitResult
        Resultat a tracer (params, metriques, estimateur).
    x_test : pandas.DataFrame
        Jeu de test, utilise pour inferer la signature et un exemple d'entree.
    y_test : array-like
        Cibles du jeu de test, utilisees pour la matrice de confusion et le
        rapport de classification.
    cv : int
        Nombre de plis de validation croisee (loggue comme parametre).
    scoring : str
        Metrique optimisee (prefixe le nom de la metrique de CV loggee).
    register_as : str, optional
        Si fourni, enregistre le modele dans le Model Registry sous ce nom.
    """
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)
        mlflow.log_param("cv", cv)
        mlflow.log_param("scoring", scoring)

        mlflow.log_params(result.best_params)
        mlflow.log_metrics(
            {
                f"cv_{scoring}": result.cv_score,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
        )

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(str, classification_report(y_test, result.preds))
        mlflow.log_text(report_text, "classification_report.txt")

        log_shap_summary(result.best_estimator, x_test, result.name)

        signature = infer_signature(x_test, result.best_estimator.predict(x_test))
        model_info = mlflow.sklearn.log_model(
            result.best_estimator,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
            skops_trusted_types=[
                "xgboost.sklearn.XGBClassifier",
                "xgboost.core.Booster",
                "lightgbm.sklearn.LGBMClassifier",
                "lightgbm.basic.Booster",
                "collections.OrderedDict",
            ],
        )

        if register_as and model_info.registered_model_version:
            describe_registered_version(
                register_as, model_info.registered_model_version, result, cv, scoring
            )


def describe_registered_version(
    name: str,
    version: int,
    result: FitResult,
    cv: int,
    scoring: str,
) -> None:
    """Documenter une version enregistree dans le Model Registry.

    Ajoute une description (algorithme, hyperparametres, metriques) et des
    tags (famille de modele, methode de recherche, scores) sur la version du
    modele afin de pouvoir comparer les versions sans rouvrir le run MLflow.

    Parameters
    ----------
    name : str
        Nom du modele enregistre dans le registry.
    version : int
        Version enregistree a documenter.
    result : FitResult
        Resultat d'optimisation associe a cette version.
    cv : int
        Nombre de plis de validation croisee utilise pour l'optimisation.
    scoring : str
        Metrique optimisee par GridSearchCV.
    """
    client = mlflow.MlflowClient()
    description = (
        f"Famille : {result.name} | recherche : GridSearchCV | cv : {cv} | scoring : {scoring}\n"
        f"Meilleurs hyperparametres : {result.best_params}\n"
        f"cv_{scoring}={result.cv_score:.4f} | f1={result.f1:.4f} | roc_auc={result.roc_auc:.4f}"
    )
    client.update_model_version(name, str(version), description=description)

    tags = {
        "model_family": result.name,
        "search_method": "grid-search",
        "cv": str(cv),
        "scoring": scoring,
        "f1": f"{result.f1:.4f}",
        "roc_auc": f"{result.roc_auc:.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, str(version), key, value)


def train_all(
    cv: int = 5,
    scoring: str = "roc_auc",
    use_mlflow: bool = True,
) -> list[FitResult]:
    """Entrainer et comparer les trois modeles, sauvegarder le meilleur.

    Le meilleur modele (selon le ROC AUC de test) est persiste dans
    ``models/model.joblib`` afin de rester compatible avec l'API d'inference.
    Lorsque ``use_mlflow`` est actif, chaque modele est suivi dans un run
    MLflow imbrique sous un run parent ``compare-models``, et le meilleur est
    enregistre dans le Model Registry sous ``MODEL_NAME``.

    Parameters
    ----------
    cv : int, optional
        Nombre de plis de validation croisee, par defaut 5.
    scoring : str, optional
        Metrique optimisee par GridSearchCV, par defaut "roc_auc".
    use_mlflow : bool, optional
        Active le suivi MLflow, par defaut True. Necessite un serveur de
        tracking accessible (voir MLFLOW_TRACKING_URI).

    Returns
    -------
    list of FitResult
        Resultats tries du meilleur au moins bon (ROC AUC decroissant).
    """
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    if use_mlflow:
        setup_experiment()

    results = [
        optimize_model(spec, x_train, y_train, x_test, y_test, cv=cv, scoring=scoring)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)

    best = results[0]
    logger.info("Meilleur modele : %s (roc_auc=%.3f)", best.name, best.roc_auc)

    if use_mlflow:
        setup_experiment()
        with mlflow.start_run(run_name="compare-models"):
            log_dataset(df, context="training")
            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)
            mlflow.set_tag("best_model", best.name)
            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_run_to_mlflow(result, x_test, y_test, cv, scoring, register_as=register_as)
        logger.info("Meilleur modele enregistre dans le registry sous '%s'", MODEL_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_estimator, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", MODEL_DIR / "model.joblib")

    return results


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5, help="Nombre de plis de validation croisee")
    parser.add_argument(
        "--scoring",
        type=str,
        default="roc_auc",
        help="Metrique optimisee par GridSearchCV (ex: roc_auc, f1, accuracy)",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Desactive le suivi MLflow (utile sans serveur de tracking)",
    )
    args = parser.parse_args()
    train_all(cv=args.cv, scoring=args.scoring, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
