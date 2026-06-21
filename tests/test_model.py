"""Tests sur le modele (Seance 11 - tests modele).

Valident le comportement d'un pipeline baseline entraine : probabilites dans
[0, 1], predictions binaires, et porte qualite (ROC AUC >= seuil configure).
"""

from __future__ import annotations

from sklearn.metrics import roc_auc_score

from bank_marketing.config import EVAL_ROC_AUC_MIN


def test_predict_proba_in_range(trained_pipeline):
    pipe, x_test, _ = trained_pipeline
    proba = pipe.predict_proba(x_test)[:, 1]
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_predictions_are_binary(trained_pipeline):
    pipe, x_test, _ = trained_pipeline
    preds = pipe.predict(x_test)
    assert set(preds.tolist()) <= {0, 1}


def test_model_meets_quality_threshold(trained_pipeline):
    # Porte qualite : un modele utile doit depasser le seuil ROC AUC du projet.
    pipe, x_test, y_test = trained_pipeline
    roc = roc_auc_score(y_test, pipe.predict_proba(x_test)[:, 1])
    assert roc >= EVAL_ROC_AUC_MIN, f"roc_auc={roc:.3f} < seuil {EVAL_ROC_AUC_MIN}"
