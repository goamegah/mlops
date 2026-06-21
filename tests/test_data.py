"""Tests sur le jeu de donnees (Seance 11 - tests donnees).

Valident le contrat du dataset prepare : cible binaire presente et propre,
colonnes attendues, absence de la fuite `duration`, types numeriques.
"""

from __future__ import annotations

import pandas as pd

from bank_marketing.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def test_dataset_not_empty(dataset):
    assert len(dataset) > 0


def test_target_present_and_binary(dataset):
    assert TARGET in dataset.columns
    assert set(dataset[TARGET].unique()) <= {0, 1}


def test_target_has_no_missing_value(dataset):
    assert dataset[TARGET].notna().all()


def test_expected_feature_columns_present(dataset):
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        assert col in dataset.columns, f"colonne manquante : {col}"


def test_no_duration_leakage(dataset):
    # `duration` n'est connue qu'apres l'appel : c'est une fuite, doit etre retiree.
    assert "duration" not in dataset.columns


def test_numeric_features_are_numeric(dataset):
    for col in NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(dataset[col]), f"{col} non numerique"


def test_both_classes_present(dataset):
    # Un dataset utilisable pour une classification binaire contient les 2 classes.
    assert dataset[TARGET].nunique() == 2
