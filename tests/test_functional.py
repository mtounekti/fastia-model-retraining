"""
test_functional.py — Tests fonctionnels du pipeline de réentraînement.

Ces tests vérifient des comportements métier complets du point de vue utilisateur :
- Le réentraînement améliore-t-il les performances ?
- La dérive de données est-elle détectable ?
- Les modèles sont-ils bien sauvegardés après entraînement ?
- Le pipeline produit-il des performances acceptables sur les vraies données ?

Contrairement aux tests unitaires, ces tests utilisent les vraies données
(df_old.csv, df_new.csv) et les vrais fichiers modèles du projet.
"""

import os
import joblib
import numpy as np
import pytest
import pandas as pd
from os.path import join

from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from models.models import create_nn_model, train_model, model_predict


DATA_DIR = "data"
MODELS_DIR = "models"

# ---------------------------------------------------------------------------
# Fixtures sur les vraies données
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_old_data():
    """Charge et prétraite df_old.csv (données d'entraînement original)."""
    df = pd.read_csv(join(DATA_DIR, "df_old.csv"))
    X, y, preprocessor = preprocessing(df)
    X_train, X_test, y_train, y_test = split(X, y)
    return X_train, X_test, y_train, y_test, preprocessor


@pytest.fixture(scope="module")
def real_new_data():
    """Charge et prétraite df_new.csv (nouvelles données)."""
    df = pd.read_csv(join(DATA_DIR, "df_new.csv"))
    X, y, preprocessor = preprocessing(df)
    X_train, X_test, y_train, y_test = split(X, y)
    return X_train, X_test, y_train, y_test, preprocessor


@pytest.fixture(scope="module")
def original_model():
    """Charge le modèle original model_2024_08.pkl."""
    return joblib.load(join(MODELS_DIR, "model_2024_08.pkl"))


# ---------------------------------------------------------------------------
# Scénario 1 : Dérive de données détectable
# ---------------------------------------------------------------------------

class TestDataDrift:

    def test_model_performs_worse_on_new_data(self, real_old_data, real_new_data, original_model):
        """
        Le modèle original doit avoir un R² plus faible sur les nouvelles données
        que sur les anciennes — confirme la dérive de données.
        """
        _, X_old_test, _, y_old_test, _ = real_old_data
        _, X_new_test, _, y_new_test, _ = real_new_data

        y_pred_old = model_predict(original_model, X_old_test)
        y_pred_new = model_predict(original_model, X_new_test)

        r2_old = evaluate_performance(y_old_test, y_pred_old)["R²"]
        r2_new = evaluate_performance(y_new_test, y_pred_new)["R²"]

        assert r2_old > r2_new, (
            f"Aucune dérive détectée : R² old={r2_old:.4f}, R² new={r2_new:.4f}. "
            f"Le modèle devrait être moins performant sur les nouvelles données."
        )

    def test_drift_is_significant(self, real_old_data, real_new_data, original_model):
        """
        La dégradation entre df_old et df_new doit être d'au moins 0.05 de R²
        pour être considérée comme significative.
        """
        _, X_old_test, _, y_old_test, _ = real_old_data
        _, X_new_test, _, y_new_test, _ = real_new_data

        r2_old = evaluate_performance(y_old_test, model_predict(original_model, X_old_test))["R²"]
        r2_new = evaluate_performance(y_new_test, model_predict(original_model, X_new_test))["R²"]

        assert (r2_old - r2_new) >= 0.05, (
            f"Dérive trop faible : {r2_old - r2_new:.4f} < 0.05"
        )


# ---------------------------------------------------------------------------
# Scénario 2 : Le réentraînement améliore les performances
# ---------------------------------------------------------------------------

class TestRetrainingImproves:

    def test_retrain_on_old_data_improves_r2(self, real_old_data):
        """
        Réentraîner le modèle original sur df_old doit produire un R² meilleur
        qu'avant réentraînement — valide que le fine-tuning fonctionne.
        """
        X_train, X_test, y_train, y_test, _ = real_old_data
        model = joblib.load(join(MODELS_DIR, "model_2024_08.pkl"))

        r2_before = evaluate_performance(y_test, model_predict(model, X_test))["R²"]
        train_model(model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=50)
        r2_after = evaluate_performance(y_test, model_predict(model, X_test))["R²"]

        assert r2_after > r2_before, (
            f"Le réentraînement n'a pas amélioré le R² : avant={r2_before:.4f}, après={r2_after:.4f}"
        )

    def test_fresh_model_learns_on_new_data(self, real_new_data):
        """
        Un modèle vierge entraîné sur df_new doit atteindre un R² > 0.50,
        prouvant qu'il apprend un signal utile sur les nouvelles données.
        """
        X_train, X_test, y_train, y_test, _ = real_new_data
        model = create_nn_model(X_train.shape[1])

        train_model(model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=100)
        r2 = evaluate_performance(y_test, model_predict(model, X_test))["R²"]

        assert r2 > 0.50, (
            f"Le nouveau modèle n'apprend pas correctement : R²={r2:.4f} < 0.50"
        )


# ---------------------------------------------------------------------------
# Scénario 3 : Performances acceptables sur les vraies données
# ---------------------------------------------------------------------------

class TestAcceptablePerformance:

    def test_original_model_r2_above_threshold_on_old_data(self, real_old_data, original_model):
        """
        Le modèle original doit atteindre R² > 0.70 sur df_old.
        En dessous, il serait considéré comme non fiable pour la production.
        """
        _, X_test, _, y_test, _ = real_old_data
        r2 = evaluate_performance(y_test, model_predict(original_model, X_test))["R²"]
        assert r2 > 0.70, f"Performance insuffisante sur df_old : R²={r2:.4f}"

    def test_original_model_r2_above_threshold_on_new_data(self, real_new_data, original_model):
        """
        Le modèle original doit atteindre R² > 0.55 sur df_new.
        Un score plus bas indiquerait une dérive trop importante pour la production.
        """
        _, X_test, _, y_test, _ = real_new_data
        r2 = evaluate_performance(y_test, model_predict(original_model, X_test))["R²"]
        assert r2 > 0.55, f"Dérive trop importante sur df_new : R²={r2:.4f}"


# ---------------------------------------------------------------------------
# Scénario 4 : Les fichiers modèles sont sauvegardés après train.py
# ---------------------------------------------------------------------------

class TestModelFilesExist:

    def test_original_model_file_exists(self):
        """Le modèle original doit exister avant tout réentraînement."""
        assert os.path.isfile(join(MODELS_DIR, "model_2024_08.pkl")), \
            "model_2024_08.pkl introuvable"

    def test_preprocessor_file_exists(self):
        """Le préprocesseur original doit exister."""
        assert os.path.isfile(join(MODELS_DIR, "preprocessor.pkl")), \
            "preprocessor.pkl introuvable"

    def test_retrained_model_file_exists(self):
        """model_retrained_new_data.pkl doit exister après exécution de train.py (Exp4)."""
        assert os.path.isfile(join(MODELS_DIR, "model_retrained_new_data.pkl")), \
            "model_retrained_new_data.pkl introuvable — avez-vous lancé train.py ?"

    def test_fresh_model_file_exists(self):
        """model_fresh_new_data.pkl doit exister après exécution de train.py (Exp5)."""
        assert os.path.isfile(join(MODELS_DIR, "model_fresh_new_data.pkl")), \
            "model_fresh_new_data.pkl introuvable — avez-vous lancé train.py ?"

    def test_retrained_model_is_loadable(self):
        """Le modèle réentraîné doit pouvoir être rechargé sans erreur."""
        model = joblib.load(join(MODELS_DIR, "model_retrained_new_data.pkl"))
        assert model is not None

    def test_retrained_model_can_predict(self, real_new_data):
        """Le modèle réentraîné rechargé doit pouvoir faire des prédictions."""
        _, X_test, _, y_test, _ = real_new_data
        model = joblib.load(join(MODELS_DIR, "model_retrained_new_data.pkl"))
        y_pred = model_predict(model, X_test)
        assert len(y_pred) == len(y_test)
        assert not np.isnan(y_pred).any()


# ---------------------------------------------------------------------------
# Scénario 5 : Reproductibilité du pipeline
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_data_same_split(self):
        """
        Le même DataFrame prétraité avec le même random_state doit toujours
        produire le même split — garantit la reproductibilité des expériences.
        """
        df = pd.read_csv(join(DATA_DIR, "df_old.csv"))
        X, y, _ = preprocessing(df)

        X_train_1, X_test_1, _, _ = split(X, y, random_state=42)
        X_train_2, X_test_2, _, _ = split(X, y, random_state=42)

        np.testing.assert_array_equal(X_train_1, X_train_2)
        np.testing.assert_array_equal(X_test_1, X_test_2)


# ---------------------------------------------------------------------------
# Scénario 6 : Early stopping évite le surapprentissage
# ---------------------------------------------------------------------------

class TestEarlyStopping:

    def test_early_stopping_stops_before_max_epochs_on_real_data(self, real_new_data):
        """
        Sur les vraies données (df_new), l'early stopping doit arrêter
        l'entraînement bien avant 300 epochs.
        """
        X_train, X_test, y_train, y_test, _ = real_new_data
        model = create_nn_model(X_train.shape[1])
        _, hist = train_model(
            model, X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=300,
            early_stopping=True,
            patience=10
        )
        assert len(hist.history["loss"]) < 300, (
            "L'early stopping n'a pas arrêté l'entraînement sur df_new"
        )

    def test_early_stopping_model_file_exists(self):
        """model_early_stopping.pkl doit exister après exécution de train.py (Exp6)."""
        assert os.path.isfile(join(MODELS_DIR, "model_early_stopping.pkl")), \
            "model_early_stopping.pkl introuvable — avez-vous lancé train.py ?"

    def test_early_stopping_model_achieves_acceptable_r2(self, real_new_data):
        """
        Le modèle sauvegardé par Exp6 (early stopping) doit atteindre R² > 0.50
        sur les nouvelles données.
        """
        _, X_test, _, y_test, _ = real_new_data
        model = joblib.load(join(MODELS_DIR, "model_early_stopping.pkl"))
        r2 = evaluate_performance(y_test, model_predict(model, X_test))["R²"]
        assert r2 > 0.50, f"Modèle early stopping insuffisant : R²={r2:.4f}"

    def test_preprocessing_is_deterministic(self):
        """
        Appliquer preprocessing() deux fois sur le même DataFrame doit
        produire exactement les mêmes features X.
        """
        df = pd.read_csv(join(DATA_DIR, "df_old.csv"))
        X1, _, _ = preprocessing(df)
        X2, _, _ = preprocessing(df)
        np.testing.assert_array_almost_equal(X1, X2)
