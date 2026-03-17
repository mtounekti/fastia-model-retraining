"""
test_integration.py — Tests d'intégration du pipeline complet.

Ces tests vérifient que les modules fonctionnent correctement ensemble,
de bout en bout, sans erreur de compatibilité entre les étapes.
"""

import numpy as np
import pytest
from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from models.models import create_nn_model, train_model, model_predict


class TestFullPipeline:

    def test_pipeline_runs_without_error(self, sample_df):
        """Le pipeline complet (preprocess → split → create → train → predict → evaluate)
        doit s'exécuter sans lever d'exception."""
        X, y, _ = preprocessing(sample_df)
        X_train, X_test, y_train, y_test = split(X, y)
        model = create_nn_model(X_train.shape[1])
        model, hist = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=3)
        y_pred = model_predict(model, X_test)
        perf = evaluate_performance(y_test, y_pred)
        assert perf is not None

    def test_preprocessing_output_shape_is_compatible_with_model(self, sample_df):
        """La shape de X après preprocessing doit correspondre à input_dim du modèle."""
        X, y, _ = preprocessing(sample_df)
        X_train, X_test, y_train, y_test = split(X, y)
        model = create_nn_model(X_train.shape[1])
        # Si les shapes sont incompatibles, predict lèvera une exception
        y_pred = model_predict(model, X_test)
        assert len(y_pred) == len(y_test)

    def test_predictions_length_matches_test_set(self, sample_df):
        """y_pred doit avoir exactement autant d'éléments que y_test."""
        X, y, _ = preprocessing(sample_df)
        X_train, X_test, y_train, y_test = split(X, y)
        model = create_nn_model(X_train.shape[1])
        y_pred = model_predict(model, X_test)
        assert len(y_pred) == len(y_test)

    def test_evaluation_metrics_are_finite(self, sample_df):
        """Toutes les métriques doivent être des nombres finis (pas NaN, pas inf)."""
        X, y, _ = preprocessing(sample_df)
        X_train, X_test, y_train, y_test = split(X, y)
        model = create_nn_model(X_train.shape[1])
        train_model(model, X_train, y_train, epochs=3)
        y_pred = model_predict(model, X_test)
        perf = evaluate_performance(y_test, y_pred)
        for key, val in perf.items():
            assert np.isfinite(val), f"Métrique {key} non finie : {val}"

    def test_retrained_model_has_different_predictions(self, sample_df):
        """Un modèle réentraîné doit produire des prédictions différentes du modèle initial."""
        X, y, _ = preprocessing(sample_df)
        X_train, X_test, y_train, y_test = split(X, y)

        model = create_nn_model(X_train.shape[1])
        y_pred_before = model_predict(model, X_test).copy()

        train_model(model, X_train, y_train, epochs=10)
        y_pred_after = model_predict(model, X_test)

        assert not np.array_equal(y_pred_before, y_pred_after), (
            "Les prédictions avant et après entraînement sont identiques — "
            "le modèle ne semble pas avoir appris"
        )

    def test_pipeline_with_missing_values(self, sample_df_with_missing):
        """Le pipeline complet doit fonctionner même si les données d'entrée ont des NaN."""
        X, y, _ = preprocessing(sample_df_with_missing)
        X_train, X_test, y_train, y_test = split(X, y)
        model = create_nn_model(X_train.shape[1])
        train_model(model, X_train, y_train, epochs=2)
        y_pred = model_predict(model, X_test)
        perf = evaluate_performance(y_test, y_pred)
        assert set(perf.keys()) == {"MSE", "MAE", "R²"}

    def test_r2_improves_after_training(self, sample_df):
        """
        Après entraînement, le R² sur le train set doit être meilleur
        qu'avec un modèle non entraîné (poids aléatoires).
        """
        X, y, _ = preprocessing(sample_df)
        X_train, X_test, y_train, y_test = split(X, y)

        model = create_nn_model(X_train.shape[1])

        # R² avant entraînement (poids aléatoires)
        y_pred_before = model_predict(model, X_train)
        r2_before = evaluate_performance(y_train, y_pred_before)["R²"]

        # Entraînement
        train_model(model, X_train, y_train, epochs=30)

        # R² après entraînement
        y_pred_after = model_predict(model, X_train)
        r2_after = evaluate_performance(y_train, y_pred_after)["R²"]

        assert r2_after > r2_before, (
            f"R² après entraînement ({r2_after:.4f}) "
            f"n'est pas meilleur qu'avant ({r2_before:.4f})"
        )
