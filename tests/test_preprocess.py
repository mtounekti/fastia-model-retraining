"""
test_preprocess.py — Tests fonctionnels pour modules/preprocess.py

Fonctions testées :
    - preprocessing(df)
    - split(X, y)
"""

import numpy as np
import pandas as pd
import pytest
from modules.preprocess import preprocessing, split


# ---------------------------------------------------------------------------
# preprocessing()
# ---------------------------------------------------------------------------

class TestPreprocessing:

    def test_returns_three_values(self, sample_df):
        """preprocessing() doit retourner exactement 3 valeurs (X, y, preprocessor)."""
        result = preprocessing(sample_df)
        assert len(result) == 3

    def test_y_is_montant_pret(self, sample_df):
        """La cible y doit correspondre à la colonne montant_pret."""
        _, y, _ = preprocessing(sample_df)
        pd.testing.assert_series_equal(y, sample_df["montant_pret"], check_names=False)

    def test_nom_prenom_excluded_from_X(self, sample_df):
        """Les colonnes nom et prenom ne doivent pas apparaître dans X."""
        X, _, preprocessor = preprocessing(sample_df)
        # Le preprocessor ne doit pas avoir traité nom/prenom
        # Vérification indirecte : X_shape doit être < nombre de colonnes brutes
        assert X.shape[1] < sample_df.shape[1]

    def test_X_shape_rows_matches_df(self, sample_df):
        """X doit avoir autant de lignes que le DataFrame d'entrée."""
        X, y, _ = preprocessing(sample_df)
        assert X.shape[0] == len(sample_df)

    def test_y_length_matches_df(self, sample_df):
        """y doit avoir autant d'éléments que le DataFrame."""
        _, y, _ = preprocessing(sample_df)
        assert len(y) == len(sample_df)

    def test_no_nan_in_X_after_preprocessing(self, sample_df_with_missing):
        """Après preprocessing, X ne doit contenir aucune valeur manquante."""
        X, _, _ = preprocessing(sample_df_with_missing)
        assert not np.isnan(X).any(), "X contient des NaN après preprocessing"

    def test_no_nan_in_X_nominal_data(self, sample_df):
        """Pas de NaN dans X même sans valeurs manquantes à l'entrée."""
        X, _, _ = preprocessing(sample_df)
        assert not np.isnan(X).any()

    def test_preprocessor_can_transform_new_data(self, sample_df):
        """Le preprocessor retourné doit pouvoir transformer un nouveau DataFrame."""
        _, _, preprocessor = preprocessing(sample_df)
        new_df = sample_df.drop(columns=["nom", "prenom", "montant_pret"]).iloc[:10]
        result = preprocessor.transform(new_df)
        assert result.shape[0] == 10

    def test_X_is_numpy_array(self, sample_df):
        """X doit être un numpy array (sortie du ColumnTransformer)."""
        X, _, _ = preprocessing(sample_df)
        assert isinstance(X, np.ndarray)

    def test_numerical_features_are_scaled(self, sample_df):
        """
        Les features numériques doivent être standardisées :
        la moyenne de chaque colonne numérique doit être proche de 0.
        Les 4 premières colonnes du pipeline sont numériques (age, taille, poids, revenu).
        """
        X, _, _ = preprocessing(sample_df)
        # Colonnes 0-3 = numériques standardisées
        means = np.abs(X[:, :4].mean(axis=0))
        assert np.all(means < 0.2), f"Moyennes numériques trop éloignées de 0 : {means}"


# ---------------------------------------------------------------------------
# split()
# ---------------------------------------------------------------------------

class TestSplit:

    def test_correct_default_split_ratio(self, sample_df):
        """Avec test_size=0.2, le test doit contenir ~20% des données."""
        X, y, _ = preprocessing(sample_df)
        X_train, X_test, y_train, y_test = split(X, y)
        total = len(y_train) + len(y_test)
        assert total == len(y)
        assert abs(len(y_test) / total - 0.2) < 0.05

    def test_no_data_leakage(self, sample_df):
        """Train + test doit égaler le total (pas de perte de données)."""
        X, y, _ = preprocessing(sample_df)
        X_train, X_test, y_train, y_test = split(X, y)
        assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
        assert len(y_train) + len(y_test) == len(y)

    def test_reproducibility_with_same_random_state(self, sample_df):
        """Deux splits avec le même random_state doivent être identiques."""
        X, y, _ = preprocessing(sample_df)
        X_train_1, X_test_1, _, _ = split(X, y, random_state=42)
        X_train_2, X_test_2, _, _ = split(X, y, random_state=42)
        np.testing.assert_array_equal(X_train_1, X_train_2)
        np.testing.assert_array_equal(X_test_1, X_test_2)

    def test_different_random_states_give_different_splits(self, sample_df):
        """Deux random_state différents doivent (très probablement) donner des splits différents."""
        X, y, _ = preprocessing(sample_df)
        X_train_1, _, _, _ = split(X, y, random_state=0)
        X_train_2, _, _, _ = split(X, y, random_state=99)
        assert not np.array_equal(X_train_1, X_train_2)

    def test_custom_test_size(self, sample_df):
        """test_size=0.3 doit donner ~30% de données de test."""
        X, y, _ = preprocessing(sample_df)
        X_train, X_test, _, _ = split(X, y, test_size=0.3)
        ratio = X_test.shape[0] / X.shape[0]
        assert abs(ratio - 0.3) < 0.05

    def test_X_and_y_train_have_same_length(self, sample_df):
        """X_train et y_train doivent avoir la même longueur."""
        X, y, _ = preprocessing(sample_df)
        X_train, _, y_train, _ = split(X, y)
        assert X_train.shape[0] == len(y_train)

    def test_X_and_y_test_have_same_length(self, sample_df):
        """X_test et y_test doivent avoir la même longueur."""
        X, y, _ = preprocessing(sample_df)
        _, X_test, _, y_test = split(X, y)
        assert X_test.shape[0] == len(y_test)
