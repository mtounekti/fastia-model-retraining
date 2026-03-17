"""
conftest.py — Fixtures partagées entre tous les tests.

Les données synthétiques reproduisent la même structure que df_old.csv / df_new.csv
afin de tester les fonctions sans dépendre des fichiers de données réels.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def sample_df():
    """
    DataFrame synthétique (100 lignes) avec la même structure que df_old/df_new.
    Utilisé pour tester preprocessing() et le pipeline complet.
    """
    n = 100
    np.random.seed(42)
    return pd.DataFrame({
        "nom":                    [f"Nom{i}" for i in range(n)],
        "prenom":                 [f"Prenom{i}" for i in range(n)],
        "age":                    np.random.randint(18, 65, n).astype(float),
        "taille":                 np.random.randint(150, 200, n).astype(float),
        "poids":                  np.random.randint(50, 100, n).astype(float),
        "revenu_estime_mois":     np.random.uniform(1000, 5000, n),
        "sexe":                   np.random.choice(["M", "F"], n),
        "sport_licence":          np.random.choice(["oui", "non"], n),
        "niveau_etude":           np.random.choice(["bac", "bac+2", "bac+3", "bac+5"], n),
        "region":                 np.random.choice(["IDF", "PACA", "ARA", "OCC"], n),
        "smoker":                 np.random.choice(["yes", "no"], n),
        "nationalité_francaise":  np.random.choice(["oui", "non"], n),
        "montant_pret":           np.random.uniform(500, 50000, n),
    })


@pytest.fixture
def sample_df_with_missing(sample_df):
    """
    DataFrame avec des valeurs manquantes intentionnelles sur colonnes
    numériques et catégorielles, pour tester l'imputation.
    """
    df = sample_df.copy()
    df.loc[0:4, "age"] = np.nan
    df.loc[5:9, "revenu_estime_mois"] = np.nan
    df.loc[10:14, "sexe"] = np.nan
    df.loc[15:19, "region"] = np.nan
    return df


@pytest.fixture
def small_arrays():
    """
    Petits arrays numpy (50 samples, 10 features) pour tester les fonctions
    du modèle rapidement sans overhead de preprocessing.
    """
    np.random.seed(42)
    X = np.random.rand(50, 10).astype(np.float32)
    y = (np.random.rand(50) * 49500 + 500).astype(np.float32)
    return X, y


@pytest.fixture
def mock_history():
    """
    Historique Keras simulé (MagicMock) pour tester les fonctions
    de visualisation sans avoir à entraîner un vrai modèle.
    """
    history = MagicMock()
    history.history = {
        "loss":     [1000.0, 800.0, 600.0, 400.0, 200.0],
        "val_loss": [1100.0, 900.0, 700.0, 500.0, 300.0],
    }
    return history


@pytest.fixture
def mock_history_no_val():
    """Historique sans val_loss, pour tester save_loss_plot sans validation."""
    history = MagicMock()
    history.history = {
        "loss": [1000.0, 800.0, 600.0, 400.0, 200.0],
    }
    return history
