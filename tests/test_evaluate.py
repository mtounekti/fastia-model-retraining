"""
test_evaluate.py — Tests fonctionnels pour modules/evaluate.py

Fonction testée : evaluate_performance(y_true, y_pred)
"""

import numpy as np
import pytest
from modules.evaluate import evaluate_performance


# ---------------------------------------------------------------------------
# Structure du résultat
# ---------------------------------------------------------------------------

def test_returns_dict():
    """La fonction doit retourner un dictionnaire."""
    y = np.array([1.0, 2.0, 3.0])
    result = evaluate_performance(y, y)
    assert isinstance(result, dict)


def test_returns_correct_keys():
    """Le dictionnaire doit contenir exactement les clés MSE, MAE et R²."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 3.1])
    perf = evaluate_performance(y_true, y_pred)
    assert set(perf.keys()) == {"MSE", "MAE", "R²"}


def test_values_are_floats():
    """Toutes les métriques doivent être des nombres flottants."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 3.1])
    perf = evaluate_performance(y_true, y_pred)
    for key, val in perf.items():
        assert isinstance(val, float), f"{key} n'est pas un float"


# ---------------------------------------------------------------------------
# Cas limites mathématiques
# ---------------------------------------------------------------------------

def test_perfect_predictions_mse_zero():
    """Prédictions parfaites → MSE = 0."""
    y = np.array([100.0, 200.0, 300.0])
    perf = evaluate_performance(y, y)
    assert perf["MSE"] == pytest.approx(0.0, abs=1e-9)


def test_perfect_predictions_mae_zero():
    """Prédictions parfaites → MAE = 0."""
    y = np.array([100.0, 200.0, 300.0])
    perf = evaluate_performance(y, y)
    assert perf["MAE"] == pytest.approx(0.0, abs=1e-9)


def test_perfect_predictions_r2_one():
    """Prédictions parfaites → R² = 1."""
    y = np.array([100.0, 200.0, 300.0])
    perf = evaluate_performance(y, y)
    assert perf["R²"] == pytest.approx(1.0, abs=1e-9)


def test_mean_predictor_r2_zero():
    """Prédire toujours la moyenne → R² = 0 (modèle naïf de référence)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.full(5, np.mean(y_true))
    perf = evaluate_performance(y_true, y_pred)
    assert perf["R²"] == pytest.approx(0.0, abs=1e-9)


def test_negative_r2_when_worse_than_mean():
    """Un modèle inversé (pire que la moyenne) → R² < 0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    perf = evaluate_performance(y_true, y_pred)
    assert perf["R²"] < 0


# ---------------------------------------------------------------------------
# Propriétés des métriques
# ---------------------------------------------------------------------------

def test_mse_always_non_negative():
    """MSE est toujours ≥ 0, quelles que soient les prédictions."""
    np.random.seed(0)
    y_true = np.random.rand(20) * 1000
    y_pred = np.random.rand(20) * 1000
    perf = evaluate_performance(y_true, y_pred)
    assert perf["MSE"] >= 0


def test_mae_always_non_negative():
    """MAE est toujours ≥ 0, quelles que soient les prédictions."""
    np.random.seed(0)
    y_true = np.random.rand(20) * 1000
    y_pred = np.random.rand(20) * 1000
    perf = evaluate_performance(y_true, y_pred)
    assert perf["MAE"] >= 0


def test_mse_greater_than_mae_for_large_errors():
    """MSE > MAE quand les erreurs sont > 1 (MSE pénalise davantage les grandes erreurs)."""
    y_true = np.zeros(5)
    y_pred = np.full(5, 10.0)   # erreur constante de 10
    perf = evaluate_performance(y_true, y_pred)
    assert perf["MSE"] > perf["MAE"]   # 100 > 10


def test_mae_equals_mse_sqrt_for_constant_error():
    """Quand l'erreur est constante, MAE = sqrt(MSE)."""
    y_true = np.zeros(5)
    y_pred = np.full(5, 3.0)   # erreur = 3 partout
    perf = evaluate_performance(y_true, y_pred)
    assert perf["MAE"] == pytest.approx(np.sqrt(perf["MSE"]), abs=1e-6)


def test_symmetry_of_errors():
    """MSE et MAE doivent être identiques si on échange y_true et y_pred."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    perf_1 = evaluate_performance(y_true, y_pred)
    perf_2 = evaluate_performance(y_pred, y_true)
    assert perf_1["MSE"] == pytest.approx(perf_2["MSE"])
    assert perf_1["MAE"] == pytest.approx(perf_2["MAE"])
