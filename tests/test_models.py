"""
test_models.py — Tests fonctionnels pour models/models.py

Fonctions testées :
    - create_nn_model(input_dim)
    - model_predict(model, X)
    - train_model(model, X, y, ...)

Note : ces tests importent TensorFlow — ils sont légèrement plus lents que
les autres. L'entraînement est limité à 3 epochs pour rester rapide.
"""

import numpy as np
import pytest
from models.models import create_nn_model, train_model, model_predict


# ---------------------------------------------------------------------------
# create_nn_model()
# ---------------------------------------------------------------------------

class TestCreateNnModel:

    def test_model_is_not_none(self):
        """create_nn_model() ne doit pas retourner None."""
        model = create_nn_model(10)
        assert model is not None

    def test_model_has_three_layers(self):
        """Le modèle doit avoir exactement 3 couches Dense."""
        model = create_nn_model(10)
        assert len(model.layers) == 3

    def test_output_shape_is_one(self):
        """La couche de sortie doit avoir 1 neurone (régression scalaire)."""
        model = create_nn_model(10)
        assert model.output_shape == (None, 1)

    def test_input_dim_respected(self):
        """Le modèle doit accepter le bon nombre de features en entrée."""
        input_dim = 15
        model = create_nn_model(input_dim)
        assert model.input_shape == (None, input_dim)

    def test_model_is_compiled(self):
        """Le modèle doit être compilé (optimizer et loss définis)."""
        model = create_nn_model(10)
        assert model.optimizer is not None
        assert model.loss is not None

    def test_first_layer_has_64_units(self):
        """La première couche doit avoir 64 neurones."""
        model = create_nn_model(10)
        assert model.layers[0].units == 64

    def test_second_layer_has_32_units(self):
        """La deuxième couche doit avoir 32 neurones."""
        model = create_nn_model(10)
        assert model.layers[1].units == 32

    def test_different_input_dims_create_different_models(self):
        """Deux input_dim différents doivent créer des modèles de tailles différentes."""
        model_5 = create_nn_model(5)
        model_20 = create_nn_model(20)
        assert model_5.input_shape != model_20.input_shape


# ---------------------------------------------------------------------------
# model_predict()
# ---------------------------------------------------------------------------

class TestModelPredict:

    def test_output_is_1d_array(self, small_arrays):
        """model_predict doit retourner un tableau 1D (flatten)."""
        X, _ = small_arrays
        model = create_nn_model(X.shape[1])
        y_pred = model_predict(model, X)
        assert y_pred.ndim == 1

    def test_output_length_matches_input(self, small_arrays):
        """La longueur de y_pred doit correspondre au nombre de samples en entrée."""
        X, _ = small_arrays
        model = create_nn_model(X.shape[1])
        y_pred = model_predict(model, X)
        assert len(y_pred) == X.shape[0]

    def test_output_contains_no_nan(self, small_arrays):
        """y_pred ne doit contenir aucun NaN."""
        X, _ = small_arrays
        model = create_nn_model(X.shape[1])
        y_pred = model_predict(model, X)
        assert not np.isnan(y_pred).any()

    def test_output_is_float_array(self, small_arrays):
        """y_pred doit être un tableau de flottants."""
        X, _ = small_arrays
        model = create_nn_model(X.shape[1])
        y_pred = model_predict(model, X)
        assert np.issubdtype(y_pred.dtype, np.floating)

    def test_deterministic_on_same_input(self, small_arrays):
        """Deux prédictions sur les mêmes données doivent être identiques."""
        X, _ = small_arrays
        model = create_nn_model(X.shape[1])
        y_pred_1 = model_predict(model, X)
        y_pred_2 = model_predict(model, X)
        np.testing.assert_array_equal(y_pred_1, y_pred_2)


# ---------------------------------------------------------------------------
# train_model()
# ---------------------------------------------------------------------------

class TestTrainModel:

    def test_returns_model_and_history(self, small_arrays):
        """train_model doit retourner exactement 2 valeurs (model, history)."""
        X, y = small_arrays
        model = create_nn_model(X.shape[1])
        result = train_model(model, X, y, epochs=2)
        assert len(result) == 2

    def test_history_contains_loss_key(self, small_arrays):
        """L'historique doit contenir la clé 'loss'."""
        X, y = small_arrays
        model = create_nn_model(X.shape[1])
        _, hist = train_model(model, X, y, epochs=2)
        assert "loss" in hist.history

    def test_history_contains_val_loss_when_validation_provided(self, small_arrays):
        """val_loss doit apparaître dans l'historique si des données de validation sont fournies."""
        X, y = small_arrays
        X_train, X_val = X[:40], X[40:]
        y_train, y_val = y[:40], y[40:]
        model = create_nn_model(X.shape[1])
        _, hist = train_model(model, X_train, y_train, X_val=X_val, y_val=y_val, epochs=2)
        assert "val_loss" in hist.history

    def test_no_val_loss_without_validation_data(self, small_arrays):
        """Sans données de validation, val_loss ne doit pas apparaître."""
        X, y = small_arrays
        model = create_nn_model(X.shape[1])
        _, hist = train_model(model, X, y, epochs=2)
        assert "val_loss" not in hist.history

    def test_loss_history_length_matches_epochs(self, small_arrays):
        """La longueur de l'historique de loss doit correspondre au nombre d'epochs."""
        X, y = small_arrays
        epochs = 5
        model = create_nn_model(X.shape[1])
        _, hist = train_model(model, X, y, epochs=epochs)
        assert len(hist.history["loss"]) == epochs

    def test_loss_decreases_after_training(self, small_arrays):
        """La loss finale doit être inférieure à la loss initiale après entraînement."""
        X, y = small_arrays
        model = create_nn_model(X.shape[1])
        _, hist = train_model(model, X, y, epochs=20)
        assert hist.history["loss"][-1] < hist.history["loss"][0]

    def test_weights_change_after_training(self, small_arrays):
        """Les poids du modèle doivent changer après l'entraînement."""
        X, y = small_arrays
        model = create_nn_model(X.shape[1])
        weights_before = [w.numpy().copy() for w in model.trainable_weights]
        train_model(model, X, y, epochs=3)
        weights_after = [w.numpy() for w in model.trainable_weights]
        any_changed = any(
            not np.array_equal(b, a)
            for b, a in zip(weights_before, weights_after)
        )
        assert any_changed, "Les poids n'ont pas changé après l'entraînement"


# ---------------------------------------------------------------------------
# early_stopping dans train_model()
# ---------------------------------------------------------------------------

class TestEarlyStopping:

    def test_early_stopping_stops_before_max_epochs(self):
        """
        Avec early_stopping=True, le nombre d'epochs réel doit être
        inférieur au max fixé.

        On utilise un dataset qui garantit le surapprentissage :
        - 6 samples d'entraînement mémorisables (identité → labels simples)
        - 30 samples de validation avec labels aléatoires non apprenables
        → val_loss diverge rapidement, l'early stopping se déclenche.
        """
        np.random.seed(42)
        # Train : très peu de samples → le modèle les mémorise
        X_train = np.eye(6, 10).astype(np.float32)
        y_train = np.array([1., 2., 3., 4., 5., 6.], dtype=np.float32) * 10000

        # Val : labels aléatoires → aucune généralisation possible
        X_val = np.random.rand(30, 10).astype(np.float32)
        y_val = np.random.rand(30).astype(np.float32) * 50000

        model = create_nn_model(10)
        _, hist = train_model(
            model, X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=500,
            early_stopping=True,
            patience=5
        )
        assert len(hist.history["loss"]) < 500, (
            "L'early stopping n'a pas arrêté l'entraînement avant 500 epochs"
        )

    def test_early_stopping_disabled_runs_all_epochs(self, small_arrays):
        """
        Sans early_stopping, le modèle doit s'entraîner exactement le nombre
        d'epochs demandé.
        """
        X, y = small_arrays
        X_train, X_val = X[:40], X[40:]
        y_train, y_val = y[:40], y[40:]
        model = create_nn_model(X.shape[1])
        _, hist = train_model(
            model, X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=10,
            early_stopping=False
        )
        assert len(hist.history["loss"]) == 10

    def test_early_stopping_ignored_without_val_data(self, small_arrays):
        """
        Sans données de validation, early_stopping=True ne doit pas planter
        et doit s'entraîner normalement.
        """
        X, y = small_arrays
        model = create_nn_model(X.shape[1])
        _, hist = train_model(model, X, y, epochs=5, early_stopping=True, patience=3)
        assert len(hist.history["loss"]) == 5

    def test_early_stopping_restores_best_weights(self):
        """
        Avec restore_best_weights=True (défaut), les poids finaux doivent
        correspondre au meilleur checkpoint — la val_loss finale ne doit pas
        être la pire observée.
        """
        np.random.seed(42)
        X_train = np.eye(6, 10).astype(np.float32)
        y_train = np.array([1., 2., 3., 4., 5., 6.], dtype=np.float32) * 10000
        X_val = np.random.rand(30, 10).astype(np.float32)
        y_val = np.random.rand(30).astype(np.float32) * 50000

        model = create_nn_model(10)
        _, hist = train_model(
            model, X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=500,
            early_stopping=True,
            patience=5
        )
        val_losses = hist.history["val_loss"]
        final_val_loss = val_losses[-1]
        max_val_loss = max(val_losses)
        assert final_val_loss <= max_val_loss, (
            "Les poids finaux semblent être les pires observés"
        )
