"""
train.py — Script de réentraînement et de suivi des performances avec MLflow.

Ce script orchestre plusieurs expériences de réentraînement afin de comparer
les performances du modèle original avec différentes combinaisons de données
et de stratégies d'entraînement.

Expériences menées :
    1. Baseline       : modèle original évalué sur les anciennes données (df_old)
    2. Dérive         : modèle original évalué sur les nouvelles données (df_new)
    3. Surentraînement: modèle original réentraîné sur les anciennes données
    4. Réentraînement : modèle original réentraîné sur les nouvelles données  ← objectif principal
    5. Nouveau modèle : modèle vierge entraîné depuis zéro sur les nouvelles données

Lancer le script :
    python train.py

Lancer l'interface MLflow :
    mlflow ui
    → http://localhost:5000
"""

import os
import mlflow
import mlflow.keras
import pandas as pd
import joblib
from os.path import join
from loguru import logger

from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from modules.print_draw import print_data, save_loss_plot
from models.models import create_nn_model, train_model, model_predict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "data"
MODELS_DIR = "models"
MLFLOW_EXPERIMENT = "OPCO-ATLAS-Loan-Prediction"

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# ---------------------------------------------------------------------------
# Chargement et prétraitement des données
# ---------------------------------------------------------------------------
logger.info("Chargement des données...")
df_old = pd.read_csv(join(DATA_DIR, "df_old.csv"))
df_new = pd.read_csv(join(DATA_DIR, "df_new.csv"))

logger.info(f"df_old : {df_old.shape[0]} lignes | df_new : {df_new.shape[0]} lignes")

# Prétraitement indépendant de chaque dataset
logger.info("Prétraitement des données...")
X_old, y_old, preprocessor_old = preprocessing(df_old)
X_new, y_new, preprocessor_new = preprocessing(df_new)

# Chargement du préprocesseur sauvegardé (fitté lors de l'entraînement initial)
preprocessor_saved = joblib.load(join(MODELS_DIR, "preprocessor.pkl"))

# Split train/test (80/20, random_state fixe pour reproductibilité)
X_old_train, X_old_test, y_old_train, y_old_test = split(X_old, y_old)
X_new_train, X_new_test, y_new_train, y_new_test = split(X_new, y_new)

# Transformer df_new avec le preprocesseur original (pour Exp 2)
X_new_with_old_prep = preprocessor_saved.transform(
    df_new.drop(columns=["nom", "prenom", "montant_pret"])
)
X_new_old_prep_train, X_new_old_prep_test, y_new_old_train, y_new_old_test = split(
    X_new_with_old_prep, y_new
)


# ---------------------------------------------------------------------------
# Fonction utilitaire d'expérience
# ---------------------------------------------------------------------------
def run_experiment(exp_name, model, X_train, y_train, X_test, y_test,
                   retrain=False, epochs=50, data_tag="old", save_model_path=None,
                   early_stopping=False, patience=10):
    """
    Exécute une expérience MLflow : entraînement optionnel puis évaluation.

    Inputs:
        exp_name (str): Nom affiché dans MLflow et dans la console.
        model: Modèle Keras à évaluer ou réentraîner.
        X_train, y_train: Données d'entraînement.
        X_test, y_test: Données de test.
        retrain (bool): Si True, réentraîne le modèle avant l'évaluation.
        epochs (int): Nombre maximum d'époques si retrain=True.
        data_tag (str): Tag MLflow indiquant le dataset utilisé ("old" ou "new").
        save_model_path (str, optional): Chemin pour sauvegarder le modèle réentraîné.
        early_stopping (bool): Si True, arrête l'entraînement quand val_loss stagne.
        patience (int): Nombre d'epochs sans amélioration avant arrêt (défaut: 10).

    Outputs:
        model: Modèle après entraînement éventuel.
        perf_test (dict): Métriques sur le jeu de test {'MSE', 'MAE', 'R²'}.
    """
    logger.info(f"--- Démarrage : {exp_name} ---")

    with mlflow.start_run(run_name=exp_name):
        mlflow.set_tag("dataset", data_tag)
        mlflow.set_tag("retrain", str(retrain))
        mlflow.set_tag("early_stopping", str(early_stopping))
        mlflow.log_param("epochs_max", epochs if retrain else 0)
        mlflow.log_param("retrain", retrain)
        mlflow.log_param("early_stopping", early_stopping)
        mlflow.log_param("patience", patience if early_stopping else "N/A")
        mlflow.log_param("train_samples", len(y_train))
        mlflow.log_param("test_samples", len(y_test))

        if retrain:
            logger.info(f"Réentraînement du modèle ({epochs} epochs max, early_stopping={early_stopping})...")
            model, hist = train_model(
                model, X_train, y_train,
                X_val=X_test, y_val=y_test,
                epochs=epochs,
                early_stopping=early_stopping,
                patience=patience
            )
            # Log du nombre réel d'epochs effectuées
            actual_epochs = len(hist.history["loss"])
            mlflow.log_param("epochs_actual", actual_epochs)
            logger.info(f"Arrêt après {actual_epochs} epochs.")
            fig_path = save_loss_plot(hist, exp_name)
            mlflow.log_artifact(fig_path)
            logger.success(f"Courbe de loss sauvegardée : {fig_path}")

        # Évaluation
        y_pred_train = model_predict(model, X_train)
        y_pred_test = model_predict(model, X_test)

        perf_train = evaluate_performance(y_train, y_pred_train)
        perf_test = evaluate_performance(y_test, y_pred_test)

        # Log des métriques dans MLflow
        mlflow.log_metrics({
            "train_mse": perf_train["MSE"],
            "train_mae": perf_train["MAE"],
            "train_r2":  perf_train["R²"],
            "test_mse":  perf_test["MSE"],
            "test_mae":  perf_test["MAE"],
            "test_r2":   perf_test["R²"],
        })

        # Log du modèle comme artefact MLflow
        mlflow.keras.log_model(model, "model")

        # Affichage console
        print_data(perf_train, exp_name=f"{exp_name} [TRAIN]")
        print_data(perf_test,  exp_name=f"{exp_name} [TEST]")

        if save_model_path:
            joblib.dump(model, save_model_path)
            mlflow.log_artifact(save_model_path)
            logger.success(f"Modèle sauvegardé : {save_model_path}")

    return model, perf_test


# ---------------------------------------------------------------------------
# EXP 1 — Baseline : modèle original sur les anciennes données
# Objectif : référence de performance initiale du modèle en production
# ---------------------------------------------------------------------------
logger.info("=== EXP 1 : Baseline — Modèle original / Données anciennes ===")
model_original = joblib.load(join(MODELS_DIR, "model_2024_08.pkl"))
run_experiment(
    "Exp1_Baseline_OldModel_OldData",
    model_original,
    X_old_train, y_old_train, X_old_test, y_old_test,
    retrain=False,
    data_tag="old"
)

# ---------------------------------------------------------------------------
# EXP 2 — Dérive de données : modèle original sur les nouvelles données
# Objectif : mesurer la dégradation causée par le décalage temporel des données
# ---------------------------------------------------------------------------
logger.info("=== EXP 2 : Dérive — Modèle original / Données nouvelles ===")
model_original = joblib.load(join(MODELS_DIR, "model_2024_08.pkl"))
run_experiment(
    "Exp2_DataDrift_OldModel_NewData",
    model_original,
    X_new_old_prep_train, y_new_old_train, X_new_old_prep_test, y_new_old_test,
    retrain=False,
    data_tag="new"
)

# ---------------------------------------------------------------------------
# EXP 3 — Surentraînement : modèle original réentraîné sur les mêmes données
# Objectif : illustrer ce qui se passe quand on réentraîne sur les mêmes données
#            (réponse à la question du brief : on fait du fine-tuning / overfitting)
# ---------------------------------------------------------------------------
logger.info("=== EXP 3 : Réentraînement sur mêmes données (overfitting test) ===")
model_for_exp3 = joblib.load(join(MODELS_DIR, "model_2024_08.pkl"))
run_experiment(
    "Exp3_Retrain_OldModel_OldData",
    model_for_exp3,
    X_old_train, y_old_train, X_old_test, y_old_test,
    retrain=True,
    epochs=50,
    data_tag="old"
)

# ---------------------------------------------------------------------------
# EXP 4 — Réentraînement principal : modèle original sur les nouvelles données
# Objectif : re-calibrer le modèle existant avec les données récentes
# ---------------------------------------------------------------------------
logger.info("=== EXP 4 : Réentraînement — Modèle original / Nouvelles données ===")
model_for_exp4 = joblib.load(join(MODELS_DIR, "model_2024_08.pkl"))
model_retrained, _ = run_experiment(
    "Exp4_Retrain_OldModel_NewData",
    model_for_exp4,
    X_new_train, y_new_train, X_new_test, y_new_test,
    retrain=True,
    epochs=50,
    data_tag="new",
    save_model_path=join(MODELS_DIR, "model_retrained_new_data.pkl")
)

# ---------------------------------------------------------------------------
# EXP 5 — Nouveau modèle vierge entraîné sur les nouvelles données
# Objectif : comparer avec l'approche de transfert learning (Exp 4)
# ---------------------------------------------------------------------------
logger.info("=== EXP 5 : Nouveau modèle vierge / Nouvelles données ===")
new_model = create_nn_model(X_new_train.shape[1])
run_experiment(
    "Exp5_FreshModel_NewData",
    new_model,
    X_new_train, y_new_train, X_new_test, y_new_test,
    retrain=True,
    epochs=100,
    data_tag="new",
    save_model_path=join(MODELS_DIR, "model_fresh_new_data.pkl")
)

# ---------------------------------------------------------------------------
# EXP 6 — Early Stopping : nouveau modèle sur nouvelles données
# Objectif : montrer que l'early stopping évite le surapprentissage en
#            arrêtant automatiquement quand val_loss ne s'améliore plus.
#            On donne 300 epochs max mais le modèle s'arrêtera bien avant.
# ---------------------------------------------------------------------------
logger.info("=== EXP 6 : Early Stopping — Nouveau modèle / Nouvelles données ===")
model_es = create_nn_model(X_new_train.shape[1])
run_experiment(
    "Exp6_EarlyStopping_NewData",
    model_es,
    X_new_train, y_new_train, X_new_test, y_new_test,
    retrain=True,
    epochs=300,
    data_tag="new",
    early_stopping=True,
    patience=10,
    save_model_path=join(MODELS_DIR, "model_early_stopping.pkl")
)

logger.success("Toutes les expériences sont terminées. Lancez 'mlflow ui' pour visualiser les résultats.")
