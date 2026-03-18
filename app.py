import sys
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from loguru import logger

# Loguru config
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
)
logger.add(
    LOG_DIR / "api.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
)

# artifact paths
MODEL_PATH = Path("models/model_early_stopping.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

# Load model & preprocessor at startup
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
    logger.info(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
except Exception as exc:
    logger.error(f"Failed to load artifacts at startup: {exc}")
    raise

# FastAPI app
app = FastAPI(
    title="FastIA — Loan Prediction API",
    description=(
        "API de prédiction du montant de prêt basée sur un réseau de neurones "
        "réentraîné avec détection de data drift."
    ),
    version="2.0.0",
)

# Pydantic schemas
class PredictionInput(BaseModel):
    age: float
    taille: float
    poids: float
    revenu_estime_mois: float
    sexe: str
    sport_licence: str
    niveau_etude: str
    region: str
    smoker: str
    nationalite_francaise: str = Field(..., alias="nationalité_francaise")

    model_config = {"populate_by_name": True}


class PredictionOutput(BaseModel):
    prediction: float
    model_version: str = "model_early_stopping"


# Routes
@app.get("/health", summary="Vérifier l'état de santé de l'API")
def health():
    """Retourne le statut du service et indique si le modèle est chargé."""
    model_loaded = model is not None
    preprocessor_loaded = preprocessor is not None
    status = "healthy" if (model_loaded and preprocessor_loaded) else "unhealthy"
    logger.info(f"Health check → status={status}")
    return {
        "status": status,
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "version": app.version,
    }


@app.post("/predict", response_model=PredictionOutput, summary="Prédire le montant du prêt")
def predict(data: PredictionInput):
    """
    Accepte les caractéristiques d'un client et retourne le montant de prêt prédit.

    Les champs attendus correspondent aux colonnes du jeu de données d'entraînement
    (hors `nom`, `prenom` et `montant_pret`).
    """
    logger.info(f"POST /predict — input: {data.model_dump(by_alias=True)}")
    try:
        # Build DataFrame with the exact column order expected by the preprocessor
        row = {
            "age": data.age,
            "taille": data.taille,
            "poids": data.poids,
            "revenu_estime_mois": data.revenu_estime_mois,
            "sexe": data.sexe,
            "sport_licence": data.sport_licence,
            "niveau_etude": data.niveau_etude,
            "region": data.region,
            "smoker": data.smoker,
            "nationalité_francaise": data.nationalite_francaise,
        }
        df = pd.DataFrame([row])
        X = preprocessor.transform(df)
        y_pred = model.predict(X)
        prediction = float(np.array(y_pred).flatten()[0])
        logger.info(f"POST /predict — prediction={prediction:.2f}")
        return PredictionOutput(prediction=prediction)
    except Exception as exc:
        logger.error(f"POST /predict — error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/retrain", summary="Réentraîner le modèle avec de nouvelles données")
async def retrain(file: UploadFile = File(...)):
    """
    Accepte un fichier CSV au même format que les données d'entraînement
    (avec les colonnes `nom`, `prenom`, `montant_pret` incluses) et réentraîne
    le modèle en place.

    Le modèle et le preprocessor sont mis à jour sur disque et en mémoire.
    """
    logger.info(f"POST /retrain — file: {file.filename}")
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        required_cols = {
            "age", "taille", "poids", "revenu_estime_mois",
            "sexe", "sport_licence", "niveau_etude", "region",
            "smoker", "nationalité_francaise", "montant_pret",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

        from modules.preprocess import preprocessing, split
        from models.models import create_nn_model, train_model

        X, y, new_preprocessor = preprocessing(df)
        X_train, X_test, y_train, y_test = split(X, y)

        input_dim = X_train.shape[1]
        new_model = create_nn_model(input_dim)
        new_model, history = train_model(
            new_model,
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=50,
            early_stopping=True,
            patience=10,
        )

        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(new_preprocessor, PREPROCESSOR_PATH)

        global model, preprocessor
        model = new_model
        preprocessor = new_preprocessor

        epochs_trained = len(history.history["loss"])
        logger.info(f"POST /retrain — completed after {epochs_trained} epochs")
        return {
            "status": "retrained",
            "epochs_trained": epochs_trained,
            "rows_used": len(df),
        }
    except ValueError as exc:
        logger.warning(f"POST /retrain — validation error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"POST /retrain — error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
