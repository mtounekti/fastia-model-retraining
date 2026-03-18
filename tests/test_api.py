"""
test_api.py — Tests fonctionnels des routes FastAPI.

Vérifie :
- /health : statut du service et structure de la réponse
- /predict : prédiction valide, données manquantes, champs invalides
- /retrain : réentraînement avec un CSV valide et avec un CSV invalide
"""

import io
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app


client = TestClient(app)

# ---------------------------------------------------------------------------
# Payload de prédiction valide (réutilisé dans plusieurs tests)
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
    "age": 35.0,
    "taille": 175.0,
    "poids": 75.0,
    "revenu_estime_mois": 2500.0,
    "sexe": "M",
    "sport_licence": "oui",
    "niveau_etude": "bac+3",
    "region": "IDF",
    "smoker": "no",
    "nationalité_francaise": "oui",
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_200(self):
        """La route /health répond avec un code 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_is_healthy(self):
        """Le statut retourné doit être 'healthy' quand le modèle est chargé."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_model_loaded(self):
        """La réponse confirme que le modèle et le preprocessor sont chargés."""
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["preprocessor_loaded"] is True

    def test_health_has_version(self):
        """La réponse contient le champ 'version'."""
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------
class TestPredict:
    def test_predict_returns_200(self):
        """Une requête valide retourne un code 200."""
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_returns_float(self):
        """La prédiction retournée est un nombre flottant."""
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)

    def test_predict_positive_amount(self):
        """Le montant prédit doit être positif (montant de prêt)."""
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["prediction"] > 0

    def test_predict_has_model_version(self):
        """La réponse contient le champ 'model_version'."""
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "model_version" in data

    def test_predict_missing_field_returns_422(self):
        """Une requête avec un champ obligatoire manquant retourne 422."""
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_numeric_type_returns_422(self):
        """Un champ numérique avec une valeur non convertible retourne 422."""
        payload = {**VALID_PAYLOAD, "age": "not_a_number"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_is_deterministic(self):
        """Deux appels identiques retournent la même prédiction."""
        pred1 = client.post("/predict", json=VALID_PAYLOAD).json()["prediction"]
        pred2 = client.post("/predict", json=VALID_PAYLOAD).json()["prediction"]
        assert pred1 == pred2

    def test_predict_different_inputs_differ(self):
        """Des revenus très différents produisent des prédictions différentes."""
        low_income = {**VALID_PAYLOAD, "revenu_estime_mois": 500.0}
        high_income = {**VALID_PAYLOAD, "revenu_estime_mois": 10000.0}
        pred_low = client.post("/predict", json=low_income).json()["prediction"]
        pred_high = client.post("/predict", json=high_income).json()["prediction"]
        assert pred_low != pred_high

    def test_predict_unknown_category_handled(self):
        """Une catégorie inconnue est gérée sans erreur (handle_unknown='ignore')."""
        payload = {**VALID_PAYLOAD, "region": "UNKNOWN_REGION"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /retrain
# ---------------------------------------------------------------------------
def _make_csv_bytes(n: int = 80) -> bytes:
    """Génère un CSV synthétique valide pour le réentraînement."""
    np.random.seed(0)
    df = pd.DataFrame({
        "nom":                   [f"Nom{i}" for i in range(n)],
        "prenom":                [f"Prenom{i}" for i in range(n)],
        "age":                   np.random.randint(18, 65, n).astype(float),
        "taille":                np.random.randint(150, 200, n).astype(float),
        "poids":                 np.random.randint(50, 100, n).astype(float),
        "revenu_estime_mois":    np.random.uniform(1000, 5000, n),
        "sexe":                  np.random.choice(["M", "F"], n),
        "sport_licence":         np.random.choice(["oui", "non"], n),
        "niveau_etude":          np.random.choice(["bac", "bac+2", "bac+3", "bac+5"], n),
        "region":                np.random.choice(["IDF", "PACA", "ARA", "OCC"], n),
        "smoker":                np.random.choice(["yes", "no"], n),
        "nationalité_francaise": np.random.choice(["oui", "non"], n),
        "montant_pret":          np.random.uniform(500, 50000, n),
    })
    return df.to_csv(index=False).encode("utf-8")


class TestRetrain:
    def test_retrain_returns_200(self):
        """Un CSV valide déclenche un réentraînement et retourne 200."""
        csv_bytes = _make_csv_bytes()
        response = client.post(
            "/retrain",
            files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert response.status_code == 200

    def test_retrain_response_structure(self):
        """La réponse contient 'status', 'epochs_trained' et 'rows_used'."""
        csv_bytes = _make_csv_bytes()
        data = client.post(
            "/retrain",
            files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
        ).json()
        assert data["status"] == "retrained"
        assert "epochs_trained" in data
        assert "rows_used" in data

    def test_retrain_rows_used_matches_csv(self):
        """Le nombre de lignes utilisées correspond à la taille du CSV."""
        n = 60
        csv_bytes = _make_csv_bytes(n)
        data = client.post(
            "/retrain",
            files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
        ).json()
        assert data["rows_used"] == n

    def test_retrain_missing_column_returns_422(self):
        """Un CSV sans la colonne 'montant_pret' retourne 422."""
        df = pd.DataFrame({
            "age": [30.0], "taille": [170.0], "poids": [70.0],
            "revenu_estime_mois": [2000.0], "sexe": ["M"],
            "sport_licence": ["oui"], "niveau_etude": ["bac+3"],
            "region": ["IDF"], "smoker": ["no"],
            "nationalité_francaise": ["oui"],
            # montant_pret manquant intentionnellement
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        response = client.post(
            "/retrain",
            files={"file": ("bad.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert response.status_code == 422

    def test_retrain_model_still_predicts_after(self):
        """Après réentraînement, /predict continue de retourner une prédiction valide."""
        csv_bytes = _make_csv_bytes()
        client.post(
            "/retrain",
            files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        assert response.json()["prediction"] > 0
