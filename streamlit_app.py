import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="FastIA — Prédiction de prêt", page_icon="💰", layout="centered")
st.title("💰 FastIA — Prédiction du montant de prêt")

# Sidebar — statut de l'API
with st.sidebar:
    st.header("État du service")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        if health["status"] == "healthy":
            st.success("API en ligne ✅")
        else:
            st.error("API dégradée ⚠️")
        st.caption(f"Version : {health.get('version', '—')}")
    except Exception:
        st.error("API inaccessible ❌")
        st.caption(f"Vérifiez que l'API tourne sur {API_URL}")

# Tabs
tab_predict, tab_retrain = st.tabs(["🔮 Prédiction", "🔄 Réentraînement"])

# Tab Prédiction
with tab_predict:
    st.subheader("Informations client")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=35)
        taille = st.number_input("Taille (cm)", min_value=100, max_value=250, value=175)
        poids = st.number_input("Poids (kg)", min_value=30, max_value=200, value=75)
        revenu = st.number_input("Revenu mensuel estimé (€)", min_value=0, value=2500, step=100)

    with col2:
        sexe = st.selectbox("Sexe", ["M", "F"])
        sport = st.selectbox("Licence sportive", ["oui", "non"])
        etude = st.selectbox("Niveau d'étude", ["bac", "bac+2", "bac+3", "bac+5"])
        region = st.selectbox("Région", ["IDF", "PACA", "ARA", "OCC", "BRE", "NAQ", "HDF"])
        smoker = st.selectbox("Fumeur", ["no", "yes"])
        nationalite = st.selectbox("Nationalité française", ["oui", "non"])

    if st.button("Prédire le montant du prêt", type="primary", use_container_width=True):
        payload = {
            "age": float(age),
            "taille": float(taille),
            "poids": float(poids),
            "revenu_estime_mois": float(revenu),
            "sexe": sexe,
            "sport_licence": sport,
            "niveau_etude": etude,
            "region": region,
            "smoker": smoker,
            "nationalité_francaise": nationalite,
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"### Montant prédit : {prediction:,.2f} €")
                st.caption(f"Modèle : {response.json()['model_version']}")
            else:
                st.error(f"Erreur API ({response.status_code}) : {response.json().get('detail')}")
        except Exception as exc:
            st.error(f"Impossible de contacter l'API : {exc}")

# Tab Réentraînement
with tab_retrain:
    st.subheader("Réentraîner le modèle")
    st.info(
        "Uploadez un fichier CSV au même format que les données d'entraînement "
        "(colonnes : nom, prenom, age, taille, poids, revenu_estime_mois, sexe, "
        "sport_licence, niveau_etude, region, smoker, nationalité_francaise, montant_pret)."
    )

    uploaded = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if uploaded and st.button("Lancer le réentraînement", type="primary", use_container_width=True):
        with st.spinner("Réentraînement en cours…"):
            try:
                response = requests.post(
                    f"{API_URL}/retrain",
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                    timeout=300,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success("Réentraînement terminé ✅")
                    st.metric("Époques entraînées", data["epochs_trained"])
                    st.metric("Lignes utilisées", data["rows_used"])
                else:
                    st.error(f"Erreur ({response.status_code}) : {response.json().get('detail')}")
            except Exception as exc:
                st.error(f"Impossible de contacter l'API : {exc}")
