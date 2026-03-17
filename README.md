# OPCO ATLAS — Module 1, Brief 1 : Réentraînement d'un modèle IA

Projet de prédiction du montant de prêt bancaire avec réentraînement du modèle et suivi des performances via **MLflow**.

---

## Contexte

Un modèle de réseau de neurones a été mis en production en août 2024 (`model_2024_08.pkl`).
Au fil du temps, les données du monde réel ont évolué : le modèle souffre de **dérive de données** (*data drift*).
Ce projet vise à :
1. Mesurer la dégradation du modèle sur les nouvelles données.
2. Réentraîner le modèle avec les données récentes.
3. Comparer les performances de toutes les stratégies via MLflow.

---

## Architecture du projet

```
.
├── data/
│   ├── df_old.csv              # Données d'entraînement initial (2024)
│   └── df_new.csv              # Nouvelles données (données récentes)
├── models/
│   ├── models.py               # Architecture du réseau de neurones
│   ├── model_2024_08.pkl       # Modèle original entraîné en août 2024
│   ├── model_retrained_new_data.pkl   # Modèle réentraîné (Exp 4) — généré par train.py
│   ├── model_fresh_new_data.pkl       # Nouveau modèle vierge (Exp 5) — généré par train.py
│   └── preprocessor.pkl        # Préprocesseur original fitté sur df_old
├── modules/
│   ├── evaluate.py             # Calcul des métriques de performance
│   ├── preprocess.py           # Prétraitement des données
│   └── print_draw.py           # Affichage et sauvegarde des graphiques
├── plots/                      # Courbes de loss générées par train.py
├── mlruns/                     # Données MLflow (générées automatiquement)
├── main.py                     # Script original du prédécesseur (référence)
├── train.py                    # Script principal : expériences + MLflow tracking
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Créer l'environnement virtuel

```bash
python -m venv .venv
```

### 2. Activer l'environnement

- **macOS / Linux** : `source .venv/bin/activate`
- **Windows (PowerShell)** : `.\.venv\Scripts\Activate.ps1`
- **Windows (CMD)** : `.\.venv\Scripts\activate.bat`

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Lancer les expériences

```bash
python train.py
```

Ce script exécute 5 expériences et les enregistre automatiquement dans MLflow.

---

## Visualiser les résultats avec MLflow

```bash
mlflow ui
```

Ouvrir dans le navigateur : [http://localhost:5000](http://localhost:5000)

Dans l'interface MLflow, sélectionner l'expérience **OPCO-ATLAS-Loan-Prediction** pour comparer tous les runs.

---

## Description des expériences

| # | Nom | Modèle | Données | Réentraînement | Objectif |
|---|-----|--------|---------|----------------|----------|
| 1 | Exp1_Baseline | Original | df_old | Non | Référence de performance initiale |
| 2 | Exp2_DataDrift | Original | df_new | Non | Mesurer la dérive des données |
| 3 | Exp3_Retrain_OldData | Original | df_old | Oui (50 epochs) | Illustrer le surapprentissage |
| 4 | Exp4_Retrain_NewData | Original | df_new | Oui (50 epochs) | **Réentraînement principal** |
| 5 | Exp5_FreshModel | Vierge | df_new | Oui (100 epochs) | Nouveau modèle depuis zéro |

---

## Métriques suivies

### MSE — Mean Squared Error
Moyenne des **erreurs au carré** entre les valeurs prédites et réelles.
Très sensible aux valeurs aberrantes (outliers). **Plus bas = meilleur.**

### MAE — Mean Absolute Error
Moyenne des **erreurs absolues**. Même unité que la cible (euros).
Plus interprétable que le MSE. **Plus bas = meilleur.**

### R² — Coefficient de Détermination
Mesure la **proportion de variance expliquée** par le modèle.
- R² = 1 : prédiction parfaite
- R² = 0 : le modèle est aussi performant que prédire la moyenne
- R² < 0 : le modèle est moins bon que la moyenne

**Plus proche de 1 = meilleur.**

---

## Courbes de loss

Deux courbes sont tracées lors de chaque entraînement :

- **Loss (entraînement)** : erreur sur les données d'entraînement — doit diminuer.
- **Val Loss (validation)** : erreur sur les données de test — doit suivre la loss.

Si la val_loss remonte alors que la loss continue de baisser : **overfitting** (surapprentissage).
Les courbes sont sauvegardées dans `plots/` et loggées comme artefacts MLflow.

---

## Architecture du modèle

Réseau de neurones feedforward (fully connected) pour la régression :

```
Input (n features)
    ↓
Dense(64, ReLU)
    ↓
Dense(32, ReLU)
    ↓
Dense(1)  ← montant_pret prédit
```

- **Optimizer** : Adam
- **Loss** : Mean Squared Error (MSE)

---

## captures écrans

<img width="1863" height="905" alt="Capture d’écran 2026-03-17 à 13 30 56" src="https://github.com/user-attachments/assets/fb709e65-50ea-415a-80a4-299b2b870093" />



---

## Conclusions attendues

L'analyse MLflow doit permettre de répondre aux questions suivantes :

1. **Exp 2 vs Exp 1** : le modèle se dégrade-t-il sur les nouvelles données ? → Dérive de données
2. **Exp 3 vs Exp 1** : que se passe-t-il quand on réentraîne sur les mêmes données ? → Légère amélioration ou overfitting
3. **Exp 4 vs Exp 2** : le réentraînement corrige-t-il la dérive ? → Objectif principal
4. **Exp 5 vs Exp 4** : vaut-il mieux réentraîner l'ancien modèle ou repartir de zéro ?

---

## Auteur

Projet réalisé dans le cadre de la formation **FastIA — Module 1, Brief 1**
Maroua Tounekti
