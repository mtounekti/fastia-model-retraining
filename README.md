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

## Lancer les tests

```bash
# Tous les tests
pytest tests/ -v

# Tests unitaires uniquement (sans TensorFlow, rapide)
pytest tests/test_evaluate.py tests/test_preprocess.py tests/test_print_draw.py -v

# Tests fonctionnels uniquement (utilisent les vraies données)
pytest tests/test_functional.py -v

# Tests d'intégration (pipeline bout en bout)
pytest tests/test_integration.py -v

# Un fichier spécifique
pytest tests/test_evaluate.py -v
pytest tests/test_preprocess.py -v
pytest tests/test_models.py -v
pytest tests/test_print_draw.py -v
pytest tests/test_integration.py -v
pytest tests/test_functional.py -v

# Avec rapport de couverture
pytest tests/ --cov=modules --cov=models --cov-report=term-missing
```

### Types de tests

| Fichier | Type | Description |
|---------|------|-------------|
| `test_evaluate.py` | Unitaire | Teste `evaluate_performance()` en isolation |
| `test_preprocess.py` | Unitaire | Teste `preprocessing()` et `split()` en isolation |
| `test_models.py` | Unitaire | Teste `create_nn_model()`, `train_model()`, `model_predict()` |
| `test_print_draw.py` | Unitaire | Teste `save_loss_plot()` avec des mocks |
| `test_integration.py` | Intégration | Pipeline complet avec données synthétiques |
| `test_functional.py` | **Fonctionnel** | Scénarios métier avec les vraies données (drift, réentraînement, fichiers) |

---

## Visualiser les résultats avec MLflow

```bash
mlflow ui
```

Ouvrir dans le navigateur : [http://localhost:5000](http://localhost:5000)

Dans l'interface MLflow, sélectionner l'expérience **OPCO-ATLAS-Loan-Prediction** pour comparer tous les runs.

---

## Description des expériences

| # | Nom | Modèle | Données | Réentraînement | R² Test | MSE Test | MAE Test |
|---|-----|--------|---------|----------------|---------|----------|----------|
| 1 | Exp1_Baseline | Original | df_old | Non | **0.7655** | 34 420 598 | 4 789 |
| 2 | Exp2_DataDrift | Original | df_new | Non | 0.6407 | 38 899 014 | 5 076 |
| 3 | Exp3_Retrain_OldData | Original | df_old | Oui (50 epochs) | **0.8020** | 29 064 990 | 4 112 |
| 4 | Exp4_Retrain_NewData | Original | df_new | Oui (50 epochs) | 0.5985 | 43 467 896 | 5 177 |
| 5 | Exp5_FreshModel | Vierge | df_new | Oui (100 epochs) | 0.5815 | 45 302 075 | 5 281 |

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

## Captures d'écran MLflow

<img width="1863" height="905" alt="Capture d’écran 2026-03-17 à 13 30 56" src="https://github.com/user-attachments/assets/fb709e65-50ea-415a-80a4-299b2b870093" />

<img width="1863" height="905" alt="image" src="https://github.com/user-attachments/assets/164396dc-5651-42e5-937c-029d8afce732" />

<img width="1863" height="905" alt="image" src="https://github.com/user-attachments/assets/76c0508a-3205-4ef2-add6-5424ee29ec8f" />

---

## Conclusions

### 1. La dérive de données est confirmée (Exp1 vs Exp2)
Le modèle original passe de **R²=0.77** sur df_old à **R²=0.64** sur df_new sans réentraînement.
La dégradation est significative (+4 300€ d'erreur moyenne) — la dérive est réelle et nécessite une action.

### 2. Réentraîner sur les mêmes données améliore légèrement (Exp3)
**R²=0.80** — le modèle consolide ce qu'il sait déjà (fine-tuning).
Répondre à la question du brief : *"si on réentraîne plusieurs fois avec les mêmes données, on fait du fine-tuning"*, avec un risque d'overfitting si on pousse trop d'epochs.

### 3. Le réentraînement sur nouvelles données est décevant (Exp4)
**R²=0.60** — moins bon que la baseline. Le modèle souffre de *catastrophic forgetting* : en apprenant les nouvelles données, il "oublie" les patterns des anciennes. 50 epochs ne suffisent pas pour reconverger.

### 4. Un nouveau modèle vierge n'est pas meilleur (Exp5)
**R²=0.58** — avec seulement 100 epochs, le modèle n'a pas eu le temps de converger sur les nouvelles données.

### Recommandation
Combiner df_old + df_new pour l'entraînement, ou augmenter significativement le nombre d'epochs (200-300) pour Exp4 et Exp5 afin de laisser le modèle converger sur les nouvelles données.

---

## Auteur

Projet réalisé dans le cadre de la formation **FastIA — Module 1, Brief 1**
Maroua Tounekti
