# Bienvenue dans ce projet d'IA

Installation : 

    python -m venv .venv


Activation :

Windows (PowerShell) :

    .\.venv\Scripts\Activate.ps1

Windows (CMD) :

    .\.venv\Scripts\activate.bat

macOS / Linux :

    source .venv/bin/activate


Installation des dépendances : 

    pip install -r requirements.txt


# Le modèle d'IA

Nous avons ici un réseau de neurones (NN) avec :
2 couches "denses"
1 couche de prédiction

# Architecture du projet

    .
    ├── data/
    │   ├── df_new.csv
    │   └── df_old.csv
    ├── models/
    │   ├── models.py
    │   ├── model_2024_08.pkl
    │   └── preprocessor.pkl
    ├── modules/
    │   ├── evaluate.py
    │   ├── preprocess.py
    │   └── print_draw.py
    ├── .gitignore
    ├── README.md
    ├── main.py
    └── requirements.txt


data/ 

C'est là que sont stockées les données.

    df_new.csv : Les données fraîches du jour, prêtes à être dévorées par notre IA.
    df_old.csv : Les anciennes données ayant servis à l'apprentissage initial du modèle

models/ 

C'est là que sont stockés les modèles

    models.py : C'est ici que l'on définit l'architecture de notre NN.
    model_2024_08.pkl : Une version sauvegardée de notre modèle. 
    preprocessor.pkl : le preparateur de données

modules/ 

C'est là que le code python se découpe en modules ayant chaque une tâche définie

    evaluate.py : Module d'évaluation du modèle.
    preprocess.py : Module de préparation des données
    print_draw.py : Module d'affichage des résultats


👤 Auteur

Projet réalisé dans le cadre de la formation FastIA — Module 0, Brief 1 - by Maroua Tounekti