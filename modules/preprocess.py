from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def split(X, y, test_size=0.2, random_state=42):
    """
    Divise le jeu de données en ensembles d'entraînement et de test.

    Inputs:
        X (np.ndarray): Features prétraitées.
        y (pd.Series): Labels cibles (montant_pret).
        test_size (float): Proportion du jeu de test (défaut: 0.2 → 80/20).
        random_state (int): Graine aléatoire pour la reproductibilité (défaut: 42).

    Outputs:
        X_train, X_test (np.ndarray): Features d'entraînement et de test.
        y_train, y_test (pd.Series): Labels d'entraînement et de test.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def preprocessing(df):
    """
    Effectue le prétraitement complet d'un DataFrame pour la modélisation.

    Étapes :
        - Colonnes numériques (age, taille, poids, revenu_estime_mois) :
          imputation par la moyenne + standardisation (StandardScaler).
        - Colonnes catégorielles (sexe, sport_licence, niveau_etude, region,
          smoker, nationalité_francaise) : imputation par la valeur la plus
          fréquente + encodage one-hot (valeurs inconnues ignorées).
        - Suppression des colonnes non-features : nom, prenom, montant_pret.

    Inputs:
        df (pd.DataFrame): DataFrame brut chargé depuis df_old.csv ou df_new.csv.

    Outputs:
        X_processed (np.ndarray): Features numériques et encodées, prêtes pour le modèle.
        y (pd.Series): Colonne cible montant_pret.
        preprocessor (ColumnTransformer): Pipeline de prétraitement fitté sur df.
                                          À sauvegarder pour réutilisation sur de nouvelles données.
    """
    numerical_cols = ["age", "taille", "poids", "revenu_estime_mois"]
    categorical_cols = ["sexe", "sport_licence", "niveau_etude", "region", "smoker", "nationalité_francaise"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    # Prétraitement
    X = df.drop(columns=["nom", "prenom", "montant_pret"])
    y = df["montant_pret"]

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor