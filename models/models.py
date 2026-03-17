from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_nn_model(input_dim):
    """
    Crée et compile un réseau de neurones feedforward pour la régression.

    Architecture :
        - Couche Dense 1 : 64 neurones, activation ReLU
        - Couche Dense 2 : 32 neurones, activation ReLU
        - Couche de sortie : 1 neurone (prédiction du montant du prêt)

    Inputs:
        input_dim (int): Nombre de features en entrée (après prétraitement).

    Outputs:
        model (Sequential): Modèle Keras compilé avec optimizer Adam et loss MSE.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(model, X, y, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=0):
    """
    Entraîne un modèle Keras sur les données fournies.

    Inputs:
        model (Sequential): Modèle Keras à entraîner.
        X (np.ndarray): Features d'entraînement prétraitées.
        y (pd.Series): Labels cibles (montant_pret).
        X_val (np.ndarray, optional): Features de validation.
        y_val (pd.Series, optional): Labels de validation.
        epochs (int): Nombre d'époques d'entraînement (défaut: 50).
        batch_size (int): Taille des mini-batches (défaut: 32).
        verbose (int): Niveau de verbosité Keras (0=silencieux, 1=barre de progression).

    Outputs:
        model (Sequential): Modèle entraîné (modifié in-place).
        hist (History): Historique Keras contenant les métriques par époque
                        (loss, val_loss). Utilisé pour tracer les courbes d'apprentissage.
    """
    hist = model.fit(
        X, y,
        validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
        epochs=epochs, batch_size=batch_size, verbose=verbose
    )
    return model, hist


def model_predict(model, X):
    """
    Réalise des prédictions avec le modèle sur un jeu de données.

    Inputs:
        model (Sequential): Modèle Keras entraîné.
        X (np.ndarray): Features prétraitées sur lesquelles prédire.

    Outputs:
        y_pred (np.ndarray): Tableau 1D des montants de prêt prédits.
    """
    y_pred = model.predict(X, verbose=0).flatten()
    return y_pred