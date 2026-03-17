from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_performance(y_true, y_pred):
    """
    Calcule trois métriques de performance pour un modèle de régression.

    Métriques :
        - MSE (Mean Squared Error) : Moyenne des erreurs au carré.
          Très sensible aux valeurs aberrantes. Plus il est bas, mieux c'est.
        - MAE (Mean Absolute Error) : Moyenne des erreurs absolues.
          Plus interprétable que le MSE (même unité que la cible).
        - R² (coefficient de détermination) : Proportion de la variance expliquée
          par le modèle. Varie de -∞ à 1 (1 = prédiction parfaite, 0 = moyenne naïve).

    Inputs:
        y_true (pd.Series ou np.ndarray): Valeurs réelles du montant_pret.
        y_pred (np.ndarray): Valeurs prédites par le modèle.

    Outputs:
        dict: Dictionnaire {'MSE': float, 'MAE': float, 'R²': float}.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae, 'R²': r2} 