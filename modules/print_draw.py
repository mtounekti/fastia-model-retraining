import os
import matplotlib.pyplot as plt


def print_data(dico, exp_name="exp 1"):
    """
    Affiche les métriques de performance dans la console de manière formatée.

    Inputs:
        dico (dict): Dictionnaire {'MSE': float, 'MAE': float, 'R²': float}
                     retourné par evaluate_performance().
        exp_name (str): Nom de l'expérience affiché comme titre (défaut: "exp 1").

    Outputs:
        None (affichage console uniquement).
    """
    mse = dico["MSE"]
    mae = dico["MAE"]
    r2 = dico["R²"]
    print(f'{exp_name:=^60}')
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print("="*60)


def draw_loss(history):
    """
    Affiche interactivement les courbes de loss et val_loss d'un entraînement.

    Ces deux courbes permettent de détecter le surapprentissage (overfitting) :
    si val_loss remonte alors que loss continue de baisser, le modèle sur-apprend.

    Inputs:
        history (History): Objet retourné par model.fit() / train_model().
                           Doit contenir history.history['loss'] et ['val_loss'].

    Outputs:
        None (affichage interactif matplotlib).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Loss (Entraînement)')
    plt.plot(history.history['val_loss'], label='Val Loss (Validation)', linestyle='--')
    plt.title('Courbes de Loss et Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def save_loss_plot(history, exp_name, output_dir="plots"):
    """
    Sauvegarde les courbes de loss/val_loss dans un fichier PNG.

    Utilisé pour logger les graphiques comme artefacts MLflow.

    Inputs:
        history (History): Objet retourné par train_model().
        exp_name (str): Nom de l'expérience, utilisé pour nommer le fichier.
        output_dir (str): Dossier de destination des images (défaut: "plots").

    Outputs:
        fig_path (str): Chemin absolu du fichier PNG sauvegardé.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = exp_name.replace(" ", "_").replace("/", "-")
    fig_path = os.path.join(output_dir, f"loss_{safe_name}.png")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Loss (Entraînement)')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss (Validation)', linestyle='--')
    plt.title(f'Courbes de Loss — {exp_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return fig_path