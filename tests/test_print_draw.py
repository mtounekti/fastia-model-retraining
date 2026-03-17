"""
test_print_draw.py — Tests fonctionnels pour modules/print_draw.py

Fonctions testées :
    - print_data(dico, exp_name)
    - save_loss_plot(history, exp_name, output_dir)
"""

import os
import pytest
from modules.print_draw import print_data, save_loss_plot


# ---------------------------------------------------------------------------
# print_data()
# ---------------------------------------------------------------------------

class TestPrintData:

    def test_prints_without_error(self, capsys):
        """print_data ne doit pas lever d'exception avec un dict valide."""
        dico = {"MSE": 100.0, "MAE": 8.5, "R²": 0.92}
        print_data(dico)

    def test_output_contains_mse(self, capsys):
        """La sortie console doit contenir 'MSE'."""
        dico = {"MSE": 100.0, "MAE": 8.5, "R²": 0.92}
        print_data(dico)
        captured = capsys.readouterr()
        assert "MSE" in captured.out

    def test_output_contains_mae(self, capsys):
        """La sortie console doit contenir 'MAE'."""
        dico = {"MSE": 100.0, "MAE": 8.5, "R²": 0.92}
        print_data(dico)
        captured = capsys.readouterr()
        assert "MAE" in captured.out

    def test_output_contains_r2(self, capsys):
        """La sortie console doit contenir 'R²'."""
        dico = {"MSE": 100.0, "MAE": 8.5, "R²": 0.92}
        print_data(dico)
        captured = capsys.readouterr()
        assert "R²" in captured.out

    def test_output_contains_exp_name(self, capsys):
        """La sortie console doit afficher le nom de l'expérience."""
        dico = {"MSE": 50.0, "MAE": 5.0, "R²": 0.85}
        print_data(dico, exp_name="mon_experience")
        captured = capsys.readouterr()
        assert "mon_experience" in captured.out

    def test_default_exp_name_is_exp1(self, capsys):
        """Le nom par défaut de l'expérience doit être 'exp 1'."""
        dico = {"MSE": 50.0, "MAE": 5.0, "R²": 0.85}
        print_data(dico)
        captured = capsys.readouterr()
        assert "exp 1" in captured.out


# ---------------------------------------------------------------------------
# save_loss_plot()
# ---------------------------------------------------------------------------

class TestSaveLossPlot:

    def test_returns_a_string_path(self, mock_history, tmp_path):
        """save_loss_plot doit retourner un chemin (str)."""
        path = save_loss_plot(mock_history, "test_exp", output_dir=str(tmp_path))
        assert isinstance(path, str)

    def test_file_is_created(self, mock_history, tmp_path):
        """Le fichier PNG doit exister après appel."""
        path = save_loss_plot(mock_history, "test_exp", output_dir=str(tmp_path))
        assert os.path.isfile(path)

    def test_file_has_png_extension(self, mock_history, tmp_path):
        """Le fichier généré doit avoir l'extension .png."""
        path = save_loss_plot(mock_history, "test_exp", output_dir=str(tmp_path))
        assert path.endswith(".png")

    def test_creates_output_dir_if_not_exists(self, mock_history, tmp_path):
        """Le dossier de sortie doit être créé automatiquement s'il n'existe pas."""
        new_dir = str(tmp_path / "new_subdir" / "plots")
        assert not os.path.exists(new_dir)
        save_loss_plot(mock_history, "test_exp", output_dir=new_dir)
        assert os.path.isdir(new_dir)

    def test_file_name_contains_exp_name(self, mock_history, tmp_path):
        """Le nom du fichier doit contenir le nom de l'expérience."""
        exp_name = "Exp4_Retrain"
        path = save_loss_plot(mock_history, exp_name, output_dir=str(tmp_path))
        assert exp_name in os.path.basename(path)

    def test_works_without_val_loss(self, mock_history_no_val, tmp_path):
        """save_loss_plot ne doit pas planter si val_loss est absent de l'historique."""
        path = save_loss_plot(mock_history_no_val, "no_val_exp", output_dir=str(tmp_path))
        assert os.path.isfile(path)

    def test_spaces_in_exp_name_are_replaced(self, mock_history, tmp_path):
        """Les espaces dans le nom d'expérience doivent être remplacés pour éviter les chemins invalides."""
        path = save_loss_plot(mock_history, "exp avec espaces", output_dir=str(tmp_path))
        assert " " not in os.path.basename(path)

    def test_file_is_not_empty(self, mock_history, tmp_path):
        """Le fichier PNG généré ne doit pas être vide."""
        path = save_loss_plot(mock_history, "test_exp", output_dir=str(tmp_path))
        assert os.path.getsize(path) > 0
