from click import unstyle
from typer.testing import CliRunner

from app.ranker_cli import cli

runner = CliRunner()


def test_ranker_cli_exposes_split_train_evaluate_and_fusion_sweep() -> None:
    result = runner.invoke(cli, ["--help"])
    output = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "split" in output
    assert "train" in output
    assert "evaluate" in output
    assert "sweep-fusion" in output


def test_ranker_split_help_exposes_seed_and_output() -> None:
    result = runner.invoke(cli, ["split", "--help"])
    output = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "--seed" in output
    assert "--output-dir" in output


def test_ranker_fusion_help_exposes_dev_calibration_controls() -> None:
    result = runner.invoke(cli, ["sweep-fusion", "--help"])
    output = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "--alphas" in output
    assert "--diversity-tolerance" in output
    assert "--candidate-mode" in output
