from click import unstyle
from typer.testing import CliRunner

from app.ranker_cli import cli

runner = CliRunner()


def test_ranker_cli_exposes_split_train_and_evaluate() -> None:
    result = runner.invoke(cli, ["--help"])
    output = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "split" in output
    assert "train" in output
    assert "evaluate" in output


def test_ranker_split_help_exposes_seed_and_output() -> None:
    result = runner.invoke(cli, ["split", "--help"])
    output = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "--seed" in output
    assert "--output-dir" in output
