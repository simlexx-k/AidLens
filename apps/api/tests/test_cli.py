from click import unstyle
from typer.testing import CliRunner

from app.cli import cli

runner = CliRunner()


def test_cli_exposes_ingest_subcommand() -> None:
    result = runner.invoke(cli, ["ingest", "--help"])
    output = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "--pages" in output
    assert "--start-page" in output
    assert "--concurrency" in output
    assert "--skip-existing" in output


def test_cli_exposes_semantic_and_evaluation_commands() -> None:
    result = runner.invoke(cli, ["--help"])
    output = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "embed" in output
    assert "corpus-report" in output
    assert "benchmark" in output
    assert "export-ranking-candidates" in output
