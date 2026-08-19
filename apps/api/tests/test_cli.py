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
    assert "compile-labels" in output


def test_candidate_export_exposes_diversity_control() -> None:
    result = runner.invoke(cli, ["export-ranking-candidates", "--help"])
    output = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "--max-per-evaluation" in output


def test_compile_labels_exposes_outputs() -> None:
    result = runner.invoke(cli, ["compile-labels", "--help"])
    output = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "--judgments-output" in output
    assert "--ranker-output" in output
