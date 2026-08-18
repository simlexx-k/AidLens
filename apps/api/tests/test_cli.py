from typer.testing import CliRunner

from app.cli import cli

runner = CliRunner()


def test_cli_exposes_ingest_subcommand() -> None:
    result = runner.invoke(cli, ["ingest", "--help"])

    assert result.exit_code == 0
    assert "--pages" in result.stdout
    assert "--start-page" in result.stdout
    assert "--concurrency" in result.stdout
