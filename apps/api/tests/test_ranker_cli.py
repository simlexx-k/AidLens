from click import unstyle
from typer.main import get_command
from typer.testing import CliRunner

from app.ranker_cli import cli

runner = CliRunner()


def test_ranker_cli_exposes_split_train_evaluate_and_fusion_commands() -> None:
    result = runner.invoke(cli, ["--help"])
    output = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "split" in output
    assert "train" in output
    assert "evaluate" in output
    assert "sweep-fusion" in output
    assert "evaluate-fusion" in output
    assert "benchmark-serving" in output


def test_ranker_split_help_exposes_seed_and_output() -> None:
    result = runner.invoke(cli, ["split", "--help"])
    output = unstyle(result.stdout)

    assert result.exit_code == 0
    assert "--seed" in output
    assert "--output-dir" in output


def test_ranker_fusion_command_exposes_dev_calibration_controls() -> None:
    root = get_command(cli)
    command = root.commands["sweep-fusion"]
    parameter_names = {parameter.name for parameter in command.params}

    assert "alphas" in parameter_names
    assert "diversity_tolerance" in parameter_names
    assert "candidate_mode" in parameter_names


def test_ranker_fixed_fusion_command_requires_frozen_alpha() -> None:
    root = get_command(cli)
    command = root.commands["evaluate-fusion"]
    parameter_names = {parameter.name for parameter in command.params}

    assert "alpha" in parameter_names
    assert "candidate_mode" in parameter_names
    assert "alphas" not in parameter_names
    assert "diversity_tolerance" not in parameter_names


def test_serving_benchmark_exposes_repeatable_latency_controls() -> None:
    root = get_command(cli)
    command = root.commands["benchmark-serving"]
    parameter_names = {parameter.name for parameter in command.params}

    assert "api_url" in parameter_names
    assert "query" in parameter_names
    assert "repeats" in parameter_names
    assert "output" in parameter_names
