import tomllib
from pathlib import Path


def test_ml_extra_includes_sentence_transformers_training_dependencies() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    ml_dependencies = config["project"]["optional-dependencies"]["ml"]

    assert any(
        dependency.startswith("sentence-transformers[train]")
        for dependency in ml_dependencies
    )
