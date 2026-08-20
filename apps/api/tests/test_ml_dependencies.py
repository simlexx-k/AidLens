import tomllib
from pathlib import Path


def _optional_dependencies() -> dict[str, list[str]]:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return config["project"]["optional-dependencies"]


def test_ml_extra_includes_sentence_transformers_training_dependencies() -> None:
    dependencies = _optional_dependencies()["ml"]

    assert any(
        dependency.startswith("sentence-transformers[train]")
        for dependency in dependencies
    )


def test_accelerated_runtime_extras_are_explicit() -> None:
    optional = _optional_dependencies()

    assert any(
        dependency.startswith("sentence-transformers[onnx]")
        for dependency in optional["onnx"]
    )
    assert any(
        dependency.startswith("sentence-transformers[openvino]")
        for dependency in optional["openvino"]
    )
