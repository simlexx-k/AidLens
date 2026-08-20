from app.services.ranker.serving import AidRankerService


def test_local_ranker_artifact_fingerprint_is_stable(tmp_path) -> None:
    model_dir = tmp_path / "ranker"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model":"test"}')
    (model_dir / "model.safetensors").write_bytes(b"weights")

    first = AidRankerService(str(model_dir)).artifact_fingerprint
    second = AidRankerService(str(model_dir)).artifact_fingerprint

    assert first is not None
    assert first.startswith("sha256:")
    assert first == second


def test_remote_ranker_reference_has_no_local_fingerprint() -> None:
    service = AidRankerService("org/model-name")

    assert service.artifact_fingerprint is None
