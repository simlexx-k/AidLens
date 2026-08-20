from app.core.config import Settings


def test_aidranker_serving_defaults_keep_validated_pool_and_batch() -> None:
    settings = Settings(database_url="postgresql+asyncpg://aidlens@db:5432/aidlens")

    assert settings.aidranker_candidate_k == 40
    assert settings.aidranker_batch_size == 8
    assert settings.aidranker_device == "auto"
    assert settings.aidranker_warmup is True
