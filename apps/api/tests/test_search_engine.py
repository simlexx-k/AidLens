import asyncio
import uuid

import pytest

from app.schemas.evaluation import EvidenceSearchHit, EvidenceSearchRequest
from app.services.search import engine


def _hit(source: str, evaluation_id: str | None = None) -> EvidenceSearchHit:
    return EvidenceSearchHit(
        chunk_id=uuid.uuid4(),
        evaluation_id=evaluation_id or f"eval-{source}",
        title=f"{source.title()} result",
        text="Relevant evidence passage.",
        score=1.0,
        lexical_score=1.0 if source == "lexical" else None,
        semantic_score=1.0 if source == "semantic" else None,
        retrieval_sources=[source],
        source_url="https://example.test/evaluation",
    )


@pytest.mark.asyncio
async def test_hybrid_search_serializes_shared_session_operations(monkeypatch) -> None:
    active_operations = 0

    async def lexical_search(session, payload):
        nonlocal active_operations
        assert active_operations == 0
        active_operations += 1
        await asyncio.sleep(0)
        active_operations -= 1
        return [_hit("lexical")]

    async def semantic_search(session, payload, query_vector):
        nonlocal active_operations
        assert active_operations == 0
        active_operations += 1
        await asyncio.sleep(0)
        active_operations -= 1
        return [_hit("semantic")]

    monkeypatch.setattr(engine, "lexical_search", lexical_search)
    monkeypatch.setattr(engine, "semantic_search", semantic_search)

    response = await engine.execute_search(
        object(),
        EvidenceSearchRequest(query="food security", mode="hybrid", top_k=10),
        query_vector=[0.1, 0.2],
        embedding_model="test-model",
    )

    assert response.mode == "hybrid"
    assert len(response.hits) == 2
    assert len(response.groups) == 2
    assert response.ranking_pipeline == ["lexical", "semantic", "rrf"]
    assert response.first_stage_latency_ms is not None
    assert response.total_search_latency_ms is not None
    assert active_operations == 0


@pytest.mark.asyncio
async def test_lexical_search_can_cap_results_per_evaluation(monkeypatch) -> None:
    async def lexical_search(session, payload):
        assert payload.top_k >= 5
        return [
            _hit("lexical", "A"),
            _hit("lexical", "A"),
            _hit("lexical", "A"),
            _hit("lexical", "B"),
            _hit("lexical", "C"),
        ]

    monkeypatch.setattr(engine, "lexical_search", lexical_search)

    response = await engine.execute_search(
        object(),
        EvidenceSearchRequest(
            query="education access",
            mode="lexical",
            top_k=3,
            max_per_evaluation=1,
        ),
    )

    assert response.max_per_evaluation == 1
    assert [hit.evaluation_id for hit in response.hits] == ["A", "B", "C"]
    assert response.ranking_pipeline == ["lexical", "diversity:max-1-per-report"]
    assert [group.evaluation_id for group in response.groups] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_auto_uses_semantic_aidranker_before_diversity(monkeypatch) -> None:
    semantic_calls = 0

    async def semantic_search(session, payload, query_vector):
        nonlocal semantic_calls
        semantic_calls += 1
        assert payload.top_k == 40
        return [
            _hit("semantic", "A"),
            _hit("semantic", "A"),
            _hit("semantic", "B"),
        ]

    async def lexical_search(session, payload):
        raise AssertionError("Auto with AidRanker should not call lexical retrieval.")

    class FakeRanker:
        name = "aidranker-v1"
        alpha = 0.5
        model_name_or_path = "fake-model"
        artifact_fingerprint = "sha256:test"
        candidate_k = 40
        fail_open = True

        async def rerank(self, query, hits):
            assert query == "food security"
            return list(reversed(hits))

    monkeypatch.setattr(engine, "semantic_search", semantic_search)
    monkeypatch.setattr(engine, "lexical_search", lexical_search)

    response = await engine.execute_search(
        object(),
        EvidenceSearchRequest(
            query="food security",
            mode="auto",
            top_k=2,
            max_per_evaluation=1,
        ),
        query_vector=[0.1, 0.2],
        embedding_model="test-model",
        reranker=FakeRanker(),
    )

    assert semantic_calls == 1
    assert response.mode == "semantic"
    assert response.reranker_applied is True
    assert response.reranker == "aidranker-v1"
    assert response.reranker_alpha == 0.5
    assert response.reranker_model_fingerprint == "sha256:test"
    assert response.reranker_latency_ms is not None
    assert response.ranking_pipeline == [
        "semantic",
        "aidranker-v1",
        "fusion:0.50",
        "diversity:max-1-per-report",
    ]
    assert [hit.evaluation_id for hit in response.hits] == ["B", "A"]


@pytest.mark.asyncio
async def test_auto_aidranker_failure_falls_back_to_semantic(monkeypatch) -> None:
    hits = [_hit("semantic", "A"), _hit("semantic", "B")]

    async def semantic_search(session, payload, query_vector):
        return hits

    class FailingRanker:
        name = "aidranker-v1"
        alpha = 0.5
        model_name_or_path = "missing-model"
        candidate_k = 40
        fail_open = True

        async def rerank(self, query, candidates):
            raise RuntimeError("model missing")

    monkeypatch.setattr(engine, "semantic_search", semantic_search)

    response = await engine.execute_search(
        object(),
        EvidenceSearchRequest(query="food security", mode="auto", top_k=2),
        query_vector=[0.1, 0.2],
        embedding_model="test-model",
        reranker=FailingRanker(),
    )

    assert response.mode == "semantic"
    assert response.reranker_applied is False
    assert response.reranker_fallback_reason == "aidranker_unavailable"
    assert response.ranking_pipeline == ["semantic"]
    assert response.hits == hits


@pytest.mark.asyncio
async def test_explicit_aidranker_failure_is_not_silently_degraded(monkeypatch) -> None:
    async def semantic_search(session, payload, query_vector):
        return [_hit("semantic")]

    class FailingRanker:
        name = "aidranker-v1"
        alpha = 0.5
        model_name_or_path = "missing-model"
        candidate_k = 40
        fail_open = True

        async def rerank(self, query, candidates):
            raise RuntimeError("model missing")

    monkeypatch.setattr(engine, "semantic_search", semantic_search)

    with pytest.raises(ValueError, match="temporarily unavailable"):
        await engine.execute_search(
            object(),
            EvidenceSearchRequest(
                query="food security",
                mode="semantic",
                rerank="aidranker",
            ),
            query_vector=[0.1, 0.2],
            embedding_model="test-model",
            reranker=FailingRanker(),
        )


def test_diversified_search_uses_deeper_candidate_pool() -> None:
    payload = EvidenceSearchRequest(
        query="education access",
        mode="hybrid",
        top_k=10,
        max_per_evaluation=3,
    )

    assert engine._diversity_pool_k(payload) == 90


def test_aidranker_pool_is_bounded_for_serving_latency() -> None:
    payload = EvidenceSearchRequest(
        query="education access",
        mode="semantic",
        top_k=10,
        max_per_evaluation=3,
    )

    assert engine._reranker_pool_k(payload, configured_candidate_k=40) == 40
