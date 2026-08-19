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


def test_diversified_search_uses_deeper_candidate_pool() -> None:
    payload = EvidenceSearchRequest(
        query="education access",
        mode="hybrid",
        top_k=10,
        max_per_evaluation=3,
    )

    assert engine._diversity_pool_k(payload) == 90
