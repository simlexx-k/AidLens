from app.perf_cli import _final_top_ids, _parse_batch_sizes


def _candidate(chunk_id: str, evaluation_id: str, semantic_score: float):
    return {
        "chunk_id": chunk_id,
        "evaluation_id": evaluation_id,
        "text": "Evidence passage.",
        "score": semantic_score,
        "semantic_score": semantic_score,
    }


def test_parse_batch_sizes_deduplicates_in_order() -> None:
    assert _parse_batch_sizes("8,16,32,40,32") == [8, 16, 32, 40]


def test_final_top_ids_applies_frozen_fusion_then_diversity() -> None:
    candidates = [
        _candidate("a1", "A", 0.90),
        _candidate("a2", "A", 0.80),
        _candidate("b1", "B", 0.70),
        _candidate("c1", "C", 0.60),
    ]
    reranker_scores = [0.0, 3.0, 2.0, 1.0]

    selected = _final_top_ids(
        candidates,
        reranker_scores,
        top_k=3,
        max_per_evaluation=1,
    )

    assert selected == ["a2", "b1", "c1"]
