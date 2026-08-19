from pathlib import Path

from app.services.evaluation.candidates import load_candidate_queries


def test_v1_seed_has_balanced_query_families() -> None:
    path = Path("apps/api/benchmarks/queries.v1.jsonl")
    queries = load_candidate_queries(path)

    assert len(queries) == 30
    families: dict[str, int] = {}
    for query in queries:
        families[query.family] = families.get(query.family, 0) + 1

    assert families == {
        "evaluation_methods": 5,
        "implementation_factors": 5,
        "intervention_outcomes": 5,
        "recommendations": 5,
        "sustainability_risks": 5,
        "transferability_context": 5,
    }
