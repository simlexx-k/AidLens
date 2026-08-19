from pathlib import Path

from app.services.evaluation.benchmark import load_benchmark_dataset
from app.services.evaluation.candidates import load_candidate_queries


def test_load_benchmark_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "judgments.jsonl"
    dataset.write_text(
        '{"query_id":"q1","query":"food security outcomes",'
        '"judgments":[{"evaluation_id":"ABC123","section":"findings",'
        '"anchor_text":"instrumental in supporting households",'
        '"relevance":3}]}\n',
        encoding="utf-8",
    )

    items = load_benchmark_dataset(dataset)

    assert len(items) == 1
    assert items[0].query_id == "q1"
    assert items[0].judgments[0].evaluation_id == "ABC123"
    assert items[0].judgments[0].anchor_text == "instrumental in supporting households"


def test_load_candidate_queries(tmp_path: Path) -> None:
    dataset = tmp_path / "queries.jsonl"
    dataset.write_text(
        '# comment\n{"query_id":"q1","query":"program sustainability"}\n',
        encoding="utf-8",
    )

    items = load_candidate_queries(dataset)

    assert len(items) == 1
    assert items[0].query == "program sustainability"
