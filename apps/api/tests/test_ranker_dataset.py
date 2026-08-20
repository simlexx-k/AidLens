import json
import uuid

import pytest

from app.schemas.benchmark import RankerTrainingRecord
from app.services.ranker.dataset import split_ranker_records, write_ranker_split


def _record(query_id: str, family: str, relevance: int = 1) -> RankerTrainingRecord:
    return RankerTrainingRecord(
        query_id=query_id,
        family=family,
        query=f"Question {query_id}?",
        chunk_id=uuid.uuid4(),
        evaluation_id=f"E-{query_id}",
        title="Evaluation",
        section="findings",
        text="Evidence passage.",
        relevance=relevance,
        retrieval_rank=1,
        score=0.5,
        semantic_score=0.5,
        retrieval_modes=["semantic"],
        mode_ranks={"semantic": 1},
    )


def test_ranker_split_is_query_grouped_and_family_balanced(tmp_path) -> None:
    records = []
    for family in ("outcomes", "methods"):
        for index in range(1, 6):
            query_id = f"{family}-{index}"
            records.extend(
                [
                    _record(query_id, family, relevance=0),
                    _record(query_id, family, relevance=3),
                ]
            )

    split = split_ranker_records(records, seed=17)

    assert len(split.query_ids["train"]) == 6
    assert len(split.query_ids["dev"]) == 2
    assert len(split.query_ids["test"]) == 2
    assert len(split.train) == 12
    assert len(split.dev) == 4
    assert len(split.test) == 4

    train = set(split.query_ids["train"])
    dev = set(split.query_ids["dev"])
    test = set(split.query_ids["test"])
    assert train.isdisjoint(dev)
    assert train.isdisjoint(test)
    assert dev.isdisjoint(test)

    for family in ("outcomes", "methods"):
        family_splits = split.family_query_ids[family]
        assert len(family_splits["train"]) == 3
        assert len(family_splits["dev"]) == 1
        assert len(family_splits["test"]) == 1

    output_dir = tmp_path / "ranker-split"
    write_ranker_split(split, output_dir, seed=17)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["query_counts"] == {"train": 6, "dev": 2, "test": 2}
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "dev.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()


def test_ranker_split_rejects_incomplete_family() -> None:
    records = [_record(f"outcomes-{index}", "outcomes") for index in range(1, 5)]

    with pytest.raises(ValueError, match="expected 5"):
        split_ranker_records(records)
