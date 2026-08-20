import json
from pathlib import Path

import pytest
import typer

from app.claim_eval_cli import (
    _classification_report,
    _reviewed_rows,
    _stratified_queries,
)


def test_stratified_queries_selects_each_family_in_source_order() -> None:
    queries = [
        {"query_id": "a1", "family": "A", "query": "one"},
        {"query_id": "a2", "family": "A", "query": "two"},
        {"query_id": "a3", "family": "A", "query": "three"},
        {"query_id": "b1", "family": "B", "query": "four"},
        {"query_id": "b2", "family": "B", "query": "five"},
    ]

    selected = _stratified_queries(queries, per_family=2)

    assert [row["query_id"] for row in selected] == ["a1", "a2", "b1", "b2"]


def test_classification_report_is_perfect_for_exact_predictions() -> None:
    gold = [
        "supports",
        "mixed",
        "contradicts",
        "insufficient",
        "not_an_effect_claim",
    ]

    report = _classification_report(gold, gold)

    assert report["accuracy"] == 1.0
    assert report["macro_f1_active_gold_classes"] == 1.0
    assert report["effect_claim_detection"]["f1"] == 1.0
    assert report["directional_exact_precision"] == 1.0
    assert report["directional_exact_recall"] == 1.0
    assert report["directional_abstention_rate"] == 0.4


def test_classification_report_tracks_false_effect_claims() -> None:
    report = _classification_report(
        ["not_an_effect_claim", "supports"],
        ["supports", "insufficient"],
    )

    assert report["accuracy"] == 0.0
    assert report["effect_claim_detection"]["precision"] == 0.5
    assert report["effect_claim_detection"]["recall"] == 1.0
    assert report["directional_exact_precision"] == 0.0


def test_reviewed_rows_requires_complete_review_by_default(tmp_path: Path) -> None:
    path = tmp_path / "review.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_id": "one",
                        "statement": "The intervention improved attendance.",
                        "gold_stance": "supports",
                    }
                ),
                json.dumps(
                    {
                        "record_id": "two",
                        "statement": "What changed?",
                        "gold_stance": None,
                    }
                ),
            ]
        )
        + "\n"
    )

    with pytest.raises(typer.BadParameter, match="still unlabeled"):
        _reviewed_rows(path, allow_partial=False)

    reviewed, skipped = _reviewed_rows(path, allow_partial=True)
    assert len(reviewed) == 1
    assert skipped == 1


def test_reviewed_rows_rejects_unknown_stance(tmp_path: Path) -> None:
    path = tmp_path / "review.jsonl"
    path.write_text(
        json.dumps(
            {
                "record_id": "one",
                "statement": "Evidence text.",
                "gold_stance": "positive",
            }
        )
        + "\n"
    )

    with pytest.raises(typer.BadParameter, match="Invalid gold_stance"):
        _reviewed_rows(path, allow_partial=True)
