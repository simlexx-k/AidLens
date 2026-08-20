from app.claim_review_cli import CHOICES, _unlabeled_count


def test_review_choices_match_stance_contract() -> None:
    assert CHOICES == {
        "1": "supports",
        "2": "mixed",
        "3": "contradicts",
        "4": "insufficient",
        "5": "not_an_effect_claim",
    }


def test_unlabeled_count_treats_null_and_empty_as_pending() -> None:
    rows = [
        {"gold_stance": None},
        {"gold_stance": ""},
        {"gold_stance": "supports"},
    ]

    assert _unlabeled_count(rows) == 2
