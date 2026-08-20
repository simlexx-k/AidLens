import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Annotated

import httpx
import typer

from app.services.search.claims import CLAIM_EXTRACTOR, _classify_sentence

STANCES = (
    "supports",
    "mixed",
    "contradicts",
    "insufficient",
    "not_an_effect_claim",
)
DIRECTIONAL_STANCES = {"supports", "mixed", "contradicts"}
NLI_LABELS = {
    "an observed positive program effect": "supports",
    "mixed or heterogeneous observed program effects": "mixed",
    "an observed null, negative, or failed program effect": "contradicts",
    "effect-related discussion without a directional observed result": "insufficient",
    "not an observed program effect claim": "not_an_effect_claim",
}

cli = typer.Typer(
    no_args_is_help=True,
    help="Build and score reviewed claim-quality benchmarks for AidLens stance extraction.",
)


@cli.command("sample")
def sample_command(
    api_url: Annotated[
        str,
        typer.Option(help="Evidence-search endpoint used to collect grounded claims."),
    ] = "http://localhost:8000/api/v1/search/evidence",
    queries_file: Annotated[
        Path,
        typer.Option("--queries", exists=True, dir_okay=False),
    ] = Path("benchmarks/queries.v1.jsonl"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", dir_okay=False),
    ] = Path("benchmarks/claim-quality.review.local.jsonl"),
    per_family: Annotated[
        int,
        typer.Option(min=1, max=5, help="Queries sampled deterministically per query family."),
    ] = 2,
    top_k: Annotated[int, typer.Option(min=1, max=20)] = 5,
    max_per_evaluation: Annotated[int, typer.Option(min=1, max=10)] = 2,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing review file.")] = False,
) -> None:
    """Create an unlabeled, deduplicated review set from the live AidLens API."""

    if output.exists() and not overwrite:
        raise typer.BadParameter(f"{output} already exists; pass --overwrite to replace it.")

    queries = _read_jsonl(queries_file)
    selected = _stratified_queries(queries, per_family=per_family)
    rows_by_id: dict[str, dict[str, object]] = {}

    with httpx.Client(timeout=120.0) as client:
        for index, query in enumerate(selected, start=1):
            typer.echo(
                f"[{index}/{len(selected)}] {query['query_id']} · {query['family']}",
                err=True,
            )
            response = client.post(
                api_url,
                json={
                    "query": query["query"],
                    "mode": "auto",
                    "rerank": "aidranker",
                    "top_k": top_k,
                    "max_per_evaluation": max_per_evaluation,
                },
            )
            response.raise_for_status()
            body = response.json()
            synthesis = body.get("synthesis") or {}
            claims = synthesis.get("claims") or []
            if synthesis.get("claim_extractor") != CLAIM_EXTRACTOR:
                raise typer.BadParameter(
                    "Live API claim extractor does not match the checked-out baseline."
                )

            for claim in claims:
                record_id = (
                    f"{claim['chunk_id']}:{claim['source_span_start']}:"
                    f"{claim['source_span_end']}"
                )
                existing = rows_by_id.get(record_id)
                if existing is not None:
                    _append_unique(existing["query_ids"], query["query_id"])
                    _append_unique(existing["query_families"], query["family"])
                    continue

                rows_by_id[record_id] = {
                    "record_id": record_id,
                    "query_ids": [query["query_id"]],
                    "query_families": [query["family"]],
                    "evaluation_id": claim["evaluation_id"],
                    "chunk_id": claim["chunk_id"],
                    "section": claim.get("section"),
                    "evidence_role": claim["evidence_role"],
                    "statement": claim["statement"],
                    "source_span_start": claim["source_span_start"],
                    "source_span_end": claim["source_span_end"],
                    "source_url": claim["source_url"],
                    "baseline_extractor": synthesis["claim_extractor"],
                    "baseline_stance": claim["stance"],
                    "baseline_confidence": claim["confidence"],
                    "baseline_stance_basis": claim["stance_basis"],
                    "gold_stance": None,
                    "reviewer_notes": "",
                }

    rows = list(rows_by_id.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, rows)
    family_counts = Counter(
        family
        for row in rows
        for family in row["query_families"]
    )
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "query_count": len(selected),
                "record_count": len(rows),
                "query_family_record_mentions": dict(sorted(family_counts.items())),
                "review_status": "unlabeled",
            },
            indent=2,
        )
    )


@cli.command("score-baseline")
def score_baseline_command(
    review_file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False),
    ],
    allow_partial: Annotated[
        bool,
        typer.Option(help="Score reviewed rows and skip unlabeled rows."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", dir_okay=False),
    ] = None,
) -> None:
    """Re-score reviewed statements with the current explicit-text baseline."""

    rows, skipped = _reviewed_rows(review_file, allow_partial=allow_partial)
    gold = [str(row["gold_stance"]) for row in rows]
    predictions: list[str] = []
    details: list[dict[str, object]] = []

    for row in rows:
        stance, confidence, basis = _classify_sentence(str(row["statement"]))
        predictions.append(stance)
        if stance != row["gold_stance"]:
            details.append(
                {
                    "record_id": row["record_id"],
                    "gold": row["gold_stance"],
                    "predicted": stance,
                    "confidence": confidence,
                    "basis": basis,
                    "statement": row["statement"],
                }
            )

    report = {
        "system": CLAIM_EXTRACTOR,
        "review_file": str(review_file),
        "reviewed_records": len(rows),
        "unreviewed_skipped": skipped,
        "metrics": _classification_report(gold, predictions),
        "disagreements": details[:50],
    }
    _emit_report(report, output)


@cli.command("compare-nli")
def compare_nli_command(
    review_file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False),
    ],
    model_name: Annotated[
        str,
        typer.Option(help="Local/Hugging Face NLI model used only for this benchmark."),
    ] = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    batch_size: Annotated[int, typer.Option(min=1, max=64)] = 8,
    device: Annotated[
        int,
        typer.Option(help="Transformers pipeline device: -1 CPU, 0 first CUDA device."),
    ] = -1,
    allow_partial: Annotated[
        bool,
        typer.Option(help="Score reviewed rows and skip unlabeled rows."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", dir_okay=False),
    ] = None,
) -> None:
    """Compare a research-only zero-shot NLI stance classifier on reviewed claims."""

    rows, skipped = _reviewed_rows(review_file, allow_partial=allow_partial)
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - optional local experiment
        raise typer.BadParameter(
            "compare-nli requires the optional NLI/ML dependencies."
        ) from exc

    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )
    statements = [str(row["statement"]) for row in rows]
    started = perf_counter()
    outputs = classifier(
        statements,
        candidate_labels=list(NLI_LABELS),
        hypothesis_template="This text is {}.",
        multi_label=False,
        batch_size=batch_size,
    )
    elapsed_ms = round((perf_counter() - started) * 1000.0, 3)
    if isinstance(outputs, dict):
        outputs = [outputs]

    predictions: list[str] = []
    details: list[dict[str, object]] = []
    for row, result in zip(rows, outputs, strict=True):
        top_label = str(result["labels"][0])
        stance = NLI_LABELS[top_label]
        score = float(result["scores"][0])
        predictions.append(stance)
        if stance != row["gold_stance"]:
            details.append(
                {
                    "record_id": row["record_id"],
                    "gold": row["gold_stance"],
                    "predicted": stance,
                    "zero_shot_score": round(score, 6),
                    "statement": row["statement"],
                }
            )

    gold = [str(row["gold_stance"]) for row in rows]
    report = {
        "system": "zero-shot-nli-research",
        "model": model_name,
        "review_file": str(review_file),
        "reviewed_records": len(rows),
        "unreviewed_skipped": skipped,
        "batch_size": batch_size,
        "device": device,
        "inference_ms": elapsed_ms,
        "metrics": _classification_report(gold, predictions),
        "disagreements": details[:50],
    }
    _emit_report(report, output)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid JSON on {path}:{line_number}.") from exc
        if not isinstance(row, dict):
            raise typer.BadParameter(f"Expected JSON object on {path}:{line_number}.")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stratified_queries(
    queries: list[dict[str, object]],
    *,
    per_family: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    family_order: list[str] = []
    for query in queries:
        family = str(query.get("family", ""))
        if not family or not query.get("query_id") or not query.get("query"):
            raise typer.BadParameter("Every query row requires query_id, family and query.")
        if family not in grouped:
            family_order.append(family)
        grouped[family].append(query)

    selected: list[dict[str, object]] = []
    for family in family_order:
        selected.extend(grouped[family][:per_family])
    return selected


def _reviewed_rows(
    review_file: Path,
    *,
    allow_partial: bool,
) -> tuple[list[dict[str, object]], int]:
    rows = _read_jsonl(review_file)
    reviewed: list[dict[str, object]] = []
    skipped = 0
    for row in rows:
        gold = row.get("gold_stance")
        if gold in (None, ""):
            skipped += 1
            continue
        if gold not in STANCES:
            raise typer.BadParameter(
                f"Invalid gold_stance={gold!r} for record {row.get('record_id')}."
            )
        reviewed.append(row)

    if skipped and not allow_partial:
        raise typer.BadParameter(
            f"{skipped} rows are still unlabeled; complete review or pass --allow-partial."
        )
    if not reviewed:
        raise typer.BadParameter("No reviewed rows are available to score.")
    return reviewed, skipped


def _classification_report(
    gold: list[str],
    predicted: list[str],
) -> dict[str, object]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold and predicted must have the same non-zero length")

    confusion = {
        label: {other: 0 for other in STANCES}
        for label in STANCES
    }
    for expected, actual in zip(gold, predicted, strict=True):
        confusion[expected][actual] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    active_f1: list[float] = []
    for label in STANCES:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[other][label] for other in STANCES)
        precision = _safe_div(tp, predicted_count)
        recall = _safe_div(tp, support)
        f1 = _f1(precision, recall)
        if support:
            active_f1.append(f1)
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
            "predicted": predicted_count,
        }

    correct = sum(expected == actual for expected, actual in zip(gold, predicted, strict=True))
    effect_gold = [label != "not_an_effect_claim" for label in gold]
    effect_pred = [label != "not_an_effect_claim" for label in predicted]
    effect_tp = sum(a and b for a, b in zip(effect_gold, effect_pred, strict=True))
    effect_fp = sum((not a) and b for a, b in zip(effect_gold, effect_pred, strict=True))
    effect_fn = sum(a and (not b) for a, b in zip(effect_gold, effect_pred, strict=True))
    effect_precision = _safe_div(effect_tp, effect_tp + effect_fp)
    effect_recall = _safe_div(effect_tp, effect_tp + effect_fn)

    predicted_directional = sum(label in DIRECTIONAL_STANCES for label in predicted)
    gold_directional = sum(label in DIRECTIONAL_STANCES for label in gold)
    exact_directional = sum(
        expected == actual and actual in DIRECTIONAL_STANCES
        for expected, actual in zip(gold, predicted, strict=True)
    )
    directional_abstentions = sum(
        label in {"insufficient", "not_an_effect_claim"}
        for label in predicted
    )

    return {
        "accuracy": round(correct / len(gold), 6),
        "macro_f1_active_gold_classes": round(sum(active_f1) / len(active_f1), 6),
        "effect_claim_detection": {
            "precision": round(effect_precision, 6),
            "recall": round(effect_recall, 6),
            "f1": round(_f1(effect_precision, effect_recall), 6),
        },
        "directional_exact_precision": round(
            _safe_div(exact_directional, predicted_directional),
            6,
        ),
        "directional_exact_recall": round(
            _safe_div(exact_directional, gold_directional),
            6,
        ),
        "directional_abstention_rate": round(directional_abstentions / len(gold), 6),
        "gold_distribution": dict(Counter(gold)),
        "predicted_distribution": dict(Counter(predicted)),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _append_unique(values: object, value: object) -> None:
    if not isinstance(values, list):
        raise TypeError("Expected mutable list")
    if value not in values:
        values.append(value)


def _emit_report(report: dict[str, object], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(f"report={output}")
    typer.echo(rendered)


if __name__ == "__main__":
    cli()
