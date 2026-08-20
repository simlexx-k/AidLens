from pathlib import Path
from typing import Annotated

import typer

from app.claim_eval_cli import STANCES, _read_jsonl, _write_jsonl

CHOICES = {str(index): stance for index, stance in enumerate(STANCES, start=1)}


def review_command(
    review_file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False),
    ],
    relabel: Annotated[
        bool,
        typer.Option(help="Include rows that already have a gold label."),
    ] = False,
    collect_notes: Annotated[
        bool,
        typer.Option("--notes", help="Prompt for reviewer notes after each label."),
    ] = False,
) -> None:
    """Review claim records interactively and save each decision immediately."""

    rows = _read_jsonl(review_file)
    pending = [
        index
        for index, row in enumerate(rows)
        if relabel or row.get("gold_stance") in (None, "")
    ]
    if not pending:
        typer.echo("No unlabeled records remain.")
        return

    typer.echo("Labels: 1 supports · 2 mixed · 3 contradicts · 4 insufficient · 5 not-effect")
    typer.echo("Other commands: s skip · q quit\n")

    reviewed_now = 0
    for sequence, row_index in enumerate(pending, start=1):
        row = rows[row_index]
        typer.echo("=" * 88)
        typer.echo(
            f"Record {sequence}/{len(pending)} · {row.get('record_id')} · "
            f"evaluation {row.get('evaluation_id')}"
        )
        typer.echo(
            f"families: {', '.join(str(item) for item in row.get('query_families', []))}"
        )
        typer.echo(
            f"section: {row.get('section')} · role: {row.get('evidence_role')} · "
            f"baseline: {row.get('baseline_stance')}"
        )
        basis = row.get("baseline_stance_basis") or []
        if basis:
            typer.echo(f"baseline basis: {' · '.join(str(item) for item in basis)}")
        typer.echo("\n" + str(row.get("statement", "")).strip() + "\n")
        typer.echo(f"source: {row.get('source_url')}")

        while True:
            choice = typer.prompt("Label [1-5/s/q]").strip().casefold()
            if choice == "q":
                typer.echo(
                    f"Stopped. Saved {reviewed_now} labels; "
                    f"{_unlabeled_count(rows)} remain."
                )
                return
            if choice == "s":
                break
            stance = CHOICES.get(choice)
            if stance is None:
                typer.echo("Choose 1, 2, 3, 4, 5, s, or q.")
                continue

            row["gold_stance"] = stance
            if collect_notes:
                row["reviewer_notes"] = typer.prompt(
                    "Notes",
                    default=str(row.get("reviewer_notes") or ""),
                    show_default=False,
                )
            _write_jsonl(review_file, rows)
            reviewed_now += 1
            typer.echo(f"saved: {stance}\n")
            break

    typer.echo(
        f"Review pass complete. Saved {reviewed_now} labels; "
        f"{_unlabeled_count(rows)} remain."
    )


def _unlabeled_count(rows: list[dict[str, object]]) -> int:
    return sum(row.get("gold_stance") in (None, "") for row in rows)


if __name__ == "__main__":
    typer.run(review_command)
