import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.schemas.benchmark import RankerTrainingRecord


@dataclass(frozen=True)
class RankerSplit:
    train: list[RankerTrainingRecord]
    dev: list[RankerTrainingRecord]
    test: list[RankerTrainingRecord]
    query_ids: dict[str, list[str]]
    family_query_ids: dict[str, dict[str, list[str]]]


def load_ranker_records(path: Path) -> list[RankerTrainingRecord]:
    records: list[RankerTrainingRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(RankerTrainingRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid ranker JSONL at {path}:{line_number}: {exc}"
                ) from exc
    if not records:
        raise ValueError(f"Ranker dataset {path} contains no records.")
    return records


def split_ranker_records(
    records: list[RankerTrainingRecord],
    *,
    seed: int = 42,
    train_per_family: int = 3,
    dev_per_family: int = 1,
    test_per_family: int = 1,
) -> RankerSplit:
    """Split by whole query while preserving evidence-family balance."""

    expected_per_family = train_per_family + dev_per_family + test_per_family
    if min(train_per_family, dev_per_family, test_per_family) < 1:
        raise ValueError("Each split must receive at least one query per family.")

    family_queries: dict[str, set[str]] = defaultdict(set)
    query_family: dict[str, str] = {}
    for record in records:
        existing = query_family.get(record.query_id)
        if existing is not None and existing != record.family:
            raise ValueError(
                f"Query {record.query_id} appears in multiple families: "
                f"{existing!r} and {record.family!r}."
            )
        query_family[record.query_id] = record.family
        family_queries[record.family].add(record.query_id)

    split_query_ids = {"train": [], "dev": [], "test": []}
    family_query_ids: dict[str, dict[str, list[str]]] = {}
    rng = random.Random(seed)

    for family in sorted(family_queries):
        query_ids = sorted(family_queries[family])
        if len(query_ids) != expected_per_family:
            raise ValueError(
                f"Family {family!r} has {len(query_ids)} queries; expected "
                f"{expected_per_family} for a balanced "
                f"{train_per_family}/{dev_per_family}/{test_per_family} split."
            )
        rng.shuffle(query_ids)
        train_end = train_per_family
        dev_end = train_end + dev_per_family
        family_splits = {
            "train": sorted(query_ids[:train_end]),
            "dev": sorted(query_ids[train_end:dev_end]),
            "test": sorted(query_ids[dev_end:]),
        }
        family_query_ids[family] = family_splits
        for split_name, items in family_splits.items():
            split_query_ids[split_name].extend(items)

    split_sets = {
        split_name: set(query_ids)
        for split_name, query_ids in split_query_ids.items()
    }
    if split_sets["train"] & split_sets["dev"]:
        raise AssertionError("Train and dev query sets overlap.")
    if split_sets["train"] & split_sets["test"]:
        raise AssertionError("Train and test query sets overlap.")
    if split_sets["dev"] & split_sets["test"]:
        raise AssertionError("Dev and test query sets overlap.")

    grouped_records = {
        split_name: [
            record for record in records if record.query_id in split_sets[split_name]
        ]
        for split_name in ("train", "dev", "test")
    }
    normalized_query_ids = {
        split_name: sorted(query_ids)
        for split_name, query_ids in split_query_ids.items()
    }
    return RankerSplit(
        train=grouped_records["train"],
        dev=grouped_records["dev"],
        test=grouped_records["test"],
        query_ids=normalized_query_ids,
        family_query_ids=family_query_ids,
    )


def write_ranker_split(split: RankerSplit, output_dir: Path, *, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "dev", "test"):
        records = getattr(split, split_name)
        _write_records(records, output_dir / f"{split_name}.jsonl")

    manifest = {
        "seed": seed,
        "query_counts": {
            split_name: len(query_ids)
            for split_name, query_ids in split.query_ids.items()
        },
        "record_counts": {
            split_name: len(getattr(split, split_name))
            for split_name in ("train", "dev", "test")
        },
        "query_ids": split.query_ids,
        "families": split.family_query_ids,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_records(records: list[RankerTrainingRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
            handle.write(payload + "\n")
