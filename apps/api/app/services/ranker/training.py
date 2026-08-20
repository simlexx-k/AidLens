import json
from collections import Counter
from pathlib import Path

from app.schemas.benchmark import RankerTrainingRecord

DEFAULT_RANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def train_aidranker(
    records: list[RankerTrainingRecord],
    output_dir: Path,
    *,
    model_name: str = DEFAULT_RANKER_MODEL,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
) -> dict[str, object]:
    """Fine-tune a local CrossEncoder on graded AidLens query-passage labels."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    labels = Counter(record.relevance for record in records)
    if labels[0] == 0:
        raise ValueError("AidRanker training requires relevance-0 hard negatives.")
    if sum(count for relevance, count in labels.items() if relevance > 0) == 0:
        raise ValueError("AidRanker training requires at least one positive record.")

    try:
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "AidRanker training requires the API ML extras. "
            "Install with `pip install -e '.[ml]'` or use the ML Docker image."
        ) from exc

    examples = [
        InputExample(
            texts=[record.query, record.text],
            label=float(record.relevance) / 3.0,
        )
        for record in records
    ]
    train_loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = max(1, round(total_steps * 0.1))

    model = CrossEncoder(model_name, num_labels=1)
    model.fit(
        train_dataloader=train_loader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        show_progress_bar=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))

    metadata: dict[str, object] = {
        "model_name": model_name,
        "records": len(records),
        "queries": len({record.query_id for record in records}),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "label_distribution": {
            str(relevance): labels[relevance] for relevance in sorted(labels)
        },
        "label_scaling": "relevance / 3.0",
    }
    (output_dir / "aidranker-training.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata
