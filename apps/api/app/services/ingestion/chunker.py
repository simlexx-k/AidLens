import re
from dataclasses import dataclass

SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("executive_summary", re.compile(r"^\s*executive summary\s*$", re.I)),
    ("methodology", re.compile(r"^\s*(evaluation )?(methodology|methods?)\s*$", re.I)),
    ("findings", re.compile(r"^\s*(key )?findings\s*$", re.I)),
    ("conclusions", re.compile(r"^\s*conclusions?\s*$", re.I)),
    ("recommendations", re.compile(r"^\s*recommendations?\s*$", re.I)),
    ("limitations", re.compile(r"^\s*(evaluation )?limitations?\s*$", re.I)),
]


@dataclass(slots=True)
class TextChunk:
    ordinal: int
    section: str | None
    text: str


def chunk_document(text: str, target_chars: int = 1800) -> list[TextChunk]:
    paragraphs = [
        re.sub(r"\s+", " ", p).strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]
    if not paragraphs:
        paragraphs = [re.sub(r"\s+", " ", text).strip()] if text.strip() else []

    chunks: list[TextChunk] = []
    buffer: list[str] = []
    size = 0
    section: str | None = None

    def flush() -> None:
        nonlocal buffer, size
        if not buffer:
            return
        chunks.append(TextChunk(len(chunks), section, "\n\n".join(buffer)))
        buffer = []
        size = 0

    for paragraph in paragraphs:
        detected = _section_name(paragraph)
        if detected:
            flush()
            section = detected
            continue

        projected = size + len(paragraph) + (2 if buffer else 0)
        if buffer and projected > target_chars:
            carry = buffer[-1] if len(buffer[-1]) <= target_chars // 3 else None
            flush()
            if carry:
                buffer.append(carry)
                size = len(carry)

        buffer.append(paragraph)
        size += len(paragraph) + (2 if len(buffer) > 1 else 0)

    flush()
    return chunks


def _section_name(paragraph: str) -> str | None:
    if len(paragraph) > 120:
        return None
    normalized = re.sub(r"^(?:\d+(?:\.\d+)*[.)]?\s*)", "", paragraph).strip(" :.-")
    for name, pattern in SECTION_PATTERNS:
        if pattern.match(normalized):
            return name
    return None
