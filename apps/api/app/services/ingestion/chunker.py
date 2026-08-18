import re
from dataclasses import dataclass

CHUNKER_VERSION = "v2"

SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"^abstract$", re.I)),
    ("executive_summary", re.compile(r"^(?:executive|management) summary$", re.I)),
    ("introduction", re.compile(r"^(?:introduction|background)$", re.I)),
    (
        "methodology",
        re.compile(
            r"^(?:(?:evaluation|research) )?"
            r"(?:methodology|methods?|approach|design)$",
            re.I,
        ),
    ),
    (
        "methodology",
        re.compile(
            r"^evaluation (?:methods? and limitations|design and methodology)$",
            re.I,
        ),
    ),
    (
        "evaluation_questions",
        re.compile(
            r"^(?:evaluation questions?|evaluation purpose(?: and evaluation questions?)?|"
            r"purpose of the evaluation)$",
            re.I,
        ),
    ),
    (
        "findings",
        re.compile(
            r"^(?:(?:key|evaluation|major) )?"
            r"(?:findings|results)(?: and discussion)?$",
            re.I,
        ),
    ),
    (
        "findings",
        re.compile(r"^findings,? conclusions?,? and recommendations?$", re.I),
    ),
    (
        "conclusions",
        re.compile(r"^(?:key )?conclusions?(?: on .+)?$", re.I),
    ),
    (
        "recommendations",
        re.compile(r"^(?:key )?recommendations?(?: on .+)?$", re.I),
    ),
    (
        "limitations",
        re.compile(
            r"^(?:(?:evaluation|study) )?limitations?|"
            r"known limitations to the evaluation design$",
            re.I,
        ),
    ),
    ("sustainability", re.compile(r"^sustainability$", re.I)),
    ("lessons_learned", re.compile(r"^(?:good practices and )?lessons? learned$", re.I)),
]

TOC_LEADER_RE = re.compile(r"(?:\.{3,}|…{2,})\s*\d*\s*$")
PAGE_PREFIX_RE = re.compile(r"^\s*\d+\s*\|\s*")
SECTION_PREFIX_RE = re.compile(
    r"^\s*(?:(?:chapter|section|part)\s+[ivxlcdm\d]+\s*[:.\-]?\s*)?"
    r"(?:\d+(?:\.\d+)*[.)]?\s*)?",
    re.I,
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(slots=True)
class TextChunk:
    ordinal: int
    section: str | None
    text: str


def chunk_document(text: str, target_chars: int = 1800) -> list[TextChunk]:
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

    for kind, value in _document_units(text):
        if kind == "section":
            flush()
            section = value
            continue

        for piece in _split_long_text(value, target_chars):
            projected = size + len(piece) + (2 if buffer else 0)
            if buffer and projected > target_chars:
                carry = buffer[-1] if len(buffer[-1]) <= target_chars // 3 else None
                flush()
                if carry:
                    buffer.append(carry)
                    size = len(carry)
            buffer.append(piece)
            size += len(piece) + (2 if len(buffer) > 1 else 0)

    flush()
    return chunks


def _document_units(text: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            units.append(("text", " ".join(paragraph)))
            paragraph = []

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            flush_paragraph()
            continue

        detected = _section_name(line)
        if detected:
            flush_paragraph()
            units.append(("section", detected))
            continue

        paragraph.append(line)

    flush_paragraph()
    if not units and text.strip():
        units.append(("text", re.sub(r"\s+", " ", text).strip()))
    return units


def _split_long_text(text: str, target_chars: int) -> list[str]:
    if len(text) <= target_chars:
        return [text]

    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    if len(sentences) == 1:
        return _split_by_words(text, target_chars)

    pieces: list[str] = []
    current: list[str] = []
    current_size = 0
    for sentence in sentences:
        if len(sentence) > target_chars:
            if current:
                pieces.append(" ".join(current))
                current = []
                current_size = 0
            pieces.extend(_split_by_words(sentence, target_chars))
            continue
        projected = current_size + len(sentence) + (1 if current else 0)
        if current and projected > target_chars:
            pieces.append(" ".join(current))
            current = []
            current_size = 0
        current.append(sentence)
        current_size += len(sentence) + (1 if len(current) > 1 else 0)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_by_words(text: str, target_chars: int) -> list[str]:
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        projected = size + len(word) + (1 if current else 0)
        if current and projected > target_chars:
            pieces.append(" ".join(current))
            current = []
            size = 0
        current.append(word)
        size += len(word) + (1 if len(current) > 1 else 0)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _section_name(line: str) -> str | None:
    if len(line) > 180 or TOC_LEADER_RE.search(line):
        return None

    normalized = PAGE_PREFIX_RE.sub("", line)
    normalized = SECTION_PREFIX_RE.sub("", normalized)
    normalized = normalized.strip(" :.-|–—")
    for name, pattern in SECTION_PATTERNS:
        if pattern.fullmatch(normalized):
            return name
    return None
