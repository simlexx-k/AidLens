import re
from collections import defaultdict

from app.schemas.evaluation import (
    ClaimStance,
    ClaimStanceCoverage,
    EvidenceEvaluationGroup,
    EvidenceSearchHit,
    GroundedEvidenceClaim,
)

CLAIM_EXTRACTOR = "explicit-text-v1"

_STANCE_ORDER: tuple[ClaimStance, ...] = (
    "supports",
    "mixed",
    "contradicts",
    "insufficient",
    "not_an_effect_claim",
)

_MIXED_PATTERNS = (
    re.compile(r"\bmixed (?:results?|evidence|effects?|outcomes?)\b", re.IGNORECASE),
    re.compile(r"\binconsistent (?:results?|effects?|outcomes?|evidence)\b", re.IGNORECASE),
    re.compile(r"\bvaried (?:substantially )?across\b", re.IGNORECASE),
    re.compile(r"\bheterogeneous (?:effects?|results?|outcomes?)\b", re.IGNORECASE),
)
_NEGATIVE_PATTERNS = (
    re.compile(
        r"\bno significant (?:effect|impact|improvement|change|difference)s?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bno evidence of (?:an? )?(?:effect|impact|improvement|benefit)\b", re.IGNORECASE),
    re.compile(
        r"\blittle evidence of (?:an? )?(?:effect|impact|improvement|benefit)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdid not (?:improve|increase|reduce|strengthen|enhance|lead|result|contribute)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bfailed to (?:improve|increase|reduce|achieve|sustain|deliver)\b", re.IGNORECASE),
    re.compile(r"\b(?:was|were|is|are) not effective\b", re.IGNORECASE),
    re.compile(r"\bineffective\b", re.IGNORECASE),
    re.compile(r"\bnegative (?:effect|impact|outcome)s?\b", re.IGNORECASE),
    re.compile(r"\bnot sustained\b", re.IGNORECASE),
    re.compile(r"\bworsened\b", re.IGNORECASE),
)
_POSITIVE_PATTERNS = (
    re.compile(r"\bimproved?\b", re.IGNORECASE),
    re.compile(r"\b(?:was|were|is|are) effective\b", re.IGNORECASE),
    re.compile(r"\bpositive (?:effect|impact|outcome)s?\b", re.IGNORECASE),
    re.compile(r"\bsuccessfully\b", re.IGNORECASE),
    re.compile(r"\bachieved (?:its |the |their )?(?:target|objective|goal)s?\b", re.IGNORECASE),
    re.compile(r"\b(?:met|exceeded) (?:its |the |their )?targets?\b", re.IGNORECASE),
    re.compile(r"\bcontributed to\b", re.IGNORECASE),
    re.compile(r"\bled to\b", re.IGNORECASE),
    re.compile(r"\bresulted in\b", re.IGNORECASE),
    re.compile(r"\bbenefited?\b", re.IGNORECASE),
    re.compile(r"\bstrengthened?\b", re.IGNORECASE),
    re.compile(r"\benhanced?\b", re.IGNORECASE),
    re.compile(
        (
            r"\bincreased? (?:access|coverage|uptake|attendance|completion|learning|"
            r"income|productivity)\b"
        ),
        re.IGNORECASE,
    ),
    re.compile(
        r"\breduced? (?:mortality|costs?|delays?|dropout|errors?|violence|poverty)\b",
        re.IGNORECASE,
    ),
)
_EFFECT_PATTERNS = (
    re.compile(r"\b(?:effect|impact|outcome|effectiveness|benefit)s?\b", re.IGNORECASE),
    re.compile(r"\b(?:improvement|reduction|increase|decrease|change)s?\b", re.IGNORECASE),
    re.compile(r"\b(?:achiev|improv|enhanc|strengthen|sustain)\w*\b", re.IGNORECASE),
)
_CONTRAST_PATTERN = re.compile(r"\b(?:but|however|although|yet|while)\b", re.IGNORECASE)
_PROSPECTIVE_EFFECT_PATTERN = re.compile(
    (
        r"\b(?:could|may|might|should|would|can|likely to|potentially)\b"
        r"[^,.;:!?]{0,80}\b(?:improve|increase|reduce|strengthen|enhance|benefit|"
        r"lead|result|contribute|achieve|sustain)\w*\b"
    ),
    re.IGNORECASE,
)
_INDICATOR_LABEL_PATTERN = re.compile(
    r"\b(?:OUTCOME|OUTPUT|PERFORMANCE)\s+INDICATORS?\b"
)
_CONDITION_PATTERN = re.compile(
    r"\b(?:only when|only where|provided that|when|where|if|among)\b[^,.;:!?]{3,140}",
    re.IGNORECASE,
)
_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def extract_grounded_claims(
    groups: list[EvidenceEvaluationGroup],
    *,
    max_claims_per_passage: int = 2,
) -> list[GroundedEvidenceClaim]:
    """Extract passage-grounded, explicit-text claim assessments.

    The extractor never rewrites the source sentence. Stance is assigned only from
    explicit lexical signals inside that sentence; section labels and metadata do
    not determine stance. Questions, prospective/modal statements and indicator
    labels abstain rather than being promoted to observed effect claims. When no
    effect signal is present, one source sentence is returned as
    `not_an_effect_claim` so absence is visible rather than silently omitted.
    """

    claims: list[GroundedEvidenceClaim] = []
    for group in groups:
        for hit in group.hits:
            assessments = _assess_hit(hit, max_claims=max_claims_per_passage)
            claims.extend(assessments)
    return claims


def stance_coverage(claims: list[GroundedEvidenceClaim]) -> list[ClaimStanceCoverage]:
    evaluations: dict[ClaimStance, set[str]] = defaultdict(set)
    counts: dict[ClaimStance, int] = defaultdict(int)
    for claim in claims:
        evaluations[claim.stance].add(claim.evaluation_id)
        counts[claim.stance] += 1

    return [
        ClaimStanceCoverage(
            stance=stance,
            evaluation_count=len(evaluations[stance]),
            claim_count=counts[stance],
        )
        for stance in _STANCE_ORDER
        if counts[stance] > 0
    ]


def effect_claim_evaluation_count(claims: list[GroundedEvidenceClaim]) -> int:
    return len(
        {
            claim.evaluation_id
            for claim in claims
            if claim.stance != "not_an_effect_claim"
        }
    )


def _assess_hit(
    hit: EvidenceSearchHit,
    *,
    max_claims: int,
) -> list[GroundedEvidenceClaim]:
    sentences = _sentence_spans(hit.text)
    if not sentences:
        return []

    assessed = [(*sentence, *_classify_sentence(sentence[0])) for sentence in sentences]
    effect_claims = [item for item in assessed if item[3] != "not_an_effect_claim"]
    selected = effect_claims[:max_claims]
    if not selected:
        selected = [assessed[0]]

    claims: list[GroundedEvidenceClaim] = []
    for index, (statement, start, end, stance, confidence, basis) in enumerate(selected):
        claims.append(
            GroundedEvidenceClaim(
                claim_id=f"{hit.chunk_id}:{index}",
                evaluation_id=hit.evaluation_id,
                chunk_id=hit.chunk_id,
                section=hit.section,
                evidence_role=hit.evidence_role,
                statement=statement,
                source_span_start=start,
                source_span_end=end,
                stance=stance,
                confidence=confidence,
                stance_basis=basis,
                explicit_conditions=_extract_conditions(statement),
                source_url=hit.source_url,
            )
        )
    return claims


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for match in _SENTENCE_PATTERN.finditer(text):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = match.start() + leading
        end = match.end() - trailing
        if end <= start:
            continue
        statement = text[start:end]
        if len(statement) < 12:
            continue
        spans.append((statement, start, end))
    return spans


def _classify_sentence(statement: str) -> tuple[ClaimStance, float, list[str]]:
    if _is_non_assertive(statement):
        return "not_an_effect_claim", 0.98, []

    mixed = _matches(statement, _MIXED_PATTERNS)
    negative = _matches(statement, _NEGATIVE_PATTERNS)
    positive = _matches(statement, _POSITIVE_PATTERNS)
    effect = _matches(statement, _EFFECT_PATTERNS)
    has_contrast = bool(_CONTRAST_PATTERN.search(statement))

    if mixed or (positive and negative and has_contrast):
        return "mixed", 0.88, _unique([*mixed, *positive, *negative])[:4]
    if negative:
        return "contradicts", 0.86, negative[:4]
    if positive:
        return "supports", 0.84, positive[:4]
    if effect:
        return "insufficient", 0.48, effect[:4]
    return "not_an_effect_claim", 0.98, []


def _is_non_assertive(statement: str) -> bool:
    stripped = statement.rstrip()
    if stripped.endswith("?"):
        return True
    if _PROSPECTIVE_EFFECT_PATTERN.search(statement):
        return True
    return bool(_INDICATOR_LABEL_PATTERN.search(statement))


def _matches(statement: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        match = pattern.search(statement)
        if match:
            matches.append(match.group(0))
    return _unique(matches)


def _extract_conditions(statement: str) -> list[str]:
    return _unique(
        match.group(0).strip() for match in _CONDITION_PATTERN.finditer(statement)
    )[:3]


def _unique(values) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = " ".join(value.casefold().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(value)
    return output
