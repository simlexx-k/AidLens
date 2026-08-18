from app.services.ingestion.chunker import CHUNKER_VERSION, chunk_document


def test_chunker_detects_sections() -> None:
    text = """EXECUTIVE SUMMARY

This evaluation examines a program and its outcomes.

FINDINGS

The intervention increased attendance among participating students.

RECOMMENDATIONS

Continue targeted support while measuring implementation quality.
"""
    chunks = chunk_document(text, target_chars=120)

    assert CHUNKER_VERSION == "v2"
    assert [chunk.section for chunk in chunks] == [
        "executive_summary",
        "findings",
        "recommendations",
    ]
    assert chunks[1].text.startswith("The intervention")


def test_chunker_handles_pdf_extracted_headings_without_blank_lines() -> None:
    text = """CONTENTS
EXECUTIVE SUMMARY ........................................ 1
EVALUATION METHODOLOGY ................................... 9
KEY FINDINGS ............................................ 18
1 | MID-TERM PERFORMANCE EVALUATION
EXECUTIVE SUMMARY
The evaluation examined relevance, effectiveness, and sustainability.
The team used a mixed-methods design and reviewed monitoring data.
CHAPTER 2: EVALUATION METHODOLOGY
The team conducted interviews, focus groups, and document review.
3. KEY FINDINGS
Teachers reported improved access to instructional materials.
5 RECOMMENDATIONS
Continue local capacity building and strengthen monitoring systems.
"""

    chunks = chunk_document(text, target_chars=180)
    sections = [chunk.section for chunk in chunks]

    assert "executive_summary" in sections
    assert "methodology" in sections
    assert "findings" in sections
    assert "recommendations" in sections
    assert sections.count("executive_summary") == 1


def test_chunker_splits_long_unbroken_pdf_text() -> None:
    text = "EXECUTIVE SUMMARY\n" + "word " * 300

    chunks = chunk_document(text, target_chars=200)

    assert len(chunks) > 1
    assert all(chunk.section == "executive_summary" for chunk in chunks)
    assert max(len(chunk.text) for chunk in chunks) <= 210
