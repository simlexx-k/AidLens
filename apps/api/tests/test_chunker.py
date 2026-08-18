from app.services.ingestion.chunker import chunk_document


def test_chunker_detects_sections() -> None:
    text = """EXECUTIVE SUMMARY

This evaluation examines a program and its outcomes.

FINDINGS

The intervention increased attendance among participating students.

RECOMMENDATIONS

Continue targeted support while measuring implementation quality.
"""
    chunks = chunk_document(text, target_chars=120)

    assert [chunk.section for chunk in chunks] == [
        "executive_summary",
        "findings",
        "recommendations",
    ]
    assert chunks[1].text.startswith("The intervention")
