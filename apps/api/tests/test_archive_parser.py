from app.services.archive.aiddata import AidDataArchiveClient


def test_parse_size_kb() -> None:
    assert AidDataArchiveClient._parse_size_kb("1124 KB") == 1124
    assert AidDataArchiveClient._parse_size_kb("1.9 MB") == 1946
    assert AidDataArchiveClient._parse_size_kb(None) is None


def test_normalize_institutions_splits_embedded_usaid_offices() -> None:
    value = (
        "7337 - Catholic Relief Services (CRS) "
        "42092 USAID. Bur. for Humanitarian Assistance (BHA) | "
        "1374 - Johns Hopkins University"
    )

    assert AidDataArchiveClient._normalize_institutions(value) == [
        "Catholic Relief Services (CRS)",
        "USAID. Bur. for Humanitarian Assistance (BHA)",
        "Johns Hopkins University",
    ]


def test_normalize_keywords_removes_taxonomy_codes_and_scores() -> None:
    value = (
        "Teacher education | Quality of education "
        "EF60 Development cooperation (76.0) | Access to education (1058.0)"
    )

    assert AidDataArchiveClient._normalize_keywords(value) == [
        "Teacher education",
        "Quality of education",
        "Development cooperation",
        "Access to education",
    ]
