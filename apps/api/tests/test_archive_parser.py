from app.services.archive.aiddata import AidDataArchiveClient


def test_parse_size_kb() -> None:
    assert AidDataArchiveClient._parse_size_kb("1124 KB") == 1124
    assert AidDataArchiveClient._parse_size_kb("1.9 MB") == 1946
    assert AidDataArchiveClient._parse_size_kb(None) is None
