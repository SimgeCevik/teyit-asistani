"""Unit tests for converting Fact Check API results (no network calls)."""

from src.factcheck_api import to_evidence


def test_to_evidence_maps_fields():
    claims = [{
        "text": "Örnek iddia",
        "claimReview": [{
            "publisher": {"name": "Teyit"},
            "url": "https://teyit.org/ornek",
            "title": "Görüntüler eski",
            "reviewDate": "2026-10-21T00:00:00Z",
            "textualRating": "Yanlış",
        }],
    }]
    item = to_evidence(claims)[0]
    assert item["source_name"] == "Teyit"
    assert item["date"] == "2026-10-21"
    assert item["label"] == "Yanlış"
    assert item["summary"] == "Görüntüler eski"


def test_to_evidence_handles_missing_fields():
    item = to_evidence([{"claimReview": [{}]}])[0]
    assert item["source_name"] == "Bilinmeyen kaynak"
    assert item["date"] == ""
    assert item["label"] == "belirtilmemiş"


def test_to_evidence_limits_results():
    claims = [{"text": str(i), "claimReview": [{"url": str(i)}]} for i in range(10)]
    assert len(to_evidence(claims, max_items=3)) == 3