"""Unit tests for prompt building and guardrails (no API calls)."""

from src.generation import(
    HEADLINE_NOTE, EvidenceAssessment, FactCheckResult,
    apply_guardrails, format_evidence,
)

def make_evidence(is_headline=False):
    return [{
        "source_name": "Teyit", "date": "2023-07-11",
        "claim": "Örnek iddia", "label": "yanlış",
        "is_headline_claim": is_headline,
        "summary": "Örnek özet.", "text": "Örnek bölüm.",
        "url": "https://teyit.org/ornek"
    }]

def test_format_evidence_contains_key_fields():
    text = format_evidence(make_evidence())
    assert "[Kanıt 1]" in text
    assert "Örnek özet." in text
    assert "https://teyit.org/ornek" in text
    assert HEADLINE_NOTE not in text

def test_format_evidence_adds_headline_note():
    assert HEADLINE_NOTE in format_evidence(make_evidence(is_headline=True))

def test_guardrail_forces_not_enough_evidence():
    result = FactCheckResult(
        evidence_assessments=[EvidenceAssessment(evidence_id=1, is_same_claim=False)],
        verdict="FALSE", explanation="...", used_evidence_ids=[1],
    )
    checked = apply_guardrails(result)
    assert checked.verdict == "NOT_ENOUGH_EVIDENCE"
    assert checked.used_evidence_ids == []

def test_guardrail_keeps_valid_verdict():
    result = FactCheckResult(
        evidence_assessments=[EvidenceAssessment(evidence_id=1, is_same_claim=True)],
        verdict="FALSE", explanation="...", used_evidence_ids=[1],
    )
    assert apply_guardrails(result).verdict == "FALSE"