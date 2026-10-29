"""Unit tests for evaluation helpers (no API calls)."""

import pytest

from src.evaluation import append_cache, load_cache, retrieval_metrics

def test_retrieval_metrics():
    ranks = [1, 3, None, 2]
    metrics = retrieval_metrics(ranks, k_values=(1, 3))
    assert metrics["recall@1"] == pytest.approx(0.25)
    assert metrics["recall@3"] == pytest.approx(0.75)
    assert metrics["mrr"] == pytest.approx((1 + 1/3 + 0 + 1/2) / 4)

# tmp_path is a pytest fixture: when you include it as a parameter in a test function, pytest automatically creates and provides a temporary directory. 
# Once the test finishes, this directory is cleaned up. This allows the test to read and write files without affecting the actual files in your project.
def test_cache_round_trip(tmp_path):
    path = tmp_path / "cache.jsonl"
    assert load_cache(path) == {}

    append_cache(path, {"doc_id": "a", "verdict": "TRUE", "text": "Türkçe"})
    append_cache(path, {"doc_id": "b", "verdict": "FALSE", "text": "metin"})

    cache = load_cache(path)
    assert set(cache) == {"a", "b"}
    assert cache["a"]["text"] == "Türkçe"