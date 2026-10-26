"""Tests for retrieval: unit tests for ranking logic, integration tests for the full search."""

import pytest

from src import config
from src.retrieval import embedding_text, rank_by_document, retrieve


# ---------- Unit tests: no model, no database ----------

def make_result(rows):
    """Build a fake ChromaDB query result from (chunk_id, text, doc_id, distance) rows."""
    return {
        "ids": [[r[0] for r in rows]],
        "documents": [[r[1] for r in rows]],
        "metadatas": [[{"doc_id": r[2]} for r in rows]],
        "distances": [[r[3] for r in rows]],
    }


def test_embedding_text_format():
    assert embedding_text("İddia", "Metin") == "passage: İddia\nMetin"


def test_rank_by_document_keeps_best_chunk_per_article():
    result = make_result([
        ("a-ev0", "text 1", "a", 0.20),
        ("a-ev1", "text 2", "a", 0.10),   # best chunk of article "a"
        ("b-ev0", "text 3", "b", 0.15),
    ])
    ranked = rank_by_document(result, top_k=5)

    assert [r["doc_id"] for r in ranked] == ["a", "b"]   # one entry per article
    assert ranked[0]["chunk_id"] == "a-ev1"
    assert ranked[0]["score"] == pytest.approx(0.90)     # 1 - distance


def test_rank_by_document_respects_top_k():
    result = make_result([
        ("a-ev0", "t", "a", 0.1),
        ("b-ev0", "t", "b", 0.2),
        ("c-ev0", "t", "c", 0.3),
    ])
    assert len(rank_by_document(result, top_k=2)) == 2


# ---------- Integration tests: need the model and the built index ----------

index_missing = not config.CHROMA_DIR.exists()
requires_index = pytest.mark.skipif(index_missing, reason="ChromaDB index has not been built")

AMAZON_QUERY = "Amazon teslim edilemeyen paketleri 2 euroya satıyormuş"


@requires_index
def test_known_claim_is_retrieved_first():
    results = retrieve(AMAZON_QUERY)
    assert results[0]["doc_id"] == "fctr-2"


@requires_index
def test_excluded_article_is_never_returned():
    results = retrieve(AMAZON_QUERY, top_k=10, exclude_doc_id="fctr-2")
    assert all(r["doc_id"] != "fctr-2" for r in results)


@requires_index
def test_results_are_unique_and_sorted():
    results = retrieve(AMAZON_QUERY, top_k=5)
    doc_ids = [r["doc_id"] for r in results]
    scores = [r["score"] for r in results]

    assert len(doc_ids) == len(set(doc_ids))       # no duplicate articles
    assert scores == sorted(scores, reverse=True)  # highest score first