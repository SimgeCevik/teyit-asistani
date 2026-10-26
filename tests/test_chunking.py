"""Unit tests for chunking."""

from src.chunking import chunk_text, split_sentences

def count_words(text):
    """Simple stand-in for a tokenizer: one word = one token."""
    return len(text.split())

def test_split_sentences():
    assert split_sentences("Bir. İki! Üç?") == ["Bir.", "İki!", "Üç?"]

def test_chunks_respect_limit_and_overlap():
    text = "Bir iki üç. Dört beş altı. Yedi sekiz dokuz. On bir on iki."
    chunks = chunk_text(text, count_words, max_tokens=6, overlap_sentences=1)

    assert chunks == [
        "Bir iki üç. Dört beş altı.",
        "Dört beş altı. Yedi sekiz dokuz.",   # starts with the overlap sentence
        "On bir on iki.",                     # overlap dropped: 3 + 4 words > 6
    ]
    assert all(count_words(c) <= 6 for c in chunks)
