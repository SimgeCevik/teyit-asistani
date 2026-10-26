"""Split fact-check articles into retrieval chunks."""

import re

import pandas as pd

def split_sentences(text):
    """Split text into sentences at '.', '!' or '?' followed by whitespace."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def chunk_text(text, count_tokens, max_tokens=300, overlap_sentences=1):
    """Pack consecutive sentences into chunks of at most max_tokens tokens."""
    sentences = split_sentences(text)
    chunks, current, current_tokens = [], [], 0

    for sentence in sentences:
        n_tokens = count_tokens(sentence)

        # Current chunk is full: save it and start a new one with some overlap
        if current and current_tokens + n_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_tokens = sum(count_tokens(s) for s in current)

            # If overlap + new sentence would still exceed the limit, drop the overlap
            if current_tokens + n_tokens > max_tokens:
                current, current_tokens = [], 0

        current.append(sentence)
        current_tokens += n_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks

def build_chunks(df, count_tokens, max_tokens=300, overlap_sentences=1):
    """Turn each article into one summary chunk plus several evidence chunks."""
    rows = []
    for doc in df.itertuples():
        metadata = {
            "doc_id": str(doc.doc_id),
            "claim": str(doc.claim),
            "source_name": str(doc.source_name),
            "date": doc.date.strftime("%Y-%m-%d"),
            "label": str(doc.label),
            "verdict_group": str(doc.verdict_group),
            "url": str(doc.url),
            "is_headline_claim": bool(doc.is_headline_claim),
        }

        rows.append({
            **metadata,
            "chunk_id": f"{doc.doc_id}-summary",
            "chunk_type": "summary",
            "text": doc.summary,
        })

        evidence_chunks = chunk_text(doc.evidence, count_tokens, max_tokens, overlap_sentences)
        for i, chunk in enumerate(evidence_chunks):
            rows.append({
                **metadata,
                "chunk_id": f"{doc.doc_id}-ev{i}",
                "chunk_type": "evidence",
                "text": chunk,
            })

    return pd.DataFrame(rows)
