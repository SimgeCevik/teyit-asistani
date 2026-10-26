"""Embedding, vector store and retrieval for fact-check chunks."""

import chromadb
from sentence_transformers import SentenceTransformer

from src import config

_embedder = None

def get_embedder():
    """Load the embedding model once and reuse it."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder

def get_collection(reset=False):
    """Open the persistent ChromaDB collection, optionally recreating it."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass # Collection did not exist yet

    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def embedding_text(claim, text):
    """Text that is embedded for a chunk: e5 prefix + claim as context header + chunk."""
    return f"passage: {claim}\n{text}"

METADATA_COLUMNS = [
    "doc_id", "claim", "source_name", "date", "label",
    "verdict_group", "url", "is_headline_claim", "chunk_type",
]

def build_index(chunks_df, batch_size=256):
    """Embed all chunks and store them in a fresh ChromaDB collection."""
    embedder = get_embedder()
    collection = get_collection(reset=True)

    for start in range(0, len(chunks_df), batch_size):
        batch = chunks_df.iloc[start:start + batch_size]
        texts = [embedding_text(c,t) for c, t in zip(batch["claim"], batch["text"])]
        vectors = embedder.encode(texts, normalize_embeddings=True, batch_size=32)

        collection.add(
            ids=batch["chunk_id"].tolist(),
            embeddings=vectors.tolist(),
            documents=batch["text"].tolist(),
            metadatas=batch[METADATA_COLUMNS].to_dict("records"),
        )
        print(f"Indexed {min(start + batch_size, len(chunks_df))} / {len(chunks_df)}")

    return collection

def rank_by_document(result, top_k):
    """Keep the best-scoring chunk per article and return the top_k articles.

    `result` is the dictionary returned by ChromaDB's collection.query().
    """
    best_per_doc = {}
    for chunk_id, text, meta, distance in zip(
        result["ids"][0], result["documents"][0],
        result["metadatas"][0], result["distances"][0],
    ):
        score = 1 - distance  # cosine distance -> cosine similarity
        doc_id = meta["doc_id"]
        if doc_id not in best_per_doc or score > best_per_doc[doc_id]["score"]:
            best_per_doc[doc_id] = {**meta, "chunk_id": chunk_id, "text": text, "score": score}

    ranked = sorted(best_per_doc.values(), key=lambda r: r["score"], reverse=True)
    return ranked[:top_k]


def retrieve(query, top_k=3, n_candidates=30, exclude_doc_id=None):
    """Return the top_k most relevant articles for a claim, one best chunk per article."""
    embedder = get_embedder()
    collection = get_collection()

    query_vector = embedder.encode(f"query: {query}", normalize_embeddings=True)
    where = {"doc_id": {"$ne": exclude_doc_id}} if exclude_doc_id else None

    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=n_candidates,
        where=where,
    )
    return rank_by_document(result, top_k)

        


    
