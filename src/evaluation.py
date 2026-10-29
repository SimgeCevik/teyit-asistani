"""Evaluation: build a test set, measure retrieval and run the pipelines with caching."""

# Imports
import json
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from src.generation import add_summaries, apply_guardrails, ask_gemini, build_prompt
from src.retrieval import retrieve

# Build test set
def build_test_set(df, n_per_class=50, seed=42):
    """Sample a balanced test set of real claims labelled TRUE or FALSE."""
    candidates = df[df["binary_label"].notna() & ~df["is_headline_claim"]]
    test = candidates.groupby("binary_label", group_keys=False).sample(
        n=n_per_class, random_state=seed
    )
    return test[["doc_id", "claim", "binary_label"]].reset_index(drop=True)

# Paraphrase
class Paraphrases(BaseModel):
    claims: list[str]

PARAPHRASE_PROMPT = """Aşağıda doğrulama kuruluşlarının incelediği iddiaların başlıkları var.
Her birini, iddiayı sosyal medyada paylaşan sıradan bir kullanıcının yazacağı gibi
tek bir cümleye dönüştür.

KURALLAR:
- Anlamı birebir koru: İddiadan daha güçlü, daha zayıf veya farklı bir şey söyleme.
  Örneğin "yıkılmadı" ile "zarar görmedi" farklı anlamlardır.
- Kişi, kurum ve yer adlarını, kullanıcı adlarını, sayıları ve tarihleri aynen koru.
- İddianın konusu olan özel bir kelime varsa (örneğin bir lakabın kökeni) o kelimeyi koru.
- İddiayı, onu paylaşan kişinin ağzından ve aynı yönde yaz.
- Soru biçimindeki başlıkları olumlu bir cümleye çevir.
- Cümle yapısını ve sıradan kelimeleri değiştir; başlığı olduğu gibi kopyalama.
- "iddiası", "iddia edildi", "paylaşımda" gibi ifadeler kullanma.
- İddianın doğru ya da yanlış olduğunu belirtme.
- Her başlık için tam olarak bir cümle yaz ve sırayı koru.

BAŞLIKLAR:
{claims}"""

def paraphrase_claims(claims, batch_size=20):
    """Rewrite archive claims as user-style statements, a batch at a time."""
    paraphrases = []
    for start in range(0, len(claims), batch_size):
        batch = claims[start:start + batch_size]
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(batch, start=1))

        result = ask_gemini(PARAPHRASE_PROMPT.format(claims=numbered), schema=Paraphrases)
        if len(result.claims) != len(batch):
            raise ValueError(f"Expected {len(batch)} paraphrases, got {len(result.claims)}")

        paraphrases.extend(result.claims)
        time.sleep(5)
    return paraphrases

# Retrieval metrics
def gold_rank(query, gold_doc_id, max_k=10):
    """Position (1-based) of the correct article in the results, or None if not found."""
    results = retrieve(query, top_k=max_k, n_candidates=50)
    doc_ids = [r["doc_id"] for r in results]
    if gold_doc_id in doc_ids:
        return doc_ids.index(gold_doc_id) + 1
    return None

def retrieval_metrics(ranks, k_values=(1, 3, 5, 10)):
    """Recall@k and MRR from a list of gold ranks (None = not found)."""
    n = len(ranks)
    metrics = {}
    for k in k_values:
        found = sum(1 for r in ranks if r is not None and r <= k)
        metrics[f"recall@{k}"] = found / n
    metrics["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return metrics

# Cache
def load_cache(path):
    """Read saved results (one JSON object per line) into a dict keyed by doc_id."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {row["doc_id"]: row for row in rows}

def append_cache(path, row):
    """Append one result as a new line to the cache file."""
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# RAG
def run_rag(test_df, cache_path, exclude_own=False, sleep=5):
    """Run the RAG pipeline on every test claim, skipping claims already in the cache."""
    cache = load_cache(cache_path)
    todo = test_df[~test_df["doc_id"].isin(cache)]
    print(f"{len(cache)} cached, {len(todo)} to run")

    for i, row in enumerate(todo.itertuples(), start=1):
        exclude = row.doc_id if exclude_own else None
        evidence = add_summaries(retrieve(row.query, exclude_doc_id=exclude))
        raw = ask_gemini(build_prompt(row.query, evidence))
        raw_verdict = raw.verdict
        result = apply_guardrails(raw)

        record = {
            "doc_id": row.doc_id,
            "query": row.query,
            "expected": "NOT_ENOUGH_EVIDENCE" if exclude_own else row.binary_label,
            "raw_verdict": raw_verdict,
            "verdict": result.verdict,
            "guardrail_applied": raw_verdict != result.verdict,
            "evidence_doc_ids": [e["doc_id"] for e in evidence],
            "same_claim": [a.is_same_claim for a in result.evidence_assessments],
            "explanation": result.explanation,
        }
        append_cache(cache_path, record)
        cache[row.doc_id] = record

        if i % 10 == 0:
            print(f" {i} / {len(todo)} done")
        time.sleep(sleep)
    
    return pd.DataFrame(list(cache.values()))

# No RAG
class NoRagResult(BaseModel):
    verdict: Literal["TRUE", "FALSE", "NOT_ENOUGH_EVIDENCE"]
    explanation: str

NO_RAG_PROMPT = """Sen Türkçe iddiaları değerlendiren bir doğrulama (fact-check) asistanısın.
Aşağıdaki iddianın doğru olup olmadığına karar ver. Emin değilsen NOT_ENOUGH_EVIDENCE de.
Açıklamayı Türkçe ve en fazla 2 cümle olarak yaz.

İDDİA:
{claim}"""

def run_no_rag(test_df, cache_path, sleep=5):
    """Ask Gemini about each claim without any retrieved evidence."""
    cache = load_cache(cache_path)
    todo = test_df[~test_df["doc_id"].isin(cache)]
    print(f"{len(cache)} cached, {len(todo)} to run")

    for i, row in enumerate(todo.itertuples(), start=1):
        result = ask_gemini(NO_RAG_PROMPT.format(claim=row.query), schema=NoRagResult)
        record = {
            "doc_id": row.doc_id,
            "query": row.query,
            "expected": row.binary_label,
            "verdict": result.verdict,
            "explanation": result.explanation,
        }
        append_cache(cache_path, record)
        cache[row.doc_id] = record

        if i % 10 == 0:
            print(f" {i} / {len(todo)} done")
        time.sleep(sleep)

    return pd.DataFrame(list(cache.values()))
