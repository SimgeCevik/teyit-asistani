"""Generation: build the RAG prompt, ask Gemini and validate the answer."""

import os
import time
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from src import config
from src.retrieval import retrieve
from src.factcheck_api import search_claims, to_evidence

_client = None

def get_client():
    """Create the Gemini client once and reuse it."""
    global _client
    if _client is None:
        load_dotenv(config.PROJECT_ROOT / ".env")
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client

# Answer schema
# Evaluation of a single piece of evidence: which piece of evidence it is (int, integer) and whether it pertains to the same claim (bool, True/False).
class EvidenceAssessment(BaseModel):
    evidence_id: int
    is_same_claim: bool

class FactCheckResult(BaseModel):
    evidence_assessments: list[EvidenceAssessment]
    verdict: Literal["TRUE", "FALSE", "MIXED", "NOT_ENOUGH_EVIDENCE"]
    explanation: str
    used_evidence_ids: list[int]

# Adding summaries
_articles = None

def get_articles():
    """Load the cleaned articles once, indexed by doc_id for fast lookup."""
    global _articles
    if _articles is None:
        _articles = pd.read_parquet(config.FCTR_CLEAN_PATH).set_index("doc_id")
    return _articles

def add_summaries(evidence):
    """Attach each article's summary to its retrieved evidence."""
    articles = get_articles()
    return [{**item, "summary": articles.loc[item["doc_id"], "summary"]} for item in evidence]

# Convert evidence to text
HEADLINE_NOTE = (
    "Not: Yukarıdaki cümle, doğrulamanın sonucunu özetleyen bir başlıktır. "
    "Kuruluşun kararı, sosyal medyada dolaşan orijinal iddia için verilmiştir."
)

def format_evidence(evidence):
    """Turn retrieved evidence into numbered text blocks for the prompt."""
    blocks = []
    for i, item in enumerate(evidence, start=1):
        lines = [
            f"[Kanıt {i}] {item['source_name']}, {item['date']}",
            f"İncelenen iddia: {item['claim']}",
            f"Kuruluşun kararı: {item['label']}",
        ]
        if item["is_headline_claim"]:
            lines.append(HEADLINE_NOTE)

        lines.append(f"Bulguların özeti: {item['summary']}")

        if item["text"]:
            lines.append(f"İlgili bölüm: {item['text']}")
        
        lines.append(f"Kaynak: {item['url']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

# Prompt
PROMPT_TEMPLATE = """Sen Türkçe iddiaları değerlendiren bir doğrulama (fact-check) asistanısın.
Sana bir kullanıcı iddiası ve doğrulama kuruluşlarının arşivinden bulunmuş kanıtlar verilecek.

ADIMLAR:
1. Her kanıt için, kanıttaki "incelenen iddia"nın kullanıcının iddiasıyla AYNI iddia olup
   olmadığına karar ver (is_same_claim). Aynı konuya değinmesi yetmez: aynı kişi, aynı olay
   ve aynı ifade hakkında olmalı.
2. Hiçbir kanıt aynı iddia değilse kararın NOT_ENOUGH_EVIDENCE olmalı.
3. Aynı iddiayı ele alan bir kanıt varsa, kuruluşun kararını kullanıcının iddiasıyla karşılaştır:
   - Kullanıcı, doğrulanan iddiayı tekrarlıyorsa kuruluşun kararını kullan.
   - Kullanıcı, doğrulanan iddianın tersini söylüyorsa kararı ters çevir.
   - Kuruluşun kararı "karma", "kısmen yanlış" veya "yarı doğru" ise MIXED de.
   - Kuruluşun kararı "sonuçlandırılamadı" ise NOT_ENOUGH_EVIDENCE de.
4. Kendi bilgini kullanma; yalnızca verilen kanıtlara dayan.
5. Açıklamayı Türkçe ve en fazla 3 cümle olarak yaz. Dayandığın doğrulamanın kaynağını ve
   tarihini belirt; durumun o tarihten sonra değişmiş olabileceğini unutma.

KULLANICININ İDDİASI:
{claim}

KANITLAR:
{evidence}"""

def build_prompt(claim, evidence):
    """Fill the prompt template with the user's claim and the formatted evidence."""
    return PROMPT_TEMPLATE.format(claim=claim, evidence=format_evidence(evidence))

# Ask gemini
def ask_gemini(prompt, schema=FactCheckResult, max_retries=5):
    """Send the prompt to Gemini and return an answer validated against `schema`."""
    client = get_client()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            if response.parsed is not None:
                return response.parsed
            return schema.model_validate_json(response.text)

        except errors.APIError as e:
            detail = str(e)
            if e.code == 429 and "PerDay" in detail:
                raise RuntimeError("Daily Gemini quota exhausted, try again tomorrow") from e
            if e.code == 429 or e.code >= 500:
                wait = 5 * 2 ** attempt
                print(f"Gemini returned {e.code}, retrying in {wait}s.")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Gemini still unavailable after {max_retries} attempts")

# Guard rules
def apply_guardrails(result):
    """Enforce rules the model must follow, regardless of what it answered."""
    if not any(a.is_same_claim for a in result.evidence_assessments):
        result.verdict = "NOT_ENOUGH_EVIDENCE"
        result.used_evidence_ids = []
    return result

# Pipeline
def fact_check(claim, top_k=3, exclude_doc_id=None):
    """Full RAG pipeline: retrieve, augment, generate, validate."""
    evidence = retrieve(claim, top_k=top_k, exclude_doc_id=exclude_doc_id) # R -> retrieval
    evidence = add_summaries(evidence) # A -> augmentation
    prompt = build_prompt(claim, evidence) # A -> augmentation
    result = apply_guardrails(ask_gemini(prompt)) # G -> generate and check
    return result, evidence

def fact_check_live(claim, max_items=3):
    """Second layer: search recent fact-checks via the Fact Check API and judge them."""
    evidence = to_evidence(search_claims(claim), max_items)
    if not evidence:
        return None, []
    prompt = build_prompt(claim, evidence)
    result = apply_guardrails(ask_gemini(prompt))
    return result, evidence


