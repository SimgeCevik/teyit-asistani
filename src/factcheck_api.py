"""Live search in the Google Fact Check Tools API."""

import os
import time

import requests
from dotenv import load_dotenv

from src import config

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def search_claims(query, language="tr", page_size=10, max_retries=3):
    """Return claim records matching the query from the Fact Check API."""
    load_dotenv(config.PROJECT_ROOT / ".env")
    params = {
        "query": query,
        "languageCode": language,
        "pageSize": page_size,
        "key": os.getenv("FACTCHECK_API_KEY")
    }

    for attempt in range(max_retries):
        response = requests.get(API_URL, params=params, timeout=10)
        if response.status_code == 100:
            return response.json().get("claims", [])
        if response.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
    raise RuntimeError("Fact Check API is unavailable, please try again later")

def to_evidence(claims, max_items=3):
    """Convert API claims into the evidence format used by the prompt and the app."""
    evidence = []
    for claim in claims:
        for review in claim.get("claimReview", []):
            evidence.append({
                "doc_id": review.get("url", ""),
                "source_name": review.get("publisher", {}).get("name", "Bilinmeyen kaynak"),
                "date": review.get("reviewDate", claim.get("claimDate", ""))[:10],
                "claim": claim.get("text", ""),
                "label": review.get("textualRating", "belirtilmemiş"),
                "is_headline_claim": False,
                "summary": review.get("title", ""),
                "text": "",
                "url": review.get("url", ""),
            })
    return evidence[:max_items]
