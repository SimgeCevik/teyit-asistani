"""Cleaning utilities for Turkish fact-check texts and labels."""

import re
import unicodedata

import pandas as pd

# CamelCase words that the heading rule would wrongly split.
# Built by reviewing CamelCase words that appear 3+ times in the FCTR evidence texts.
PROTECTED_WORDS = [
    # Platforms and brands
    "WhatsApp", "LinkedIn", "BuzzFeed", "BlackRock", "OceanGate", "BelkoAir",
    "AccuWeather", "MarineTraffic", "CasioTeam",
    # Media outlets
    "HaberTürk", "SuperHaber", "NewsPunch", "InfoWars", "ÇapaMag", "WeedFeed", "SuperSaf",
    # Fact-checking organizations and tools
    "PolitiFact", "FactCheck", "FullFact", "CheckYourFact", "MythDetector",
    "FlightRadar", "FotoForensics", "VigiAccess", "EudraVigilance",
    # Vaccines, pharma and science
    "AstraZeneca", "CoronaVac", "SinoVac", "BionTech", "GlaxoSmithKline",
    "BioCubaFarma", "EctoLife", "bioRxiv",
    # Social media handles
    "AzizGwenSancar", "GenclikHudPar", "SheikhResit", "DuruTulinduru3",
    "ErdalBesikci06", "AhmedMia", "AdalarErdemGul",
    # Other
    "CryptoRobo", "SpecTor",
]

RELATED_LINK_MARKER = "İlginizi çekebilir:"

LABEL_FIXES = {"kismen yanliş": "kısmen yanlış"}

VERDICT_GROUPS = {
    "doğru": "TRUE",
    "çoğunlukla doğru": "TRUE",
    "yanlış": "FALSE",
    "çoğunlukla yanlış": "FALSE",
    "karma": "MIXED",
    "yarı doğru": "MIXED",
    "kısmen yanlış": "MIXED",
    "sonuçlandırılamadı": "UNPROVEN",
}

SOURCE_NAMES = {
    "teyit.org": "Teyit",
    "dogrulukpayi.com": "Doğruluk Payı",
    "dogrula.org": "Doğrula",
}

SENTENCE_END_CHARS = set('.!?"”)')

def remove_related_links(text, tail=200):
    """Remove a trailing 'İlginizi çekebilir: ...' recommendation."""
    pos = text.rfind(RELATED_LINK_MARKER)

    if pos != -1 and pos >= len(text) - tail:
        return text[:pos]
    return text

def split_glued_text(text):
    """Insert missing separators between glued sentences and headings."""
    # 1. Temporarily replace protected words so the regex cannot split them
    placeholders = {}
    for i, word in enumerate(PROTECTED_WORDS):
        if word in text:
            key = f"@@{i}@@"
            text = text.replace(word, key)
            placeholders[key] = word

    # 2. "edildi.İddialara" -> "edildi. İddialara"
    text  = re.sub(r"([.!?])([A-ZÇĞİÖŞÜ])", r"\1 \2", text)

    # 3. "eklenmişGörseldeki" -> "eklenmiş. Görseldeki"
    text = re.sub(r"([a-zçğıöşü]{3})([A-ZÇĞİÖŞÜ][a-zçğıöşü])", r"\1. \2", text)

    # 4. Put the protected words back
    for key, word in placeholders.items():
        tezt = text.replace(key, word)
    return text

def clean_text(text):
    """Full cleaning pipeline for a single text."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)

    text = remove_related_links(text)

    text = split_glued_text(text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

def is_headline_style(claim):
    """True if a claim reads like a conclusion headline rather than a claim."""
    claim = claim.strip()
    return not re.search(r"(iddiası|iddia edildi\.?|\?)$", claim)

def clean_fctr(df):
    """Clean the raw FCTR dataframe and return a new dataframe."""
    out = df.copy()

    # Text
    out["claim"] = out["claim"].map(
        lambda t: re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip()
    )

    out["evidence"] = out["evidence"].map(clean_text)

    out["summary"] = out["summary"].map(clean_text)

    # Labels: keep the original, fix spelling, add normalized groups
    out["label_original"] = out["label"]

    out["label"] = out["label"].replace(LABEL_FIXES)

    out["verdict_group"] = out["label"].map(VERDICT_GROUPS)

    out["binary_label"] = out["label"].map({"doğru": "TRUE", "yanlış": "FALSE"})

    # Metadata
    out["date"] = pd.to_datetime(out["date"], format="%d/%m/%Y")

    out["source"] = out["url"].str.extract(r"https?://(?:www\.)?([^/]+)")[0]

    out["source_name"] = out["source"].map(SOURCE_NAMES)

    out["doc_id"] = "fctr-" + out["claim_id"].astype(str)

    # Quality flags (kept, not removed)
    out["is_headline_claim"] = out["claim"].map(is_headline_style)

    last_char = out["evidence"].str[-1].fillna("")

    out["possibly_truncated"] = ~last_char.isin(SENTENCE_END_CHARS)

    columns = [
        "doc_id", "claim_id", "source", "source_name", "date",
        "claim", "evidence", "summary",
        "label_original", "label", "verdict_group", "binary_label",
        "is_headline_claim", "possibly_truncated", "url",
    ]

    return out[columns]





