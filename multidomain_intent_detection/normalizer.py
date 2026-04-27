"""
Multidomain Intent Detection — Query Normalizer & Entity Extractor
===================================================================

Text preprocessing pipeline:
  1. normalize_query()   — strips claim/sequence numbers so embeddings
                           focus on intent semantics, not numeric IDs.
  2. extract_entities()  — pulls structured entities (claim_number,
                           sequence_number, NPI, NDC, member_id) from
                           the *raw* query text.
"""

import re
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Compiled Patterns — Query Normalization
# ─────────────────────────────────────────────────────────────────────────────

_CLAIM_NUM_PATTERN = re.compile(r'\b\d{12,18}\b')
_SEQ_PATTERN = re.compile(r'\bsequence\s+\d{1,3}\b', re.IGNORECASE)
_SEQ_NUM = re.compile(r'\bseq\s+\d{1,3}\b', re.IGNORECASE)
_WHITESPACE = re.compile(r'\s+')


def normalize_query(text: str) -> str:
    """Strip claim/sequence numbers so embedding focuses on intent semantics.

    Before:  "Prescriber details for claim 132435151040074 sequence 001."
    After:   "prescriber details for claim"

    This makes training templates and test queries land in the same
    embedding region because the semantic content (not numeric IDs)
    drives the vector.
    """
    t = text.lower().strip()
    t = _SEQ_PATTERN.sub('', t)
    t = _SEQ_NUM.sub('', t)
    t = _CLAIM_NUM_PATTERN.sub('', t)
    t = t.replace('.', ' ').replace('?', ' ').replace('!', ' ')
    t = _WHITESPACE.sub(' ', t).strip()
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Compiled Patterns — Entity Extraction
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_CLAIM_NUM = re.compile(r'\b(\d{15})\b')
_ENTITY_SEQ_NUM = re.compile(r'\bsequence\s+(\d{1,3})\b', re.IGNORECASE)
_ENTITY_NPI = re.compile(r'\bNPI\s+(\d{10})\b', re.IGNORECASE)
_ENTITY_NDC = re.compile(r'\bNDC\s+([\d-]{10,13})\b', re.IGNORECASE)
_ENTITY_MEMBER_ID = re.compile(r'\bmember\s+(?:ID\s+)?(\d{6,12})\b', re.IGNORECASE)
_ENTITY_DRUG_NAME = re.compile(
    r'\b([A-Z]{3,}(?:\s+[A-Z]{2,})?)\b'  # e.g. METFORMIN, ATORVASTATIN CALCIUM
)
_ENTITY_REJECT_CODE = re.compile(r'\breject\s*(?:code)?\s+(\d{1,3}|[A-Z]{2})\b', re.IGNORECASE)
_ENTITY_SETTLEMENT_CODE = re.compile(r'\bsettlement\s+(?:code\s+)?(\d{1,4})\b', re.IGNORECASE)
_ENTITY_DATE_RANGE = re.compile(
    r'\b(january|february|march|april|may|june|july|august|september|'
    r'october|november|december|last\s+month|this\s+month|this\s+year|'
    r'last\s+year|yesterday|today)\b',
    re.IGNORECASE,
)
_ENTITY_PHARMACY_NAME = re.compile(
    r'\b(CVS|WALGREENS?|RITE\s+AID|WALMART|TARGET|COSTCO|KROGER)\s*(PHARMACY)?\s*(\d{3,6})?\b',
    re.IGNORECASE,
)


def extract_entities(text: str) -> Dict[str, Optional[str]]:
    """Extract structured entities from the raw query text.

    Returns a dict containing only the entities that were found.
    Keys: claim_number, sequence_number, npi, ndc, member_id,
          drug_name, reject_code, settlement_code, date_reference,
          pharmacy_name.
    """
    entities: Dict[str, Optional[str]] = {}

    m = _ENTITY_CLAIM_NUM.search(text)
    if m:
        entities["claim_number"] = m.group(1)

    m = _ENTITY_SEQ_NUM.search(text)
    if m:
        entities["sequence_number"] = m.group(1).zfill(3)

    m = _ENTITY_NPI.search(text)
    if m:
        entities["npi"] = m.group(1)

    m = _ENTITY_NDC.search(text)
    if m:
        entities["ndc"] = m.group(1)

    m = _ENTITY_MEMBER_ID.search(text)
    if m:
        entities["member_id"] = m.group(1)

    m = _ENTITY_REJECT_CODE.search(text)
    if m:
        entities["reject_code"] = m.group(1)

    m = _ENTITY_SETTLEMENT_CODE.search(text)
    if m:
        entities["settlement_code"] = m.group(1)

    # Date references (collect all)
    date_matches = _ENTITY_DATE_RANGE.findall(text)
    if date_matches:
        entities["date_reference"] = ", ".join(d.strip() for d in date_matches)

    # Pharmacy name (with optional store number)
    m = _ENTITY_PHARMACY_NAME.search(text)
    if m:
        name = m.group(0).strip()
        entities["pharmacy_name"] = name

    # Drug name — uppercase words ≥3 chars that aren't common stop words
    _STOP_WORDS = {
        "THE", "FOR", "AND", "THIS", "THAT", "WITH", "FROM", "WHAT",
        "WHEN", "WHERE", "WHICH", "SHOW", "TELL", "GIVE", "LIST",
        "DISPLAY", "GENERATE", "RETRIEVE", "FETCH", "PROVIDE", "HOW",
        "DOES", "DID", "WAS", "ARE", "ALL", "HAS", "HAD", "HAVE",
        "NOT", "BUT", "WHO", "WHY", "CLAIM", "CLAIMS", "MEMBER",
        "PHARMACY", "PRESCRIBER", "SEQUENCE", "STATUS", "REJECT",
        "SETTLEMENT", "NDC", "NPI", "PAY", "PAID", "COST", "PRICE",
        "INFORMATION", "DETAILS", "SUMMARY", "REPORT", "OVERRIDE",
        "CVS", "WALGREENS", "WALGREEN", "RITE", "AID", "WALMART",
        "TARGET", "COSTCO", "KROGER", "FILLED", "FILLS",
    }
    # Skip drug extraction if a pharmacy name was already found (avoid overlap)
    pharmacy_text = entities.get("pharmacy_name", "").upper()
    drug_candidates = _ENTITY_DRUG_NAME.findall(text)
    for candidate in drug_candidates:
        if (
            candidate.upper() not in _STOP_WORDS
            and len(candidate) >= 4
            and candidate.upper() not in pharmacy_text
        ):
            entities["drug_name"] = candidate
            break

    return entities
