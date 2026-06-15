"""
Multidomain Intent Detection — Query Normalizer & Entity Extractor
===================================================================

Text preprocessing pipeline:
  1. extract_entities()        — pulls structured entities (claim numbers, NDC,
                                  NPI, member IDs, dates, etc.) from the raw query
                                  using precise regex patterns.
                                  Drug and pharmacy names are handled separately by
                                  llm_entity_extractor.py via Gemini Flash.
  2. sanitize_for_embedding()  — replaces drug/pharmacy names with generic
                                  placeholders so embeddings focus on intent
                                  semantics rather than specific entity values.
  3. normalize_query()         — strips claim/sequence numbers for embedding.
"""

import re
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Compiled Patterns — Query Normalization
# ─────────────────────────────────────────────────────────────────────────────

_CLAIM_NUM_PATTERN = re.compile(r'\b\d{12,18}\b')
_SEQ_PATTERN = re.compile(r'\bsequence\s+\d{1,3}\b', re.IGNORECASE)
_SEQ_NUM = re.compile(r'\bseq\s+\d{1,3}\b', re.IGNORECASE)
_PA_NUM_PATTERN = re.compile(r'\bPA\s+[A-Z0-9]{5,15}\b', re.IGNORECASE)
_NDC_NUM_PATTERN = re.compile(r'\bNDC\s+[\d-]{10,13}\b', re.IGNORECASE)
_WHITESPACE = re.compile(r'\s+')


def normalize_query(text: str) -> str:
    """Strip claim/sequence/PA numbers so embedding focuses on intent semantics.

    Before:  "Prescriber details for claim 132435151040074 sequence 001."
    After:   "prescriber details for claim"

    Also strips PA identifiers: "PA JW012726LC" → "pa"
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # truncate to prevent regex DoS
    t = text.lower().strip()
    t = _SEQ_PATTERN.sub('', t)
    t = _SEQ_NUM.sub('', t)
    t = _CLAIM_NUM_PATTERN.sub('claim_id', t)
    t = _PA_NUM_PATTERN.sub('pa', t)
    t = _NDC_NUM_PATTERN.sub('ndc', t)

    # Removing punctuations
    t = t.replace('.', ' ').replace('?', ' ').replace('!', ' ')

    # Replace multiple spaces with a single space and trim leading/trailing spaces
    t = _WHITESPACE.sub(' ', t).strip()
    return t


# ─────────────────────────────────────────────────────────────────────────
# Compiled Patterns — Structured Entity Extraction
# ─────────────────────────────────────────────────────────────────────────

_ENTITY_CLAIM_NUM = re.compile(r'\b(\d{12,18})\b')
_ENTITY_SEQ_NUM = re.compile(r'\bsequence\s+(\d{1,3})\b', re.IGNORECASE)
_ENTITY_NPI = re.compile(r'\bNPI\s+(\d{10})\b', re.IGNORECASE)
_ENTITY_NDC = re.compile(r'\bNDC\s+([\d-]{10,13})\b', re.IGNORECASE)
_ENTITY_MEMBER_ID = re.compile(r'\bmember\s+(?:ID\s+)?(\d{6,12})\b', re.IGNORECASE)
_ENTITY_REJECT_CODE = re.compile(r'\breject\s*(?:code)?\s+(\d{1,3}|[A-Z]{2})\b', re.IGNORECASE)
_ENTITY_SETTLEMENT_CODE = re.compile(r'\bsettlement\s+(?:code\s+)?(\d{1,4})\b', re.IGNORECASE)
_ENTITY_PA_REASON_CODE = re.compile(
    r'\b(?:reason\s+code|override\s+reason|PA\s+reason)\s+'
    r'(U1|LC|OD|OA|US|U3|MB|ES|HS|PN|OM|PA|2A|2B|2C)\b',
    re.IGNORECASE,
)
_ENTITY_DATE_RANGE = re.compile(
    r'\b(january|february|march|april|may|june|july|august|september|'
    r'october|november|december|last\s+month|this\s+month|this\s+year|'
    r'last\s+year|yesterday|today)\b',
    re.IGNORECASE,
)


def extract_entities(text: str) -> Dict[str, Optional[str]]:
    """Extract structured entities from the raw query text using regex.

    Handles entities that have a fixed, unambiguous format:
    claim numbers, sequence numbers, NPI, NDC, member IDs, reject/settlement
    codes, and date references.

    Drug and pharmacy names are NOT extracted here — open-ended names like
    "DEXCOM G7 SENSOR" or "Metformin 500mg" cannot be reliably matched by
    regex. Use llm_entity_extractor.extract_drug_pharmacy_entities() for those.

    Returns a dict containing only the entities that were found.
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

    date_matches = _ENTITY_DATE_RANGE.findall(text)
    if date_matches:
        entities["date_reference"] = ", ".join(d.strip() for d in date_matches)

    return entities


def sanitize_for_embedding(text: str, entities: Dict[str, Optional[str]]) -> str:
    """Replace drug/pharmacy names with generic placeholders before embedding.

    Ensures the embedding model sees intent-bearing language ("when was the
    drug taken?") rather than specific entity values that do not appear in
    training examples.

    The original entities dict is unchanged and flows through to the API call.

    "Tell me when INSULIN LISPRO KWIKPEN was taken?"
      → "Tell me when drug was taken?"

    "Was DEXCOM G7 SENSOR covered at CVS Pharmacy 1234?"
      → "Was drug covered at pharmacy?"
    """
    sanitized = text
    # Pharmacy first — avoids partial overlap when a word appears in both
    if entities.get("pharmacy_name"):
        sanitized = re.sub(
            re.escape(entities["pharmacy_name"]), "pharmacy", sanitized, flags=re.IGNORECASE
        )
    if entities.get("drug_name"):
        sanitized = re.sub(
            re.escape(entities["drug_name"]), "drug", sanitized, flags=re.IGNORECASE
        )
    return sanitized
