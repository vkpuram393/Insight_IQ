"""
LLM-based entity extractor for drug and pharmacy names.

Regex cannot reliably handle open-ended pharmaceutical product names:
  "DEXCOM G7 SENSOR", "NOVOLOG FLEXPEN 3ML", "5-FU 500MG/10ML",
  "INSULIN LISPRO KWIKPEN", "Metformin 500mg", "HumaLOG Mix 75/25"

This module uses Gemini Flash to extract drug_name and pharmacy_name from
the raw query.  The classifier then replaces those values with generic
placeholders ("drug", "pharmacy") so the embedding focuses on intent
semantics rather than the specific entity text.

The original values flow through unchanged to the downstream API call.
"""

import json
import logging
import os
import re
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Singleton client (same pattern as llm_fallback.py) ───────────────────────

_llm_client = None
_llm_lock = threading.Lock()


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        with _llm_lock:
            if _llm_client is None:
                from google import genai
                _llm_client = genai.Client(
                    vertexai=True,
                    project=os.getenv("PROJECT_ID", "pbm-poc-coderev-genai-poc"),
                    location=os.getenv("LOCATION", "us-central1"),
                )
    return _llm_client


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = """\
You extract drug and pharmacy entities from pharmacy benefit manager (PBM) queries.
Return ONLY a JSON object. Omit any field not clearly present in the query.

Fields:
  "drug_name"     — full name of the drug, medication, or medical device exactly
                    as it appears in the query (includes insulin pens, glucose
                    sensors, compound drugs, inhalers, etc.)
  "pharmacy_name" — name and optional store number of the pharmacy exactly as
                    it appears in the query

Examples:
  Query: "Tell me when INSULIN LISPRO KWIKPEN was taken"
  {"drug_name": "INSULIN LISPRO KWIKPEN"}

  Query: "Was DEXCOM G7 SENSOR covered at CVS Pharmacy 1234?"
  {"drug_name": "DEXCOM G7 SENSOR", "pharmacy_name": "CVS Pharmacy 1234"}

  Query: "Show rejected claims for Metformin 500mg filled at Walgreens"
  {"drug_name": "Metformin 500mg", "pharmacy_name": "Walgreens"}

  Query: "What is the copay on claim 132435151040074 sequence 001?"
  {}

  Query: "Show all claims for member 123456"
  {}
"""

_JSON_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)


def extract_drug_pharmacy_entities(query: str) -> Dict[str, Optional[str]]:
    """Extract drug_name and pharmacy_name from the raw query via Gemini Flash.

    Returns a dict with zero, one, or both keys present.
    Returns {} on any failure so the pipeline degrades gracefully —
    intent detection still works, just without placeholder substitution.
    """
    try:
        from google.genai import types

        client = _get_llm_client()
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=query,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=80,
                system_instruction=_SYSTEM_INSTRUCTION,
            ),
        )
        text = response.text.strip()

        # Strip markdown fences that some model versions emit
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        m = _JSON_RE.search(text)
        if not m:
            logger.debug("LLM entity extractor returned no JSON object")
            return {}

        result = json.loads(m.group(0))
        entities: Dict[str, Optional[str]] = {}
        if result.get("drug_name"):
            entities["drug_name"] = result["drug_name"]
        if result.get("pharmacy_name"):
            entities["pharmacy_name"] = result["pharmacy_name"]

        logger.debug(f"LLM entity extraction: {entities}")
        return entities

    except Exception as exc:
        logger.warning(f"LLM entity extraction failed, continuing without drug/pharmacy entities: {exc}")
        return {}
