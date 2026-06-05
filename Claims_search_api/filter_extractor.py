"""
Claims_search_api.filter_extractor

LLM-based structured filter extraction + deterministic Python application.

Problem solved
--------------
The raw claims API returns up to 500 claims (~70 K lines of JSON).  Only a
small subset is relevant to any given user query.  The old approach
(search.py / generalized_claims_query) used cascading regex — brittle,
early-exit, and silently returns ALL claims when no pattern matches.

This module replaces that with two steps:

  1. extract_filter_spec(user_query) — a tiny LLM call (~200 prompt tokens,
     ~100 completion tokens).  The LLM receives ONLY the user's question and
     outputs a FilterSpec JSON.  The 70 K-line payload never goes to this LLM.

  2. apply_filter_spec(claims, spec) — pure Python, no LLM, microseconds.
     Applies every non-None field in the FilterSpec as an AND filter using the
     existing atomic filter functions from search.py.

Result: only the matching subset (e.g. 3-20 claims) is sent to the
summarisation LLM, keeping that prompt small regardless of history size.

Public API
----------
    extract_filter_spec(user_query, current_date=None) -> FilterSpec
    apply_filter_spec(claims, spec) -> List[dict]
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FilterSpec
# ---------------------------------------------------------------------------

@dataclass
class FilterSpec:
    """Structured representation of what the user is asking for."""

    # Drug identity
    drug_name: Optional[str] = None
    is_generic: Optional[bool] = None
    is_compound: Optional[bool] = None
    is_specialty: Optional[bool] = None
    ndc: Optional[str] = None
    gpi: Optional[str] = None
    manufacturer: Optional[str] = None

    # Claim status / codes
    status: Optional[str] = None           # "P" | "R" | "X"
    reject_code: Optional[str] = None
    settlement_code: Optional[str] = None
    daw_code: Optional[str] = None         # "1"–"9" non-zero DAW
    has_prior_auth: Optional[bool] = None
    has_reversal: Optional[bool] = None
    has_pricing: Optional[bool] = None

    # Date filters — most-specific wins; precedence: fill_date > range > month+year > month > year
    fill_date: Optional[str] = None        # exact YYYY-MM-DD
    date_from: Optional[str] = None        # range start YYYY-MM-DD
    date_to: Optional[str] = None          # range end   YYYY-MM-DD
    month: Optional[int] = None            # 1–12
    year: Optional[int] = None

    # Prescription identifiers
    rx_number: Optional[str] = None
    claim_number: Optional[str] = None
    refill_number: Optional[str] = None    # "00" = original fill
    days_supply: Optional[str] = None
    diagnosis_code: Optional[str] = None
    plan_id: Optional[str] = None

    # Pharmacy
    pharmacy_name: Optional[str] = None
    pharmacy_city: Optional[str] = None
    pharmacy_state: Optional[str] = None
    pharmacy_type: Optional[str] = None    # "Retail" | "Mail"

    # Prescriber
    prescriber_name: Optional[str] = None
    prescriber_npi: Optional[str] = None

    def is_empty(self) -> bool:
        """True when no filter criteria have been set."""
        return all(v is None for v in self.__dict__.values())


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = (
    "You are a structured data extractor for a pharmacy claims system.\n"
    "Extract filter criteria from the user's question as a JSON object.\n"
    "Only include fields that are explicitly or clearly implied by the question.\n"
    "Leave everything else as null.\n"
    "Output ONLY valid JSON — no explanation, no markdown, no code fences."
)

_EXTRACTION_SCHEMA = """{
  "drug_name":       string or null,
  "is_generic":      true/false/null,
  "is_compound":     true/false/null,
  "is_specialty":    true/false/null,
  "ndc":             string or null,
  "gpi":             string or null,
  "manufacturer":    string or null,
  "status":          "P"/"R"/"X"/null,
  "reject_code":     string or null,
  "settlement_code": string or null,
  "daw_code":        string or null,
  "has_prior_auth":  true/false/null,
  "has_reversal":    true/false/null,
  "has_pricing":     true/false/null,
  "fill_date":       "YYYY-MM-DD"/null,
  "date_from":       "YYYY-MM-DD"/null,
  "date_to":         "YYYY-MM-DD"/null,
  "month":           1-12/null,
  "year":            YYYY/null,
  "rx_number":       string or null,
  "claim_number":    string or null,
  "refill_number":   "00"-"99"/null,
  "days_supply":     string or null,
  "diagnosis_code":  string or null,
  "plan_id":         string or null,
  "pharmacy_name":   string or null,
  "pharmacy_city":   string or null,
  "pharmacy_state":  string or null,
  "pharmacy_type":   "Retail"/"Mail"/null,
  "prescriber_name": string or null,
  "prescriber_npi":  string or null
}"""

# Status synonyms resolved before the LLM sees them — avoids ambiguity
_STATUS_HINT = (
    'status values: "P"=paid/approved, "R"=rejected/denied, "X"=reversed/cancelled'
)

_RELATIVE_DATE_HINT = (
    "Resolve relative dates (this month, last month, last year, yesterday, etc.) "
    "to absolute YYYY-MM-DD or month/year integers using today's date."
)


def _build_extraction_prompt(user_query: str, current_date: date) -> str:
    return (
        f"{_EXTRACTION_SYSTEM}\n\n"
        f"Today's date: {current_date.isoformat()}\n"
        f"Note: {_STATUS_HINT}\n"
        f"Note: {_RELATIVE_DATE_HINT}\n\n"
        f"Output schema (fill values, null for unspecified):\n"
        f"{_EXTRACTION_SCHEMA}\n\n"
        f"User question: {user_query.strip()}"
    )


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Strip markdown fences or prose wrapping around a JSON object."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        return m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _opt_str(v: Any) -> Optional[str]:
    if v is None or v == "":
        return None
    return str(v).strip() or None


def _opt_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "yes", "1")


def _opt_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _parse_spec(data: dict) -> FilterSpec:
    return FilterSpec(
        drug_name=_opt_str(data.get("drug_name")),
        is_generic=_opt_bool(data.get("is_generic")),
        is_compound=_opt_bool(data.get("is_compound")),
        is_specialty=_opt_bool(data.get("is_specialty")),
        ndc=_opt_str(data.get("ndc")),
        gpi=_opt_str(data.get("gpi")),
        manufacturer=_opt_str(data.get("manufacturer")),
        status=_opt_str(data.get("status")),
        reject_code=_opt_str(data.get("reject_code")),
        settlement_code=_opt_str(data.get("settlement_code")),
        daw_code=_opt_str(data.get("daw_code")),
        has_prior_auth=_opt_bool(data.get("has_prior_auth")),
        has_reversal=_opt_bool(data.get("has_reversal")),
        has_pricing=_opt_bool(data.get("has_pricing")),
        fill_date=_opt_str(data.get("fill_date")),
        date_from=_opt_str(data.get("date_from")),
        date_to=_opt_str(data.get("date_to")),
        month=_opt_int(data.get("month")),
        year=_opt_int(data.get("year")),
        rx_number=_opt_str(data.get("rx_number")),
        claim_number=_opt_str(data.get("claim_number")),
        refill_number=_opt_str(data.get("refill_number")),
        days_supply=_opt_str(data.get("days_supply")),
        diagnosis_code=_opt_str(data.get("diagnosis_code")),
        plan_id=_opt_str(data.get("plan_id")),
        pharmacy_name=_opt_str(data.get("pharmacy_name")),
        pharmacy_city=_opt_str(data.get("pharmacy_city")),
        pharmacy_state=_opt_str(data.get("pharmacy_state")),
        pharmacy_type=_opt_str(data.get("pharmacy_type")),
        prescriber_name=_opt_str(data.get("prescriber_name")),
        prescriber_npi=_opt_str(data.get("prescriber_npi")),
    )


# ---------------------------------------------------------------------------
# Public: extract_filter_spec
# ---------------------------------------------------------------------------

def extract_filter_spec(
    user_query: str,
    current_date: Optional[date] = None,
) -> FilterSpec:
    """
    Call the LLM to extract a FilterSpec from a natural-language query.

    The LLM sees only the user query (~50 words) — never the claims payload.
    The call costs ~200 prompt tokens and ~100 completion tokens.
    Falls back to an empty FilterSpec (= return all claims) on any error.
    """
    from services.llm_connection import generate  # late import — avoids circular dep

    if not user_query or not user_query.strip():
        return FilterSpec()

    if current_date is None:
        current_date = datetime.now().date()

    prompt = _build_extraction_prompt(user_query, current_date)

    try:
        raw = generate(prompt, temperature=0.0, max_output_tokens=512)
        spec = _parse_spec(json.loads(_extract_json(raw)))
        logger.info("[FilterExtractor] Extracted spec: %s", spec)
        return spec
    except json.JSONDecodeError as exc:
        logger.warning("[FilterExtractor] JSON parse error (%s) — returning empty spec", exc)
        return FilterSpec()
    except Exception as exc:
        logger.warning("[FilterExtractor] Extraction failed (%s) — returning empty spec", exc)
        return FilterSpec()


# ---------------------------------------------------------------------------
# Public: apply_filter_spec
# ---------------------------------------------------------------------------

def apply_filter_spec(
    claims: List[Dict[str, Any]],
    spec: FilterSpec,
) -> List[Dict[str, Any]]:
    """
    Apply a FilterSpec to a list of claims using AND logic.

    Every non-None field narrows the result set.  All claims matching ALL
    criteria are returned, sorted newest-first.  Returns all claims
    (sorted newest-first) when the spec is empty.
    """
    from Claims_search_api.search import (
        filter_claims_by_claim_number,
        filter_claims_by_compound,
        filter_claims_by_date_range,
        filter_claims_by_days_supply,
        filter_claims_by_diagnosis_code,
        filter_claims_by_daw,
        filter_claims_by_drug_name,
        filter_claims_by_fill_date,
        filter_claims_by_generic_indicator,
        filter_claims_by_gpi,
        filter_claims_by_manufacturer,
        filter_claims_by_month,
        filter_claims_by_ndc,
        filter_claims_by_pharmacy,
        filter_claims_by_pharmacy_city,
        filter_claims_by_pharmacy_state,
        filter_claims_by_pharmacy_type,
        filter_claims_by_plan,
        filter_claims_by_prescriber,
        filter_claims_by_prescriber_npi,
        filter_claims_by_refill_number,
        filter_claims_by_reject_code,
        filter_claims_by_rx_number,
        filter_claims_by_settlement_code,
        filter_claims_by_specialty,
        filter_claims_by_status,
        filter_claims_with_prior_auth,
        filter_claims_with_reversal,
    )

    result = list(claims)

    # --- Drug ---
    if spec.drug_name:
        result = filter_claims_by_drug_name(result, spec.drug_name)
    if spec.is_generic is not None:
        result = filter_claims_by_generic_indicator(result, spec.is_generic)
    if spec.is_compound is not None:
        result = filter_claims_by_compound(result, spec.is_compound)
    if spec.is_specialty is not None:
        result = filter_claims_by_specialty(result, spec.is_specialty)
    if spec.ndc:
        result = filter_claims_by_ndc(result, spec.ndc)
    if spec.gpi:
        result = filter_claims_by_gpi(result, spec.gpi)
    if spec.manufacturer:
        result = filter_claims_by_manufacturer(result, spec.manufacturer)

    # --- Claim status / codes ---
    if spec.status:
        result = filter_claims_by_status(result, spec.status)
    if spec.reject_code:
        result = filter_claims_by_reject_code(result, spec.reject_code)
    if spec.settlement_code:
        result = filter_claims_by_settlement_code(result, spec.settlement_code)
    if spec.daw_code:
        result = filter_claims_by_daw(result, daw_code=spec.daw_code)
    if spec.has_prior_auth is True:
        result = filter_claims_with_prior_auth(result)
    if spec.has_reversal is True:
        result = filter_claims_with_reversal(result)
    if spec.has_pricing is True:
        result = [
            c for c in result
            if any(v is not None for v in (c.get("pricing") or {}).values())
        ]

    # --- Dates (most-specific wins) ---
    if spec.fill_date:
        result = filter_claims_by_fill_date(result, spec.fill_date)
    elif spec.date_from or spec.date_to:
        try:
            start = date.fromisoformat(spec.date_from) if spec.date_from else date.min
            end = date.fromisoformat(spec.date_to) if spec.date_to else date.max
            result = filter_claims_by_date_range(result, start, end)
        except ValueError:
            logger.warning(
                "[FilterExtractor] Invalid date range %s – %s; skipping date filter",
                spec.date_from, spec.date_to,
            )
    elif spec.month and spec.year:
        result = filter_claims_by_month(result, spec.year, spec.month)
    elif spec.month:
        result = [
            c for c in result
            if (c.get("claimInformation") or {}).get("fillDate", "")[5:7]
            == f"{spec.month:02d}"
        ]
    elif spec.year:
        result = [
            c for c in result
            if (c.get("claimInformation") or {}).get("fillDate", "").startswith(str(spec.year))
        ]

    # --- Prescription identifiers ---
    if spec.rx_number:
        result = filter_claims_by_rx_number(result, spec.rx_number)
    if spec.claim_number:
        result = filter_claims_by_claim_number(result, spec.claim_number)
    if spec.refill_number:
        result = filter_claims_by_refill_number(result, spec.refill_number)
    if spec.days_supply:
        result = filter_claims_by_days_supply(result, spec.days_supply)
    if spec.diagnosis_code:
        result = filter_claims_by_diagnosis_code(result, spec.diagnosis_code)
    if spec.plan_id:
        result = filter_claims_by_plan(result, spec.plan_id)

    # --- Pharmacy ---
    if spec.pharmacy_name:
        result = filter_claims_by_pharmacy(result, spec.pharmacy_name)
    if spec.pharmacy_city:
        result = filter_claims_by_pharmacy_city(result, spec.pharmacy_city)
    if spec.pharmacy_state:
        result = filter_claims_by_pharmacy_state(result, spec.pharmacy_state)
    if spec.pharmacy_type:
        result = filter_claims_by_pharmacy_type(result, spec.pharmacy_type)

    # --- Prescriber ---
    if spec.prescriber_name:
        result = filter_claims_by_prescriber(result, spec.prescriber_name)
    if spec.prescriber_npi:
        result = filter_claims_by_prescriber_npi(result, spec.prescriber_npi)

    result.sort(
        key=lambda c: (c.get("claimInformation") or {}).get("fillDate") or "",
        reverse=True,
    )
    return result
