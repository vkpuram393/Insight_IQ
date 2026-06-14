"""
Overrides_api.response_trimmer — whitelist + slim raw PA records.

Mirrors the Claims_search_api/llm_query_responder.py _CLAIM_FIELD_WHITELIST pattern:
trim raw API records to business-relevant fields before they enter the LLM
prompt. Reduces token usage and prevents PII bleed (memberId, agentCode are
either dropped or surface-only).
"""

from typing import Any, Dict, List


# Authoritative whitelist — fields the LLM is allowed to see / cite.
# Anything not in this list is dropped from the slim record.
PA_FIELD_WHITELIST: List[str] = [
    "paReferenceNumber",
    "paStatusCode",
    "paStatusDescription",
    "drugName",
    "ndc",
    "gpi",
    "effectiveDate",
    "terminationDate",
    "quantityAllowed",
    "daysSupplyAllowed",
    "approvedRefillCount",
    "rejectReasonCode",
    "rejectReasonDescription",
    "overrideCode",
    "overrideDescription",
    "agentCode",
    "ignoreStatus",
    "drugTypeIndicator",
    "transformCarePlanIndicator",
    "followMeIndicator",
    "clinicalAdminCode",
    "specialtyRxOverrideIndicator",
    "copayAmount",
    "pricingInfo",            # sub-object, kept as-is
    "modificationHistory",    # list of {modDate, modByAgent, modReasonCode}
]


# Fields that contain PII — explicitly excluded from the slim record so the LLM
# never sees them. These are still in the raw response stored in the cache /
# tool_results envelope, but the LLM-visible payload omits them.
PA_PII_FIELDS = frozenset({
    "memberId",
    "memberFirstName",
    "memberLastName",
    "memberDateOfBirth",
    "carrierId",
    "accountId",
    "groupId",
    "personCode",
})


def _slim_pa_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the field whitelist to a single PA record."""
    if not isinstance(rec, dict):
        return {}
    slim: Dict[str, Any] = {}
    for field in PA_FIELD_WHITELIST:
        if field in PA_PII_FIELDS:
            continue
        value = rec.get(field)
        if value is None:
            continue
        slim[field] = value
    return slim


def _sort_by_effective_date_desc(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Newest PAs first; missing dates sort to the end."""
    def _key(r: Dict[str, Any]) -> str:
        return r.get("effectiveDate") or ""
    return sorted(records, key=_key, reverse=True)


def trim_overrides_response(
    raw_response: Dict[str, Any],
    *,
    max_records: int = 25,
) -> List[Dict[str, Any]]:
    """
    Extract and slim PA records from the raw Overrides API response.

    Args:
        raw_response: dict shaped like {"priorAuthorizations": [ {...}, ... ], ...}
        max_records:  hard cap to bound prompt token usage. PA records are
                      denser than claims so 25 is conservative.

    Returns:
        List of slim PA records, sorted newest-first by effectiveDate.
    """
    records = (raw_response or {}).get("priorAuthorizations") or []
    if not isinstance(records, list):
        return []

    slim_records = [_slim_pa_record(r) for r in records if r]
    slim_records = [r for r in slim_records if r]   # drop empties
    sorted_records = _sort_by_effective_date_desc(slim_records)

    if len(sorted_records) > max_records:
        sorted_records = sorted_records[:max_records]

    return sorted_records


def extract_member_summary_from_cagm(cagm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a non-PII member summary block from the CAGM dict for the LLM context.
    Only carrier identifier (last 4 chars) is exposed — full IDs stay masked.
    """
    if not cagm:
        return {}
    def _mask(value: Any) -> str:
        s = str(value or "").strip()
        if not s:
            return ""
        return ("..." + s[-4:]) if len(s) > 4 else s
    return {
        "carrierId":  _mask(cagm.get("carrierId")),
        "accountId":  _mask(cagm.get("accountId")),
        "groupId":    _mask(cagm.get("groupId")),
        "memberId":   _mask(cagm.get("memberId")),
    }
