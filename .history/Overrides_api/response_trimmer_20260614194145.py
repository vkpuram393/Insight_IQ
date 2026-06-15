"""
Overrides_api.response_trimmer — normalise + slim raw PA records.

Maps raw Override-API field names → canonical LLM-friendly names, flattens
nested structures (effectivePeriod, transformCare), resolves drug identifiers
(NDC / GPI / List) based on authorizedDrugType, and strips PII sections
(member, priorAuthorizationIdentifier).

Mirrors Claims_search_api/response_trimmer.py in purpose — reduce payload
size and PII surface before the data enters the LLM prompt.
"""

from typing import Any, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Field mapping — actual API field names → normalised LLM-friendly names
# ─────────────────────────────────────────────────────────────────────────────

_API_FIELD_MAP: Dict[str, str] = {
    "priorAuthorizationNumber":                     "paReferenceNumber",
    "authorizedDrugType":                           "drugTypeIndicator",
    "authorizedDrugNumber":                         "authorizedDrugNumber",
    "reasonCode":                                   "reasonCode",
    "drugDescription":                              "drugName",
    "agentCode":                                    "agentCode",
    "ignoreStatusCode":                             "ignoreStatus",
    "overrideSpecialtyPrescriptionRejectIndicator": "specialtyRxOverrideIndicator",
    "clinicalAdministrationCode":                   "clinicalAdminCode",
    "followMeLogicIndicator":                       "followMeIndicator",
    "modifyDateTime":                               "lastModifiedDateTime",
    "incidentId":                                   "incidentId",
}

DRUG_TYPE_LABELS: Dict[str, str] = {
    "N": "NDC",
    "G": "GPI",
    "M": "Manual NDC",
    "1": "Drug List",
    "2": "GPI-based List",
}


# Canonical whitelist of normalised field names the LLM is allowed to see.
PA_FIELD_WHITELIST: List[str] = [
    "paReferenceNumber",
    "drugName",
    "drugTypeIndicator",
    "drugTypeLabel",
    "authorizedDrugNumber",
    "ndc",
    "gpi",
    "drugListId",
    "reasonCode",
    "effectiveDate",
    "terminationDate",
    "agentCode",
    "ignoreStatus",
    "specialtyRxOverrideIndicator",
    "clinicalAdminCode",
    "followMeIndicator",
    "transformCarePlanIndicator",
    "lastModifiedDateTime",
    "incidentId",
]


# Sections that contain PII — dropped entirely from the slim record.
PA_PII_FIELDS = frozenset({
    "member",
    "priorAuthorizationIdentifier",
})


def _slim_pa_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise and slim a single raw PA record from the Override API shape.

    Handles:
      - Flat field renaming via _API_FIELD_MAP
      - Nested field flattening (effectivePeriod, transformCare)
      - Drug identifier resolution (NDC / GPI / List based on authorizedDrugType)
      - Empty / null value stripping
    """
    if not isinstance(rec, dict):
        return {}
    slim: Dict[str, Any] = {}

    # 1. Map flat API fields → normalised names
    for api_field, llm_field in _API_FIELD_MAP.items():
        value = rec.get(api_field)
        if value is None or value == "":
            continue
        slim[llm_field] = value

    # 2. Flatten effectivePeriod → effectiveDate / terminationDate
    eff = rec.get("effectivePeriod") or {}
    if eff.get("dateBegin"):
        slim["effectiveDate"] = eff["dateBegin"]
    if eff.get("dateEnd"):
        slim["terminationDate"] = eff["dateEnd"]

    # 3. Flatten transformCare → transformCarePlanIndicator
    tc = rec.get("transformCare") or {}
    tc_type = (tc.get("type") or "").strip()
    if tc_type:
        slim["transformCarePlanIndicator"] = tc_type

    # 4. Resolve drug identifier based on authorizedDrugType
    drug_type = rec.get("authorizedDrugType", "")
    drug_number = rec.get("authorizedDrugNumber", "")
    if drug_number:
        if drug_type in ("N", "M"):
            slim["ndc"] = drug_number
        elif drug_type == "G":
            slim["gpi"] = drug_number
        elif drug_type in ("1", "2"):
            slim["drugListId"] = drug_number

    # 5. Add human-readable drug type label
    if drug_type:
        slim["drugTypeLabel"] = DRUG_TYPE_LABELS.get(drug_type, drug_type)

    return slim


def _sort_pa_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort by lastModifiedDateTime descending (newest first), effectiveDate as tiebreaker."""
    def _key(r: Dict[str, Any]) -> tuple:
        return (
            r.get("lastModifiedDateTime") or "",
            r.get("effectiveDate") or "",
        )
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
    sorted_records = _sort_pa_records(slim_records)

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
