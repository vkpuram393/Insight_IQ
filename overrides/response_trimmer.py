"""
overrides.response_trimmer

Reduces the large Prior Authorization API response to a compact, LLM-friendly
structure by:
  - Stripping internal-only fields (priorAuthorizationIdentifier)
  - Deduplicating the member block (extracted once, removed from each record)
  - Removing always-empty fields (universalId, incidentId, clinicalAdministrationCode)
  - Normalizing YYYYMMDD date strings to MM/DD/YYYY
  - Normalizing boolean fields to "Yes"/"No"
  - Normalizing transformCare {"type": " "} to just the trimmed type value
  - Capping records to protect the LLM context window

Typical reduction: ~50-65% smaller than the raw API payload.
"""
from typing import Any, Dict, List, Optional
import re

# ---------------------------------------------------------------------------
# Fields stripped from every PA record unconditionally
# ---------------------------------------------------------------------------
_RECORD_DROP_KEYS = frozenset({
    "priorAuthorizationIdentifier",  # internal ID object (idValue, resourceId)
    "universalId",                   # always empty string in real API
    "incidentId",                    # always empty string in real API
})

# Fields stripped when their value is an empty string / whitespace-only
_STRIP_WHEN_EMPTY = frozenset({
    "clinicalAdministrationCode",
})

# Boolean fields that should be normalized to "Yes" / "No"
_BOOL_FIELDS = frozenset({
    "overrideSpecialtyPrescriptionRejectIndicator",
    "followMeLogicIndicator",
})

_YYYYMMDD = re.compile(r"^\d{8}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_date(value: Any) -> Any:
    """Convert YYYYMMDD → MM/DD/YYYY.  Returns value unchanged if not YYYYMMDD."""
    if isinstance(value, str) and _YYYYMMDD.match(value):
        return f"{value[4:6]}/{value[6:8]}/{value[:4]}"
    return value


def _normalize_bool(value: Any) -> str:
    """Convert Python/JSON bool or 'Y'/'N' string to readable 'Yes'/'No'."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        if value.upper() in ("Y", "YES", "TRUE"):
            return "Yes"
        if value.upper() in ("N", "NO", "FALSE"):
            return "No"
    return str(value)


def _normalize_transform_care(value: Any) -> Any:
    """
    transformCare arrives as {"type": " "} — extract the type value.
    Returns None when the type is blank/whitespace so the key gets dropped.
    """
    if isinstance(value, dict):
        t = (value.get("type") or "").strip()
        return t if t else None
    if isinstance(value, str):
        return value.strip() or None
    return value


def _strip_nulls_and_empty(d: Any) -> Any:
    """Recursively remove keys whose value is None, empty string, or empty dict."""
    if isinstance(d, dict):
        return {
            k: _strip_nulls_and_empty(v)
            for k, v in d.items()
            if v is not None and v != "" and v != {}
        }
    if isinstance(d, list):
        return [_strip_nulls_and_empty(item) for item in d if item is not None]
    return d


# ---------------------------------------------------------------------------
# Per-record trimming
# ---------------------------------------------------------------------------

def _trim_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trim a single PA record:
      - Drop internal / always-empty keys
      - Normalize date strings in effectivePeriod
      - Normalize boolean fields
      - Normalize transformCare
    """
    out: Dict[str, Any] = {}

    for key, value in record.items():
        # Hard-drop list
        if key in _RECORD_DROP_KEYS:
            continue

        # Drop when value is an empty/whitespace string for these specific fields
        if key in _STRIP_WHEN_EMPTY:
            if not (value or "").strip() if isinstance(value, str) else not value:
                continue

        # Normalize effectivePeriod dates
        if key == "effectivePeriod" and isinstance(value, dict):
            value = {
                k: _normalize_date(v) for k, v in value.items()
            }

        # Normalize modifyDateTime (ISO-ish string — leave as-is but keep)
        # Normalize boolean fields
        if key in _BOOL_FIELDS:
            value = _normalize_bool(value)

        # Normalize transformCare
        if key == "transformCare":
            value = _normalize_transform_care(value)
            if value is None:
                continue  # blank type → drop field entirely

        out[key] = value

    return out


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def trim_overrides_response(
    response: Dict[str, Any],
    max_records: int = 50,
) -> Dict[str, Any]:
    """
    Slim the raw Overrides API response before passing to the LLM.

    Input shape (real API):
    {
        "priorAuthorizations": [
            {
                "priorAuthorizationNumber": "...",
                "priorAuthorizationIdentifier": {...},  # dropped
                "authorizedDrugType": "...",
                "authorizedDrugNumber": "...",
                "reasonCode": "...",
                "drugDescription": "...",
                "effectivePeriod": {"dateBegin": "20260202", "dateEnd": "20260202"},
                "modifyDateTime": "...",
                "agentCode": "...",
                "incidentId": "",          # dropped
                "ignoreStatusCode": "...",
                "universalId": "",         # dropped
                "overrideSpecialtyPrescriptionRejectIndicator": true,   # → "Yes"
                "clinicalAdministrationCode": "",  # dropped when empty
                "transformCare": {"type": " "},    # → dropped (blank) or type value
                "followMeLogicIndicator": false,   # → "No"
                "member": { carrierId, accountId, groupId, memberId, personCode }
            },
            ...
        ]
    }

    Output shape:
    {
        "totalRecords": int,
        "member": { carrierId, accountId, groupId, memberId, personCode },  # extracted once
        "priorAuthorizations": [
            { ... trimmed record without member block ... },
            ...
        ]
    }

    Args:
        response:    Raw dict from the Overrides API (or fallback mock).
        max_records: Cap on number of PA records passed to the LLM.

    Returns:
        Compact dict safe for LLM consumption.
    """
    if not response or not isinstance(response, dict):
        return response or {}

    raw_records: List[Dict[str, Any]] = response.get("priorAuthorizations") or []
    total = len(raw_records)

    if not raw_records:
        return {"totalRecords": 0, "member": None, "priorAuthorizations": []}

    # Extract member block once from the first record (identical across all records)
    member_info: Dict[str, Any] = _strip_nulls_and_empty(
        raw_records[0].get("member") or {}
    )

    # Apply cap BEFORE expensive per-record work
    capped = raw_records[:max_records]

    trimmed_records: List[Dict[str, Any]] = []
    for record in capped:
        trimmed = _trim_record(record)
        trimmed.pop("member", None)           # deduplicated to top level
        trimmed = _strip_nulls_and_empty(trimmed)
        trimmed_records.append(trimmed)

    return {
        "totalRecords": total,
        "returnedRecords": len(trimmed_records),
        "member": member_info,
        "priorAuthorizations": trimmed_records,
    }
