"""
Claims_search_api.response_trimmer

Reduces the large claims API response to a compact, LLM-friendly structure
by removing null/empty fields, deduplicating member data, stripping internal
audit fields, and condensing redundant qualifier descriptions.

Typical reduction: ~60-70% smaller payload.
"""
from typing import Any, Dict, List, Optional
import copy


# ---------------------------------------------------------------------------
# Fields that are purely internal / not useful for LLM answers
# ---------------------------------------------------------------------------
_CLAIM_INFO_DROP_KEYS = frozenset({
    "addUser", "addProgram", "changeUser", "changeProgram",
    "trackingnumber", "locationCode", "adjudicationEnvironment",
    "extractStatus", "extractStatusDescription",
    "adminType", "adminTypeDescription",
    "participationCode", "participationCodeDescription",
    "scc", "addDate", "addTime", "changeDate", "changeTime",
})

_MEMBER_DROP_KEYS = frozenset({
    "ssn", "memberPhone", "memberState",
    "memberProductCode", "memberRiderCode",
    "basePlanId", "eligibilityFrom", "eligibilityThru",
})

_PRESCRIPTION_DROP_KEYS = frozenset({
    "prescriberQualifier", "prescriberQualifierDescription",
    "pharmacyQualifier", "pharmacyQualifierDescription",
    "rxNumberQualifier", "rxNumberQualifierDescription",
    "productIDQualifierDescription",
    "versionReleaseNumber", "binNumber",
    "processControlNumber", "groupNumber",
    "personCode", "transactionCode",
})

_DRUG_DROP_KEYS = frozenset({
    "productIDQualifier", "productIDQualifierDescription",
    "productSelectionCode", "productSelectionCodeDescription",
    "metricQuantity", "unitOfMeasure",
})

_OVERRIDES_DROP_KEYS = frozenset({
    "paNumber", "paReasonCode", "paLayered",
})

# Entire top-level claim sections to drop unconditionally
_DROP_SECTIONS = frozenset({
    "audit", "additionalDetails", "pricingAdditionalDTO",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_nulls_and_empty(d: Any) -> Any:
    """Recursively remove keys whose value is None, empty string, or empty list."""
    if isinstance(d, dict):
        return {
            k: _strip_nulls_and_empty(v)
            for k, v in d.items()
            if v is not None and v != "" and v != [] and v != {}
        }
    if isinstance(d, list):
        return [_strip_nulls_and_empty(item) for item in d if item is not None]
    return d


def _drop_keys(d: dict, drop: frozenset) -> dict:
    """Return a new dict with specified keys removed."""
    return {k: v for k, v in d.items() if k not in drop}


def _compact_overrides(overrides: Optional[dict]) -> Optional[dict]:
    """Keep only override flags that are 'Yes' or have meaningful values."""
    if not overrides:
        return None
    result = {}
    for k, v in overrides.items():
        if k in _OVERRIDES_DROP_KEYS:
            continue
        # Only keep flags that are 'Yes' or non-None/non-'No'
        if v is not None and v != "No":
            result[k] = v
    return result or None


def _compact_pricing(pricing: Optional[dict]) -> Optional[dict]:
    """Keep only pricing fields that have actual values."""
    if not pricing:
        return None
    result = {}
    for k, v in pricing.items():
        if v is not None:
            result[k] = v
    return result or None


def _compact_messages(messages: Optional[dict]) -> Optional[dict]:
    """Keep only non-null message sections."""
    if not messages:
        return None
    result = {}
    for k, v in messages.items():
        if v is not None and v != []:
            result[k] = v
    return result or None


# ---------------------------------------------------------------------------
# Core trimming
# ---------------------------------------------------------------------------

def trim_single_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trim a single claim dict, removing internal fields, nulls,
    and redundant qualifiers. Returns a new compact dict.
    """
    trimmed = {}

    # --- claimInformation ---
    ci = claim.get("claimInformation", {})
    ci = _drop_keys(ci, _CLAIM_INFO_DROP_KEYS)
    # Drop secondary/primary claim fields if all null
    for prefix in ("secondary", "primary"):
        sub_keys = [k for k in ci if k.startswith(prefix)]
        if all(ci.get(k) is None for k in sub_keys):
            ci = {k: v for k, v in ci.items() if k not in sub_keys}
    trimmed["claimInformation"] = ci

    # --- member (will be deduplicated at the top level later) ---
    member = claim.get("member", {})
    member = _drop_keys(member, _MEMBER_DROP_KEYS)
    trimmed["member"] = member

    # --- drug ---
    drug = claim.get("drug", {})
    drug = _drop_keys(drug, _DRUG_DROP_KEYS)
    trimmed["drug"] = drug

    # --- pricing ---
    trimmed["pricing"] = _compact_pricing(claim.get("pricing"))

    # --- prescription ---
    rx = claim.get("prescription", {})
    rx = _drop_keys(rx, _PRESCRIPTION_DROP_KEYS)
    trimmed["prescription"] = rx

    # --- priorAuthorization ---
    pa = claim.get("priorAuthorization", {})
    trimmed["priorAuthorization"] = pa

    # --- overrides ---
    trimmed["overrides"] = _compact_overrides(claim.get("overrides"))

    # --- messages ---
    trimmed["messages"] = _compact_messages(claim.get("messages"))

    # Drop sections that are always null/empty
    for section in _DROP_SECTIONS:
        trimmed.pop(section, None)

    # Final null/empty sweep
    trimmed = _strip_nulls_and_empty(trimmed)

    return trimmed


def trim_api_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trim the full API response payload.

    Returns a compact dict with structure:
    {
        "totalCount": int,
        "member": { ... },          # extracted once
        "claims": [ ... ],           # trimmed per-claim (member removed from each)
        "summary": { ... }           # optional
    }
    """
    if not response or not isinstance(response, dict):
        return response

    claims_raw = response.get("claims", [])
    if not claims_raw:
        return {
            "totalCount": 0,
            "member": None,
            "claims": [],
        }

    # Extract member from first claim (same across all claims for a member search)
    first_member = claims_raw[0].get("member", {})
    member_info = _drop_keys(first_member, _MEMBER_DROP_KEYS)
    member_info = _strip_nulls_and_empty(member_info)

    # Trim each claim and remove the duplicated member section
    trimmed_claims = []
    for claim in claims_raw:
        tc = trim_single_claim(claim)
        tc.pop("member", None)  # deduplicated to top level
        trimmed_claims.append(tc)

    result = {
        "totalCount": response.get("totalCount", len(trimmed_claims)),
        "member": member_info,
        "claims": trimmed_claims,
    }

    return result


def trim_single_claim_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trim a response that contains a single claim lookup (LIST API by claim number).
    Preserves member inside each claim since it may differ across sequence numbers.
    """
    if not response or not isinstance(response, dict):
        return response

    claims_raw = response.get("claims", [])
    trimmed_claims = [trim_single_claim(c) for c in claims_raw]

    result = {
        "totalCount": response.get("totalCount", len(trimmed_claims)),
        "claims": trimmed_claims,
    }
    return result
