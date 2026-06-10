"""
Claims_search_api.llm_formatter

Converts a large claims API response into a compact, LLM-ready string.

Pipeline:
  1. Trim the raw API response  (response_trimmer)
  2. Optionally pre-filter claims by user query  (search.py)
  3. Format the result into a concise text block the LLM can reason over

The output is small enough to fit comfortably within LLM context limits
even for member-history queries returning dozens of claims.
"""
import json
from typing import Any, Dict, List, Optional

from Claims_search_api.response_trimmer import trim_api_response, trim_single_claim_response
from Claims_search_api.filter_extractor import extract_filter_spec, apply_filter_spec
from Claims_search_api.search import generalized_claims_query


# ---------------------------------------------------------------------------
# Query-aware pre-filtering
# ---------------------------------------------------------------------------

def prefilter_claims_by_query(
    claims: List[Dict[str, Any]],
    user_query: str,
) -> List[Dict[str, Any]]:
    """
    Narrow down claims to those relevant to the user query.

    Strategy (LLM-first, regex fallback):
      1. extract_filter_spec() asks the LLM to parse the query into a
         structured FilterSpec (~200 prompt tokens, temperature=0).
         The 70 K-line payload is never sent to this LLM call.
      2. apply_filter_spec() applies every filter deterministically in
         Python using AND logic — all non-None fields must match.
      3. If the LLM call fails or returns no results, fall back to the
         regex-based generalized_claims_query() so nothing is lost.
    """
    if not user_query or not claims:
        return claims

    try:
        spec = extract_filter_spec(user_query)
        if not spec.is_empty():
            filtered = apply_filter_spec(claims, spec)
            if filtered:
                return filtered
    except Exception:
        pass  # extraction error — fall through to regex fallback

    # Regex fallback: preserves existing behaviour for edge cases
    filtered = generalized_claims_query(claims, user_query)
    filtered = [c for c in (filtered or []) if c is not None]
    return filtered if filtered else claims


# ---------------------------------------------------------------------------
# Compact JSON formatter
# ---------------------------------------------------------------------------

def _format_claim_compact(claim: Dict[str, Any], index: int) -> str:
    """
    Format a single trimmed claim into a concise multi-line text block.
    """
    ci = claim.get("claimInformation", {})
    drug = claim.get("drug", {})
    pricing = claim.get("pricing", {})
    rx = claim.get("prescription", {})
    pa = claim.get("priorAuthorization", {})
    overrides = claim.get("overrides", {})
    messages = claim.get("messages", {})

    lines = [f"--- Claim {index} ---"]

    # Core claim info
    lines.append(
        f"Claim#: {ci.get('claimNumber', 'N/A')} | "
        f"Seq: {ci.get('claimSequenceNumber', 'N/A')} | "
        f"Status: {ci.get('claimStatusDescription', ci.get('claimStatus', 'N/A'))} | "
        f"Fill Date: {ci.get('fillDate', 'N/A')}"
    )

    # Additional claim info
    extras = []
    if ci.get("quantity"):
        extras.append(f"Qty: {ci['quantity']}")
    if ci.get("daysSupplied"):
        extras.append(f"Days: {ci['daysSupplied']}")
    if ci.get("claimType"):
        extras.append(f"Type: {ci['claimType']}")
    if ci.get("coverageType"):
        extras.append(f"Coverage: {ci['coverageType']}")
    if ci.get("cobIndicator"):
        extras.append(f"COB: {ci['cobIndicator']}")
    if ci.get("compound") and ci["compound"] != "N":
        extras.append(f"Compound: {ci['compound']}")
    if ci.get("speciality") and ci["speciality"] != "N":
        extras.append(f"Specialty: {ci['speciality']}")
    if ci.get("dispenseAsWritten") and ci["dispenseAsWritten"] != "0":
        extras.append(f"DAW: {ci['dispenseAsWritten']}")
    if ci.get("originationFlagDescription"):
        extras.append(f"Origin: {ci['originationFlagDescription']}")
    if ci.get("secondaryClaimNumber"):
        extras.append(f"Secondary: {ci['secondaryClaimNumber']}")
    if extras:
        lines.append(" | ".join(extras))

    # Drug
    drug_parts = []
    if drug.get("productName"):
        drug_parts.append(f"Drug: {drug['productName']}")
    if drug.get("productNdc"):
        drug_parts.append(f"NDC: {drug['productNdc']}")
    if drug.get("genericIndicator"):
        drug_parts.append(f"Generic: {drug['genericIndicator']}")
    if drug.get("manufacturer"):
        drug_parts.append(f"Mfr: {drug['manufacturer']}")
    if drug.get("gpi"):
        drug_parts.append(f"GPI: {drug['gpi']}")
    if drug.get("multiSourceIndicatorDescription"):
        drug_parts.append(f"Source: {drug['multiSourceIndicatorDescription']}")
    if drug_parts:
        lines.append(" | ".join(drug_parts))

    # Pricing (only non-null)
    if pricing:
        price_parts = [f"{k}: ${v}" for k, v in pricing.items() if v is not None]
        if price_parts:
            lines.append("Pricing: " + " | ".join(price_parts))

    # Prescription
    rx_parts = []
    if rx.get("prescriberFirstName") or rx.get("prescriberLastName"):
        rx_parts.append(
            f"Prescriber: {rx.get('prescriberFirstName', '')} {rx.get('prescriberLastName', '')}"
        )
    if rx.get("prescriberID"):
        rx_parts.append(f"NPI: {rx['prescriberID']}")
    if rx.get("refillNumber"):
        rx_parts.append(f"Refill#: {rx['refillNumber']}")
    if rx.get("rxNumber"):
        rx_parts.append(f"Rx#: {rx['rxNumber']}")
    if rx.get("submitDate"):
        rx_parts.append(f"Submit: {rx['submitDate']}")
    if rx_parts:
        lines.append(" | ".join(rx_parts))

    # Pharmacy
    pharm_parts = []
    if rx.get("pharmacyName"):
        pharm_parts.append(f"Pharmacy: {rx['pharmacyName']}")
    if rx.get("pharmacyCity"):
        pharm_parts.append(f"{rx.get('pharmacyCity', '')}, {rx.get('pharmacyState', '')}")
    if rx.get("pharmacyPhone"):
        pharm_parts.append(f"Ph: {rx['pharmacyPhone']}")
    if rx.get("pharmacyNcpdp"):
        pharm_parts.append(f"NCPDP: {rx['pharmacyNcpdp']}")
    if pharm_parts:
        lines.append(" | ".join(pharm_parts))

    # Diagnosis
    if rx.get("diagnosisCodeQualifier") or rx.get("submittedDiagnosisCodeIndicator"):
        diag_parts = []
        if rx.get("diagnosisCodeQualifier"):
            diag_parts.append(f"Diag: {rx['diagnosisCodeQualifier']}")
        if rx.get("submittedDiagnosisCodeIndicator"):
            diag_parts.append(f"Code: {rx['submittedDiagnosisCodeIndicator']}")
        lines.append(" | ".join(diag_parts))

    # Reversal
    if rx.get("reversalDate"):
        lines.append(f"Reversal Date: {rx['reversalDate']}")

    # Prior Authorization
    pa_parts = []
    if pa.get("paIndicator"):
        pa_parts.append(f"PA Indicator: {pa['paIndicator']}")
    if pa.get("number") and pa["number"] != ci.get("claimNumber"):
        pa_parts.append(f"PA#: {pa['number']}")
    if pa_parts:
        lines.append(" | ".join(pa_parts))

    # Overrides (only 'Yes' flags)
    if overrides:
        override_parts = [f"{k}: {v}" for k, v in overrides.items()]
        if override_parts:
            lines.append("Overrides: " + " | ".join(override_parts))

    # Messages - reject codes, settlement codes, messages
    if messages:
        if messages.get("rejectCodes"):
            reject_strs = [
                f"{rc.get('code', '?')}"
                + (f" ({rc['description']})" if rc.get("description") else "")
                for rc in messages["rejectCodes"]
            ]
            lines.append(f"Reject Codes: {', '.join(reject_strs)}")
        if messages.get("settlementCodes"):
            settle_strs = [
                f"{sc.get('code', '?')}"
                + (f" ({sc['description']})" if sc.get("description") else "")
                for sc in messages["settlementCodes"]
            ]
            lines.append(f"Settlement: {', '.join(settle_strs)}")
        if messages.get("messages"):
            lines.append(f"Messages: {'; '.join(messages['messages'])}")
        if messages.get("approvedMessages"):
            appr_strs = [
                f"{am.get('code', '?')}"
                + (f" ({am['description']})" if am.get("description") else "")
                for am in messages["approvedMessages"]
            ]
            lines.append(f"Approved: {', '.join(appr_strs)}")

    return "\n".join(lines)


def _format_member_header(member: Optional[Dict[str, Any]]) -> str:
    """Format the deduplicated member info as a header block."""
    if not member:
        return "Member: Unknown"

    parts = []
    if member.get("memberId"):
        parts.append(f"Member ID: {member['memberId']}")
    if member.get("firstName") or member.get("lastName"):
        parts.append(f"Name: {member.get('firstName', '')} {member.get('lastName', '')}")
    if member.get("dateOfBirth"):
        parts.append(f"DOB: {member['dateOfBirth']}")
    if member.get("genderDescription"):
        parts.append(f"Gender: {member['genderDescription']}")
    if member.get("age"):
        parts.append(f"Age: {member['age']}")
    if member.get("relationshipDescription"):
        parts.append(f"Relationship: {member['relationshipDescription']}")
    if member.get("carrierId"):
        parts.append(f"Carrier: {member['carrierId']}")
    if member.get("accountId"):
        parts.append(f"Account: {member['accountId']}")
    if member.get("groupId"):
        parts.append(f"Group: {member['groupId']}")
    if member.get("clientPlanCode"):
        parts.append(f"Plan: {member['clientPlanCode']}")
    if member.get("cardholderId"):
        parts.append(f"Cardholder: {member['cardholderId']}")
    if member.get("clientId"):
        parts.append(f"Client: {member['clientId']}")

    return "MEMBER: " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_claims_for_llm(
    api_response: Dict[str, Any],
    user_query: Optional[str] = None,
    *,
    is_member_history: bool = True,
    max_claims: int = 50,
) -> str:
    """
    Main entry point: convert a raw claims API response into a compact
    text block ready to be injected into an LLM prompt.

    Args:
        api_response:     Raw dict from the claims search / member history API.
        user_query:       The user's original question (used for pre-filtering).
        is_member_history: True if the response is a member-level history
                          (member extracted once). False for single claim lookup.
        max_claims:       Safety cap on how many claims to include.

    Returns:
        A compact multi-line string suitable for LLM context.
    """
    if not api_response or not api_response.get("claims"):
        return "No claims data available."

    # Step 1: Trim
    if is_member_history:
        trimmed = trim_api_response(api_response, max_claims=max_claims)
    else:
        trimmed = trim_single_claim_response(api_response)

    # Step 2: Pre-filter by user query
    raw_claims = api_response.get("claims", [])
    if user_query and is_member_history:
        filtered_raw = prefilter_claims_by_query(raw_claims, user_query)
        # Re-trim only the filtered claims
        filtered_response = {**api_response, "claims": filtered_raw}
        trimmed = trim_api_response(filtered_response, max_claims=max_claims)

    claims = trimmed.get("claims", [])[:max_claims]
    total = trimmed.get("totalCount", len(claims))

    # Step 3: Build text output
    sections = []

    # Header
    if is_member_history:
        sections.append(_format_member_header(trimmed.get("member")))
    sections.append(f"Total Claims: {total} | Showing: {len(claims)}")
    sections.append("")

    # Each claim
    for i, claim in enumerate(claims, 1):
        sections.append(_format_claim_compact(claim, i))
        sections.append("")

    return "\n".join(sections)


def format_claims_as_compact_json(
    api_response: Dict[str, Any],
    user_query: Optional[str] = None,
    *,
    is_member_history: bool = True,
    max_claims: int = 50,
) -> str:
    """
    Alternative formatter that returns compact JSON instead of text.
    Useful when the downstream LLM works better with structured data.
    """
    if not api_response or not api_response.get("claims"):
        return json.dumps({"error": "No claims data available."})

    if is_member_history:
        trimmed = trim_api_response(api_response, max_claims=max_claims)
    else:
        trimmed = trim_single_claim_response(api_response)

    raw_claims = api_response.get("claims", [])
    if user_query and is_member_history:
        filtered_raw = prefilter_claims_by_query(raw_claims, user_query)
        filtered_response = {**api_response, "claims": filtered_raw}
        trimmed = trim_api_response(filtered_response, max_claims=max_claims)

    trimmed["claims"] = trimmed.get("claims", [])[:max_claims]
    trimmed["totalCount"] = len(trimmed["claims"])

    return json.dumps(trimmed, indent=None, separators=(",", ":"))
