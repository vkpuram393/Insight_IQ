"""
Overrides_api.llm_formatter

Converts slim PA records into a compact, LLM-ready text string.

Mirrors Claims_search_api/llm_formatter.py:
  1. Slim records from response_trimmer.py (already normalised)
  2. Format each record into a concise multi-line text block
  3. Concatenate with member summary header

The compact text output reduces token usage by ~70-80% compared to
raw JSON, keeping prompts well within context limits even for members
with dozens of PA records.
"""

from typing import Any, Dict, List, Optional

from .response_trimmer import DRUG_TYPE_LABELS


# ---------------------------------------------------------------------------
# Per-record compact formatter
# ---------------------------------------------------------------------------

def _format_pa_compact(record: Dict[str, Any], index: int) -> str:
    """
    Format a single slim PA record into a concise multi-line text block.

    Example output:
        --- PA 1 ---
        PA#: 100000001 | Drug: ACT FLUORIDE SOL 0.05% | Reason: OD
        Drug Type: N (NDC) | NDC: 41167009428
        Effective: 20250101 - 20391231
        Agent: A | Ignore Status: Y | Specialty Rx Override: No
        Modified: 2025-10-31T03:54:22
    """
    lines = [f"--- PA {index} ---"]

    # Core identification
    ref = record.get("paReferenceNumber", "—")
    drug = record.get("drugName", "—")
    reason = record.get("reasonCode", "—")
    lines.append(f"PA#: {ref} | Drug: {drug} | Reason: {reason}")

    # Drug type and identifier
    drug_parts: List[str] = []
    drug_type = record.get("drugTypeIndicator", "")
    drug_label = record.get("drugTypeLabel", "")
    if drug_type:
        label = f" ({drug_label})" if drug_label else ""
        drug_parts.append(f"Drug Type: {drug_type}{label}")
    if record.get("ndc"):
        drug_parts.append(f"NDC: {record['ndc']}")
    if record.get("gpi"):
        drug_parts.append(f"GPI: {record['gpi']}")
    if record.get("drugListId"):
        drug_parts.append(f"List ID: {record['drugListId']}")
    if drug_parts:
        lines.append(" | ".join(drug_parts))

    # Effective period
    eff = record.get("effectiveDate", "—")
    end = record.get("terminationDate", "—")
    lines.append(f"Effective: {eff} - {end}")

    # Agent and status flags
    flags: List[str] = []
    if record.get("agentCode"):
        flags.append(f"Agent: {record['agentCode']}")
    if record.get("ignoreStatus"):
        flags.append(f"Ignore Status: {record['ignoreStatus']}")
    spec_rx = record.get("specialtyRxOverrideIndicator")
    if spec_rx is not None:
        flags.append(f"Specialty Rx Override: {'Yes' if spec_rx else 'No'}")
    if flags:
        lines.append(" | ".join(flags))

    # Additional flags (only when non-empty / non-default)
    extras: List[str] = []
    if record.get("clinicalAdminCode"):
        extras.append(f"Clinical Admin: {record['clinicalAdminCode']}")
    follow_me = record.get("followMeIndicator")
    if follow_me is not None:
        extras.append(f"Follow-Me: {'Yes' if follow_me else 'No'}")
    if record.get("transformCarePlanIndicator"):
        extras.append(f"Transform Care: {record['transformCarePlanIndicator']}")
    if extras:
        lines.append(" | ".join(extras))

    # Modification timestamp
    if record.get("lastModifiedDateTime"):
        lines.append(f"Modified: {record['lastModifiedDateTime']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_pa_records_for_llm(
    slim_records: List[Dict[str, Any]],
    member_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Main entry point: convert slim PA records into a compact text block
    suitable for LLM context injection.

    Mirrors Claims_search_api/llm_formatter.py:format_claims_for_llm().

    Args:
        slim_records:    List of normalised PA records from response_trimmer.
        member_summary:  Optional member summary dict (masked IDs).

    Returns:
        Compact multi-line text string.
    """
    if not slim_records:
        return "No Prior Authorization records available."

    sections: List[str] = []

    # Member header (masked identifiers only)
    if member_summary:
        parts = [f"{k}: {v}" for k, v in member_summary.items() if v]
        if parts:
            sections.append("MEMBER: " + " | ".join(parts))

    sections.append(f"Total PA Records: {len(slim_records)}")
    sections.append("")

    for i, rec in enumerate(slim_records, 1):
        sections.append(_format_pa_compact(rec, i))
        sections.append("")

    return "\n".join(sections)
