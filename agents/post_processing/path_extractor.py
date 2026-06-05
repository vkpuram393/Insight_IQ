"""Deterministic dot-notation path extraction.

The LLM produces column definitions like:
    {"header": "Claim #", "path": "claimInformation.claimNumber", "format": "text"}

This module traverses the raw API JSON using those paths to extract actual
values. The LLM never sees or outputs the values themselves — zero
hallucination risk on financial data.

Row dict keys are the column *header* (display label). Using the header
rather than the last path segment avoids key collisions when two columns
share a final segment (e.g. primary.patientPay vs secondary.patientPay).
Headers are guaranteed unique because the LLM produces distinct labels.
"""

from typing import Any, Dict, List

from agents.post_processing.column_mapping import ColumnMapping


def get_by_path(obj: Any, path: str) -> Any:
    """Traverse a dot-notation path through nested dicts/lists.

    Returns '' (empty string) on any miss — never raises.

    List handling: if a path segment is resolved to a list, the first
    element is used to continue traversal. This covers API responses like
    data.claims[0].claimInformation.claimNumber.
    """
    if not path:
        return ""

    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        elif isinstance(obj, list):
            if obj and isinstance(obj[0], dict):
                obj = obj[0].get(key)
            else:
                return ""
        else:
            return ""

        if obj is None:
            return ""

    return obj if obj is not None else ""


def extract_rows(tool_results: dict, mapping: ColumnMapping) -> List[Dict[str, Any]]:
    """Build flat row dicts from raw tool_results using LLM-provided column paths.

    Dict keys are the column header (display label) — unique by design,
    which prevents collisions when two columns share a last path segment.

    Args:
        tool_results: Raw API response dict (including "data", "status", etc.)
        mapping:      ColumnMapping produced by StructureExtractor.

    Returns:
        List of flat row dicts, one per API record.  Empty list if the
        data_path resolves to nothing or the records list is empty.
    """
    records = get_by_path(tool_results, mapping.data_path)

    if not isinstance(records, list):
        records = [records] if records else []

    rows: List[Dict[str, Any]] = []
    for record in records:
        if not record or not isinstance(record, dict):
            continue

        row: Dict[str, Any] = {}
        for col in mapping.columns:
            row[col.header] = get_by_path(record, col.path)

        rows.append(row)

    return rows
