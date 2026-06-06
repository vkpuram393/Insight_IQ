"""Schema extractor — strips all values, keeps only field names and type names.

This is the PII firewall between the pharmacy API response and the LLM.
No claim numbers, member IDs, drug names, dates, or dollar amounts are
ever sent to Gemini. Only the *structure* (key names + primitive types) leaves
this process.

Example:
    Input:  {"claimInformation": {"claimNumber": "260173639698000", "patientPay": 50.0}}
    Output: {"claimInformation": {"claimNumber": "string", "patientPay": "number"}}
"""

from typing import Any


def extract_schema(data: Any, max_depth: int = 6) -> Any:
    """Recursively replace leaf values with their type name.

    Args:
        data:      Any JSON-compatible Python value.
        max_depth: Maximum nesting depth before stopping (guards against
                   pathological inputs; returns "any" beyond the limit).

    Returns:
        Structurally identical object with all scalar values replaced by
        one of: "string", "number", "boolean", "null", "any".
    """
    if max_depth <= 0:
        return "any"

    if isinstance(data, dict):
        return {k: extract_schema(v, max_depth - 1) for k, v in data.items()}

    if isinstance(data, list):
        # Only describe the first element — lists are homogeneous in API responses.
        # An empty list produces an empty schema list.
        return [extract_schema(data[0], max_depth - 1)] if data else []

    if isinstance(data, bool):
        return "boolean"

    if isinstance(data, str):
        return "string"

    if isinstance(data, (int, float)):
        return "number"

    return "null"
