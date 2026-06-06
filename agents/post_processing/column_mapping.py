"""Column mapping data models — produced by StructureExtractor, stored in ExtractionCache.

A ColumnMapping describes:
  - where the list of records lives in the raw API response (data_path)
  - which fields to display and how to format them (columns)

These are the only structures the LLM produces. The LLM never outputs actual
claim data — only dot-notation paths and format hints.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ColumnDef:
    """Single column definition produced by the LLM."""

    header: str    # Display label, e.g. "Claim #"
    path: str      # Dot-notation path into each record, e.g. "claimInformation.claimNumber"
    format: str    # One of VALID_FORMAT_TYPES: "text" | "date" | "currency" |
                   #   "status_badge" | "reject_codes" | "title"


@dataclass
class ColumnMapping:
    """Full column mapping for one (tool_name, intent) pair."""

    data_path: str           # Dot-notation path to the records list, e.g. "data.claims"
    columns: List[ColumnDef]
    tool_name: str           # API tool name, e.g. "claims_api"
    intent: str              # Resolved intent, e.g. "claim_list"
    created_at: str          # ISO-8601 timestamp — for cache inspection / TTL logic
