"""LLM-based structure extractor — the ONLY file that calls the LLM.

Calls Gemini via generate() to infer a column mapping from a PII-free API
response schema. The two-attempt JSON parsing pattern mirrors response_agent.py:
  1. Strip markdown fences, try json.loads() on the whole string.
  2. Scan '{' positions with json.JSONDecoder().raw_decode() as fallback.

Any failure (bad JSON, < 2 valid columns, LLM down) raises ValueError.
The caller (_extract_with_llm) catches all exceptions and falls back to
the existing shape-detection pipeline.
"""

import json
from datetime import datetime, timezone
from typing import Any, List

from agents.post_processing.column_mapping import ColumnDef, ColumnMapping
from agents.post_processing.path_extractor import get_by_path
from core.logger import get_logger

logger = get_logger(__name__)

# Imported lazily inside methods to avoid circular imports and to allow
# monkeypatching in tests without triggering Vertex AI initialisation.
# from services.llm_connection import generate
# from agents.post_processing.rendering_themes import INTENT_DESCRIPTIONS, VALID_FORMAT_TYPES

_SYSTEM_PROMPT = """You are a data extraction assistant for a pharmacy benefits management system.
Given an API response schema and user intent, return a JSON column mapping.

Available format types:
- text: plain string display
- date: YYYYMMDD or YYYY-MM-DD → displays as MM/DD/YYYY
- currency: dollar string or number → displays as $X.XX
- status_badge: claim status string → colored badge (Paid=green, Denied=red, Pending=yellow)
- reject_codes: denial reason codes → yellow badge
- title: string → Title Case

Rules:
1. data_path: exact dot-notation path to the array of records as it appears in the provided schema (e.g. "data.claims" if the schema shows data.claims, "data.claimDetails" if it shows data.claimDetails) — derive it from the schema, do NOT copy the example
2. Include only fields relevant to the user intent — maximum 8 columns
3. Order columns by relevance to the intent
4. Use human-readable headers, not technical field names
5. For status fields: always prefer the human-readable description field (e.g. "statusDescription", "claimStatusDescription") over raw code fields (e.g. "claimStatus", "status") — description fields contain the full text like "Reversed/Cancelled" that matches what users see in explanations
6. Column paths are relative to each individual record in the array, not to the root response
7. Return ONLY valid JSON — no explanation, no markdown fences

Output format (data_path MUST reflect the actual schema, not this example):
{"data_path": "data.<actual_list_key>", "columns": [
  {"header": "Claim #", "path": "claimInformation.claimNumber", "format": "text"},
  {"header": "Status",  "path": "claimInformation.claimStatusDescription", "format": "status_badge"}
]}"""


class StructureExtractor:
    """Calls the LLM to infer a ColumnMapping from an API response schema."""

    def extract(
        self,
        schema: Any,
        intent: str,
        tool_name: str = "",
    ) -> ColumnMapping:
        """Infer a column mapping by calling the LLM.

        Args:
            schema:    PII-free schema dict (values replaced with type names).
            intent:    Resolved user intent string, e.g. "claim_list".
            tool_name: API tool name (stored in ColumnMapping for cache keying).

        Returns:
            ColumnMapping with at least 2 validated columns.

        Raises:
            ValueError: If the LLM response is unparseable or yields < 2 columns.
            Exception:  Any error from generate() propagates up so the caller
                        can catch and fall back to shape detection.
        """
        # Lazy imports to avoid circular dependencies and Vertex AI startup cost
        # in test environments that never reach this code path.
        from services.llm_connection import generate
        from agents.post_processing.rendering_themes import INTENT_DESCRIPTIONS

        intent_desc = INTENT_DESCRIPTIONS.get(intent, intent.replace("_", " "))
        schema_json = json.dumps(schema, indent=2)

        user_prompt = (
            f"User intent: {intent}\n"
            f"Description: {intent_desc}\n\n"
            f"API response schema (field names and value types only):\n{schema_json}"
        )

        logger.info(
            "structure_extractor: calling LLM tool_name=%s intent=%s schema_keys=%s",
            tool_name,
            intent,
            list(schema.keys()) if isinstance(schema, dict) else "non-dict",
        )

        raw_text = generate(
            prompt=user_prompt,
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=2048,
        )

        logger.debug("structure_extractor: LLM raw response (first 300): %s", raw_text[:300])

        parsed = self._parse_json(raw_text)
        columns = self._build_columns(parsed)

        logger.info(
            "structure_extractor: extracted %d columns for tool=%s intent=%s",
            len(columns),
            tool_name,
            intent,
        )

        return ColumnMapping(
            data_path=parsed.get("data_path", "data"),
            columns=columns,
            tool_name=tool_name,
            intent=intent,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------ #
    # JSON parsing — two-attempt pattern from response_agent.py           #
    # ------------------------------------------------------------------ #

    def _parse_json(self, raw: str) -> dict:
        """Parse LLM output to extract the column mapping JSON.

        Attempt 1: Strip markdown fences, try json.loads() on the whole string.
        Attempt 2: Scan for '{' positions and use raw_decode() to find embedded JSON.

        Raises:
            ValueError: If no parseable JSON with a 'columns' key is found.
        """
        cleaned = raw.strip()

        # Strip markdown code fences the LLM sometimes wraps around JSON.
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Attempt 1 — direct parse of the whole cleaned string.
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "columns" in parsed:
                logger.debug("structure_extractor: JSON parsed via direct json.loads()")
                return parsed
        except json.JSONDecodeError:
            pass

        # Attempt 2 — raw_decode scan.  The LLM sometimes prepends explanation
        # text before the JSON object.  raw_decode() correctly handles nested
        # brackets, escaped quotes, and special characters.
        decoder = json.JSONDecoder()
        scan_start = 0
        while scan_start < len(cleaned):
            brace_pos = cleaned.find("{", scan_start)
            if brace_pos == -1:
                break
            try:
                candidate, _ = decoder.raw_decode(cleaned, brace_pos)
                if isinstance(candidate, dict) and "columns" in candidate:
                    logger.debug(
                        "structure_extractor: JSON found via raw_decode at pos=%d",
                        brace_pos,
                    )
                    return candidate
            except json.JSONDecodeError:
                pass
            scan_start = brace_pos + 1

        raise ValueError(
            f"No parseable JSON with 'columns' key in LLM response: {raw[:120]!r}"
        )

    # ------------------------------------------------------------------ #
    # Column building                                                      #
    # ------------------------------------------------------------------ #

    def _build_columns(self, parsed: dict) -> List[ColumnDef]:
        """Convert raw parsed dict to List[ColumnDef].

        Raises:
            ValueError: If the parsed dict yields < 2 column definitions.
        """
        from agents.post_processing.rendering_themes import VALID_FORMAT_TYPES

        raw_cols = parsed.get("columns", [])
        if not isinstance(raw_cols, list):
            raise ValueError(f"'columns' field is not a list: {type(raw_cols)}")

        cols: List[ColumnDef] = []
        for item in raw_cols:
            if not isinstance(item, dict):
                continue
            fmt = str(item.get("format", "text")).strip()
            if fmt not in VALID_FORMAT_TYPES:
                logger.debug(
                    "structure_extractor: unknown format '%s' — defaulting to 'text'", fmt
                )
                fmt = "text"
            header = str(item.get("header", "")).strip()
            path = str(item.get("path", "")).strip()
            if not header or not path:
                logger.debug("structure_extractor: skipping column with empty header or path")
                continue
            cols.append(ColumnDef(header=header, path=path, format=fmt))

        if len(cols) < 2:
            raise ValueError(
                f"StructureExtractor: only {len(cols)} valid column(s) after parsing — "
                "need at least 2 to render a meaningful table"
            )

        return cols


def validate_columns(
    columns: List[ColumnDef],
    sample_record: dict,
) -> List[ColumnDef]:
    """Filter columns whose paths resolve to non-empty values in the sample record.

    Called by _extract_with_llm() after StructureExtractor.extract() returns,
    before the mapping is written to cache.  Ensures the LLM-provided paths
    actually work against the live API response structure.

    Args:
        columns:       List of ColumnDef from the LLM.
        sample_record: One actual record from the API response (not the schema).

    Returns:
        Filtered list with only columns whose paths resolve to non-empty values
        AND whose format types are valid.

    Raises:
        ValueError: If fewer than 2 columns pass validation.
    """
    from agents.post_processing.rendering_themes import VALID_FORMAT_TYPES

    valid = [
        col
        for col in columns
        if str(get_by_path(sample_record, col.path)).strip()
        and col.format in VALID_FORMAT_TYPES
    ]

    if len(valid) < 2:
        raise ValueError(
            f"validate_columns: only {len(valid)} column path(s) resolve in the sample record — "
            "insufficient for rendering (need at least 2)"
        )

    logger.info(
        "validate_columns: %d/%d columns passed path validation",
        len(valid),
        len(columns),
    )
    return valid
