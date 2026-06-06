"""
MyclaimsRenderingAgent — post-graph HTML renderer.

Called from api/routes.py after run_graph() completes.
Never raises. All user data embedded in HTML is escaped with html.escape().

Extraction is driven entirely by render_dsl from the response agent — the LLM
already chose columns, paths, and formats tailored to the user's question.
Supports table (flat), pivot (categories-as-rows), and multi-section layouts.

DOMPurify in the Angular integration allows onclick via ADD_ATTR: ['onclick'],
so the fullscreen button uses requestFullscreen() exactly like the POC.

CSS uses mc-poc- prefix.  Digital Pulse var(--ps-*) tokens with hardcoded
fallbacks so it renders correctly with or without the design-token library.
"""

import html
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents.post_processing.rendering_models import RenderingResult
from agents.post_processing.rendering_themes import (
    STATUS_CODE_MAP, TABLE_TITLES, CSS_SCOPE, NO_RENDER_INTENTS,
    DEFAULT_TABLE_TITLE, MUST_RENDER_INTENTS,
    VALID_RENDER_MODES, VALID_FORMAT_TYPES,
)
from core.logger import get_logger

logger = get_logger(__name__)

_MAX_ROWS = 500

_ASCII_TABLE_RE = re.compile(r'^\s*\|.+\|\s*$', re.MULTILINE)


def _has_ascii_table(text: str) -> bool:
    """Return True if text contains 2+ pipe-delimited table rows."""
    return len(_ASCII_TABLE_RE.findall(text)) >= 2


_HEADER_CONTEXT_MAP = {
    "prescriber": "prescriber",
    "pharmacy":   "pharmacy",
    "member":     "beneficiary",
    "beneficiary":"beneficiary",
    "drug":       "drug",
    "benefit":    "benefit",
    "linked":     "linkedclaim",
}

_BLOCKED_FIELDS = frozenset({
    "description43Name",   # field does not exist in any API response
})

# Field aliases the LLM commonly writes that differ from the real flat API key names.
# Applied in _extract_section() BEFORE the blocked-fields filter so columns that
# would otherwise be empty are silently corrected and render with real data.
_FIELD_REMAP: Dict[str, str] = {
    # Fill-date aliases — LLM carries "date2" from response-text data paths into DSL
    "date2":                    "dateOfFill",
    "fillDate2":                "dateOfFill",
    "filldate2":                "dateOfFill",
    "fillDate":                 "dateOfFill",
    # DOB alias — "date8" appears in response text as the DOB path alias
    "date8":                    "dateOfBirth",
    # Reject-code aliases — LLM invents "code"/"rejectCode"; real key is below
    "code":                     "responseRejectCode",
    "rejectCode":               "responseRejectCode",
    "reject_code":              "responseRejectCode",
    "rejectionCode":            "responseRejectCode",
    # Status aliases — LLM writes claimStatusDescription or claimStatus instead of statusDescription
    "status":                   "statusDescription",
    "claimStatus":              "statusDescription",
    "claimStatusDescription":   "statusDescription",
    "approvalStatus":           "statusDescription",
    "adjudicationStatus":       "statusDescription",
    # Drug-name aliases
    "drugName":                 "drugLabelName",
    "productName":              "drugLabelName",
    "medicationName":           "drugLabelName",
    "drug":                     "drugLabelName",
    # Member/beneficiary name aliases
    "memberName":               "lastNameFirstName",
    "beneficiaryName":          "lastNameFirstName",
    "patientName":              "lastNameFirstName",
    "name":                     "lastNameFirstName",
    # DOB aliases
    "dob":                      "dateOfBirth",
    "birthDate":                "dateOfBirth",
    "dateOfBirth":              "dateOfBirth",   # already correct — no-op safety
    # Member ID aliases
    "id":                       "memberId",
    "memberID":                 "memberId",
    "cardholderID":             "memberId",
    # NDC aliases
    "ndc":                      "submittedProductId",
    "ndcNumber":                "submittedProductId",
    # Common dot-path fragments that survive the dot-stripping step
    "message":                  "settlementMessage",
    # Claim number alias (some LLM outputs)
    "claim":                    "claimNumber",
    "claimId":                  "claimNumber",
}

_RE_DATE_8DIGIT = re.compile(r"^\d{8}$")
_RE_DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")


class MyclaimsRenderingAgent:

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def execute(
        self,
        intent: str,
        tool_results: Dict[str, Any],
        entities: Dict[str, Any],
        render_dsl: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        response_text: Optional[str] = None,
    ) -> RenderingResult:
        t0 = time.perf_counter()
        try:
            result = self._execute(intent, tool_results, entities, render_dsl, render_mode, response_text)
        except Exception as exc:
            logger.warning("rendering agent exception: %s", exc, exc_info=True)
            result = self._fallback(error=str(exc))
        result.render_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        return result

    def _execute(self, intent, tool_results, entities, render_dsl=None, render_mode=None, response_text=None) -> RenderingResult:
        # Gate 1: Intent that never produces renderable data
        if intent in NO_RENDER_INTENTS:
            logger.debug("intent=%s in NO_RENDER_INTENTS — skipping", intent)
            return self._fallback(error=f"intent '{intent}' in NO_RENDER_INTENTS")

        # Gate 2: Tool call failed — nothing to render
        status = tool_results.get("status", "")
        if getattr(status, "value", status).lower() != "success":
            logger.debug("intent=%s tool status=%s — skipping", intent, status)
            return self._fallback(error="tool status != success")

        tool_name = str(tool_results.get("tool_name") or "unknown")

        # ASCII safety net: if LLM wrote a pipe-table in response but render_dsl is present → override
        if render_mode == "text_only" and response_text and render_dsl:
            if _has_ascii_table(response_text):
                logger.info(
                    "rendering: ASCII table detected in response_text — "
                    "overriding render_mode to table intent=%s", intent
                )
                render_mode = "table"

        # Gate 3: Skip rendering? (text_only signal + tier safety nets)
        if self._should_skip_rendering(render_mode, render_dsl, intent):
            logger.info(
                "rendering: text_only — intent=%s render_mode=%s",
                intent, render_mode,
            )
            return self._fallback(error=f"render_mode=text_only intent={intent}")

        # Extract rows via render_dsl from response agent
        section_results: List[Tuple[List[dict], List, str]] = []
        if render_dsl:
            layout = render_dsl.get("layout", "table")
            if layout == "pivot":
                section_results = self._extract_pivot(tool_results, render_dsl)
            else:
                section_results = self._extract_from_dsl(tool_results, render_dsl)
                if not section_results:
                    has_groups = any(
                        s.get("groups") and not s.get("columns")
                        for s in (render_dsl.get("sections") or [])
                    )
                    if has_groups:
                        logger.info("rendering: table empty but sections have groups — trying pivot")
                        section_results = self._extract_pivot(tool_results, render_dsl)
            if section_results:
                logger.info(
                    "rendering: extraction succeeded tool=%s intent=%s sections=%d rows=%d",
                    tool_name, intent, len(section_results),
                    sum(len(r) for r, _, _ in section_results),
                )
            else:
                logger.warning(
                    "rendering: extraction returned 0 sections tool=%s intent=%s",
                    tool_name, intent,
                )
        else:
            logger.warning(
                "rendering: no render_dsl — LLM omitted render block tool=%s intent=%s",
                tool_name, intent,
            )

        if not section_results:
            return self._fallback(error="no data extracted from tool results")

        # Python decides text/table from actual data shape — never relies on LLM for this
        effective_mode = self._pick_visual_mode(section_results, intent)
        logger.info(
            "rendering: effective_mode=%s intent=%s tool=%s",
            effective_mode, intent, tool_name,
        )

        if effective_mode == "text":
            return self._fallback(error="single row — promoting to text")

        if len(section_results) == 1:
            rows, visible, _ = section_results[0]
            total_count = len(rows)
            truncated = total_count > _MAX_ROWS
            return self._render(
                rows[:_MAX_ROWS], intent,
                truncated=truncated, total_count=total_count, visible=visible,
            )

        return self._render_multi(section_results, intent)

    # ------------------------------------------------------------------ #
    # Tier 1 — render_dsl from response agent                              #
    # ------------------------------------------------------------------ #

    def _extract_from_dsl(
        self,
        tool_results: dict,
        render_dsl: dict,
    ) -> List[Tuple[List[dict], List, str]]:
        """Extract rows from ALL sections in the render_dsl.

        Returns a list of (rows, visible, section_title) tuples — one per
        section that produced data.  Returns [] on total failure.
        """
        try:
            sections = render_dsl.get("sections") or []
            if not sections:
                return []

            results: List[Tuple[List[dict], List, str]] = []
            for section in sections:
                if not isinstance(section, dict):
                    continue
                rows, visible = self._extract_section(tool_results, section)
                if rows and visible:
                    title = section.get("title", "")
                    results.append((rows, visible, title))

            return results

        except Exception as exc:
            logger.error(
                "rendering: _extract_from_dsl failed — %s: %s",
                type(exc).__name__, exc, exc_info=True,
            )
            return []

    def _extract_section(
        self,
        tool_results: dict,
        section: dict,
    ) -> Tuple[List[dict], Optional[List]]:
        """Extract rows from a single DSL section.

        Returns (rows, visible) on success or ([], None) on failure.
        """
        # Ensure columns is a list of dicts — guard against LLM returning strings/nulls
        raw_cols = section.get("columns") or []
        columns = [c for c in raw_cols if isinstance(c, dict)]
        if not columns:
            items = section.get("items") or []
            columns = [
                {"header": i.get("label", ""), "field": i.get("field", i.get("path", "")),
                 "format": i.get("format", "text")}
                for i in items
                if isinstance(i, dict)
            ]
        if not columns:
            return [], None

        # Sanitize format type — unknown values silently become "text"
        for col in columns:
            if col.get("format") not in VALID_FORMAT_TYPES:
                col["format"] = "text"

        # Step 1 — strip dot-notation FIRST so remapping uses the leaf key name
        for col in columns:
            raw = col.get("field") or col.get("path", "")
            if raw and "." in raw:
                col["field"] = raw.rsplit(".", 1)[-1]

        # Step 2 — remap known LLM alias field names to real API key names
        for col in columns:
            raw = col.get("field") or col.get("path", "")
            corrected = _FIELD_REMAP.get(raw)
            if corrected:
                col["field"] = corrected
                logger.debug(
                    "_extract_section: remapped field %r → %r (header=%r)",
                    raw, corrected, col.get("header", ""),
                )

        # Step 3 — remove genuinely useless fields after remapping
        columns = [
            c for c in columns
            if (c.get("field") or c.get("path", "")) not in _BLOCKED_FIELDS
        ]
        if not columns:
            return [], None

        # Step 4 — deduplicate by field name; keep the first occurrence of each field
        seen_fields: set = set()
        deduped: list = []
        for col in columns:
            f = col.get("field") or col.get("path", "")
            if f and f not in seen_fields:
                seen_fields.add(f)
                deduped.append(col)
        columns = deduped
        if not columns:
            return [], None

        field_names = {
            col.get("field") or col.get("path", "")
            for col in columns
        } - {""}
        if not field_names:
            return [], None

        records = self._find_records(tool_results, field_names)
        if not records:
            return [], None

        rows: List[dict] = []
        for record in records:
            row: Dict[str, Any] = {}
            for col in columns:
                header = col.get("header", "")
                field = col.get("field") or col.get("path", "")
                if header and field:
                    hint = self._context_hint_from_header(header)
                    value = self._find_field_value(record, field, context_hint=hint)
                    if value is None:
                        # Fallback: field may exist outside the selected sub-record
                        # (e.g., claim-level fields when record is a DUR sub-object).
                        value = self._find_field_value(tool_results, field, context_hint=hint)
                    row[header] = value
            if row:
                rows.append(row)

        if not rows:
            return [], None

        visible: List[Tuple[str, str, str]] = [
            (col.get("header", ""), col.get("header", ""),
             col.get("format", "text"))
            for col in columns
            if col.get("header")
        ]

        non_empty = [
            (key, label, fmt) for key, label, fmt in visible
            if any(str(r.get(key, "")).strip() for r in rows)
        ]
        if not non_empty:
            return [], None

        if len(non_empty) < len(visible):
            keep = {key for key, _, _ in non_empty}
            rows = [{k: v for k, v in r.items() if k in keep} for r in rows]

        return rows, non_empty

    # ------------------------------------------------------------------ #
    # Pivot layout extraction                                               #
    # ------------------------------------------------------------------ #

    def _extract_pivot(
        self,
        tool_results: dict,
        render_dsl: dict,
    ) -> List[Tuple[List[dict], List, str]]:
        """Extract rows for a pivot layout where categories become rows.

        The DSL provides ``identifier_columns`` (e.g. Claim Number) and
        ``groups`` — each group becomes one row with a "Component" label
        and one value column per field key (e.g. Primary / Secondary / Final).

        Returns the same ``List[Tuple[rows, visible, title]]`` format as
        ``_extract_from_dsl()`` so the rest of the pipeline is unchanged.
        """
        try:
            sections = render_dsl.get("sections") or []
            if not sections:
                return []

            section = sections[0]
            groups = section.get("groups") or []
            if not groups:
                return []

            id_cols = section.get("identifier_columns") or []

            # Collect every field referenced in groups so we can find records
            all_fields: set = set()
            for g in groups:
                for pf in (g.get("fields") or {}).values():
                    f = pf.get("field", "") if isinstance(pf, dict) else ""
                    if f:
                        all_fields.add(f)
            for ic in id_cols:
                f = ic.get("field", "")
                if f:
                    all_fields.add(f)

            if not all_fields:
                return []

            records = self._find_records(tool_results, all_fields)
            if not records:
                return []

            # Determine value columns from group field keys (e.g. Primary, Secondary, Final)
            value_col_names: List[str] = []
            seen: set = set()
            for g in groups:
                for col_name in (g.get("fields") or {}):
                    if col_name not in seen:
                        value_col_names.append(col_name)
                        seen.add(col_name)

            # Build header row: identifier cols + "Component" + value columns
            visible: List[Tuple[str, str, str]] = []
            for ic in id_cols:
                h = ic.get("header", "")
                fmt = ic.get("format", "text")
                if h:
                    visible.append((h, h, fmt))

            visible.append(("Component", "Component", "text"))
            for vc in value_col_names:
                visible.append((vc, vc, "currency"))

            # Build rows — one row per group per record
            rows: List[dict] = []
            for record in records:
                # Extract identifier values once per record
                id_vals: Dict[str, Any] = {}
                for ic in id_cols:
                    h = ic.get("header", "")
                    f = ic.get("field", "")
                    if h and f:
                        id_vals[h] = self._find_field_value(record, f)

                for g in groups:
                    row: Dict[str, Any] = {}
                    row.update(id_vals)
                    row["Component"] = g.get("label", "")

                    for col_name, pf in (g.get("fields") or {}).items():
                        f = pf.get("field", "") if isinstance(pf, dict) else ""
                        fmt = pf.get("format", "currency") if isinstance(pf, dict) else "currency"
                        if f:
                            row[col_name] = self._find_field_value(record, f)
                        else:
                            row[col_name] = None

                    rows.append(row)

            if not rows:
                return []

            # Update visible formats from actual group data
            fmt_map: Dict[str, str] = {}
            for g in groups:
                for col_name, pf in (g.get("fields") or {}).items():
                    if isinstance(pf, dict) and col_name not in fmt_map:
                        fmt_map[col_name] = pf.get("format", "currency")

            visible = [
                (key, label, fmt_map.get(key, fmt))
                for key, label, fmt in visible
            ]

            title = section.get("title", "")
            return [(rows, visible, title)]

        except Exception as exc:
            logger.error(
                "rendering: _extract_pivot failed — %s: %s",
                type(exc).__name__, exc, exc_info=True,
            )
            return []

    # ------------------------------------------------------------------ #
    # Dynamic record / field discovery                                      #
    # ------------------------------------------------------------------ #

    def _find_records(
        self, tool_results: Any, field_names: set,
    ) -> List[dict]:
        """Recursively scan tool_results to find the records containing the
        requested field names.  Returns a list of record dicts (may be
        length 1 for single-claim responses).  No hardcoded paths."""
        if not isinstance(tool_results, dict) or not field_names:
            return []
        candidates: List[Tuple[int, int, str, Any]] = []

        def _scan(obj: Any, depth: int = 0) -> None:
            if isinstance(obj, dict):
                match_count = sum(
                    1 for f in field_names
                    if self._find_field_value(
                        obj, f, traverse_lists=False,
                    ) is not None
                )
                if match_count > 0:
                    candidates.append((match_count, depth, "dict", obj))
                for v in obj.values():
                    _scan(v, depth + 1)
            elif isinstance(obj, list) and obj:
                first_dict = next(
                    (item for item in obj if isinstance(item, dict)), None,
                )
                if first_dict is not None:
                    match_count = sum(
                        1 for f in field_names
                        if self._find_field_value(
                            first_dict, f, traverse_lists=False,
                        ) is not None
                    )
                    if match_count > 0:
                        candidates.append((match_count, depth, "list", obj))

        _scan(tool_results)

        if not candidates:
            return []

        # Pick the candidate with the most field matches; on tie prefer deeper
        # (more specific) containers.
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        best = candidates[0]

        if best[2] == "dict":
            return [best[3]]
        return [item for item in best[3] if isinstance(item, dict)]

    def _context_hint_from_header(self, header: str) -> str:
        """Extract a sub-object keyword from the column header for disambiguation."""
        lower = header.lower()
        for keyword, subkey in _HEADER_CONTEXT_MAP.items():
            if keyword in lower:
                return subkey
        return ""

    @staticmethod
    def _find_hinted_container(record: dict, hint: str) -> Optional[dict]:
        """DFS for the first sub-object whose key contains *hint*."""
        for key, val in record.items():
            if hint in key.lower() and isinstance(val, dict):
                return val
        for val in record.values():
            if isinstance(val, dict):
                found = MyclaimsRenderingAgent._find_hinted_container(val, hint)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _find_field_value(
        record: Any,
        field_name: str,
        context_hint: str = "",
        traverse_lists: bool = True,
    ) -> Any:
        """Recursively search *record* for *field_name*.

        When *context_hint* is provided (e.g. ``"prescriber"``), the search
        first looks inside a sub-object whose key contains that hint word.
        This prevents ``lastName`` from matching the member when the column
        header says "Prescriber Last Name".

        When *traverse_lists* is True (default), also searches inside list
        items so fields inside arrays (e.g. reject codes in
        ``settlementCodesDetail``) are reachable.  Set to False during
        container scoring in ``_find_records()`` to avoid inflating match
        counts for parent containers.
        """
        # Guard: must be a non-empty dict with a non-empty field name
        if not isinstance(record, dict) or not record or not field_name:
            return None

        # Phase 1: context-biased search — find hinted sub-object at any depth
        if context_hint:
            hinted = MyclaimsRenderingAgent._find_hinted_container(
                record, context_hint,
            )
            if hinted is not None:
                result = MyclaimsRenderingAgent._find_field_value(
                    hinted, field_name, traverse_lists=traverse_lists,
                )
                if result is not None:
                    return result

        # Phase 2: normal depth-first recursive search
        if field_name in record:
            val = record[field_name]
            if not isinstance(val, dict):
                return val
        for v in record.values():
            if isinstance(v, dict):
                result = MyclaimsRenderingAgent._find_field_value(
                    v, field_name, traverse_lists=traverse_lists,
                )
                if result is not None:
                    return result
            elif traverse_lists and isinstance(v, list) and v:
                hits = []
                for item in v:
                    if isinstance(item, dict):
                        result = MyclaimsRenderingAgent._find_field_value(
                            item, field_name, traverse_lists=True,
                        )
                        if result is not None:
                            hits.append(str(result))
                if hits:
                    unique_hits = list(dict.fromkeys(hits))  # deduplicate, preserve order
                    capped = unique_hits[:5]                  # cap to prevent cell overflow
                    return capped[0] if len(capped) == 1 else " | ".join(capped)
        return None


    # ------------------------------------------------------------------ #
    # Main renderer                                                         #
    # ------------------------------------------------------------------ #

    def _resolve_visible(
        self, rows: List[dict], visible: Optional[List],
    ) -> List[Tuple[str, str, str]]:
        """Resolve and filter column visibility list."""
        if visible is None:
            all_keys = list(dict.fromkeys(k for row in rows for k in row))
            return [
                (k, k.replace("_", " ").title(), "text") for k in all_keys
                if any(str(r.get(k, "")).strip() for r in rows)
            ]
        filtered = [
            (key, label, fmt) for key, label, fmt in visible
            if any(str(r.get(key, "")).strip() for r in rows)
        ]
        if filtered:
            return filtered
        all_keys = list(dict.fromkeys(k for row in rows for k in row))
        return [
            (k, k.replace("_", " ").title(), "text") for k in all_keys
            if any(str(r.get(k, "")).strip() for r in rows)
        ]


    def _render(
        self,
        rows: List[dict],
        intent: str,
        *,
        truncated: bool = False,
        total_count: Optional[int] = None,
        visible: Optional[List] = None,
        section_title: str = "",
    ) -> RenderingResult:
        p = CSS_SCOPE
        visible = self._resolve_visible(rows, visible)

        context_card = self._context_card(rows, intent, visible, p)
        table_html = self._table_html(rows, visible, intent, p, title_override=section_title)
        truncate_note = (
            f'<div class="{p}-truncation-notice">'
            f'Showing first {_MAX_ROWS} of {total_count or len(rows)} records. '
            f'Refine your search to narrow results.'
            f'</div>'
        ) if truncated else ""

        full_html = (
            f'<div class="{p}-enhanced-business-container">\n'
            f'{context_card}\n'
            f'{truncate_note}'
            f'{table_html}\n'
            f'</div>'
        )

        title = section_title or TABLE_TITLES.get(intent, DEFAULT_TABLE_TITLE)
        n = len(rows)
        header = f"{html.escape(title)} — {n} record{'s' if n != 1 else ''}"

        logger.info("render complete intent=%s rows=%d columns=%d", intent, n, len(visible))
        return RenderingResult(
            render_format="html_table",
            html_content=full_html,
            css_content="",
            answer_header=header,
            success=True,
        )

    def _render_multi(
        self,
        section_results: List[Tuple[List[dict], List, str]],
        intent: str,
    ) -> RenderingResult:
        """Render multiple DSL sections as stacked tables in one container."""
        p = CSS_SCOPE

        all_rows = [r for rows, _, _ in section_results for r in rows]
        first_visible = section_results[0][1]
        context_card = self._context_card(all_rows, intent, first_visible, p)

        tables_html = ""
        for rows, visible, section_title in section_results:
            visible = self._resolve_visible(rows, visible)
            tables_html += self._table_html(
                rows, visible, intent, p, title_override=section_title,
            )

        full_html = (
            f'<div class="{p}-enhanced-business-container">\n'
            f'{context_card}\n'
            f'{tables_html}\n'
            f'</div>'
        )

        title = TABLE_TITLES.get(intent, DEFAULT_TABLE_TITLE)
        n = len(all_rows)
        header = f"{html.escape(title)} — {n} record{'s' if n != 1 else ''}"

        logger.info(
            "render_multi complete intent=%s sections=%d total_rows=%d",
            intent, len(section_results), n,
        )
        return RenderingResult(
            render_format="html_table",
            html_content=full_html,
            css_content="",
            answer_header=header,
            success=True,
        )

    # ------------------------------------------------------------------ #
    # Section 1 — Data Preview Card (compact view)                         #
    # ------------------------------------------------------------------ #

    def _context_card(
        self,
        rows: List[dict],
        intent: str,
        visible: List[Tuple[str, str, str]],
        p: str,
    ) -> str:
        """Build the compact data preview card shown above the table.

        Shows actual field values from the first row (up to 4 columns),
        formatted via _cell() so dates/currency/status badges render correctly.
        Shows a record count and "... and N more" footer for multi-row results.
        """
        total = len(rows)
        field_rows_html = ""

        if rows and visible:
            first_row = rows[0]
            for key, label, fmt in visible[:4]:
                raw_value = first_row.get(key)
                formatted = self._cell(raw_value, p, fmt)
                field_rows_html += (
                    f'<div class="{p}-metric-row">'
                    f'<span class="{p}-metric-label">{html.escape(label)}</span>'
                    f'<span class="{p}-metric-value">{formatted}</span>'
                    f'</div>'
                )
            if total > 1:
                remaining = total - 1
                field_rows_html += (
                    f'<div class="{p}-metric-row">'
                    f'<span class="{p}-metric-label {p}-metric-more">'
                    f'... and {remaining} more record{"s" if remaining != 1 else ""}'
                    f'</span>'
                    f'</div>'
                )

        return (
            f'<div class="{p}-business-context-card">'
            f'<div class="{p}-context-header">'
            f'<span class="{p}-context-title">Data Preview</span>'
            f'<span class="{p}-record-count">'
            f'{total} record{"s" if total != 1 else ""}'
            f'</span>'
            f'</div>'
            f'<div class="{p}-context-body">{field_rows_html}</div>'
            f'</div>'
        )

    # ------------------------------------------------------------------ #
    # Section 2 — Table                                                    #
    # ------------------------------------------------------------------ #

    def _table_html(
        self,
        rows: List[dict],
        visible: List[Tuple[str, str, str]],
        intent: str,
        p: str,
        title_override: str = "",
    ) -> str:
        count = len(rows)
        title = title_override or TABLE_TITLES.get(intent, DEFAULT_TABLE_TITLE)

        header_cells = "".join(
            f'<th class="{p}-cvs-header">{html.escape(label)}</th>'
            for _, label, _ in visible
        )

        body_rows = "".join(
            f'<tr class="{p}-cvs-row">'
            + "".join(
                f'<td class="{p}-cvs-cell">{self._cell(row.get(key, ""), p, fmt)}</td>'
                for key, _, fmt in visible
            )
            + "</tr>"
            for row in rows
        )

        return (
            f'<div class="{p}-enhanced-table-container">'
            f'<div class="{p}-table-header">'
            f'<h2>{html.escape(title)}</h2>'
            f'<div class="{p}-table-metrics"><span>{count} record{"s" if count != 1 else ""}</span></div>'
            f'</div>'
            f'<div class="{p}-table-scroll-container">'
            f'<table class="{p}-cvs-health-table {p}-enhanced">'
            f'<thead><tr>{header_cells}</tr></thead>'
            f'<tbody>{body_rows}</tbody>'
            f'</table></div></div>'
        )

    # ------------------------------------------------------------------ #
    # Cell rendering                                                        #
    # ------------------------------------------------------------------ #

    def _cell(self, value: Any, p: str, format_type: str) -> str:
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value if v is not None and str(v).strip())

        if value is None or str(value).strip() == "":
            return f'<span class="{p}-missing-value">—</span>'

        s = str(value).strip()

        if format_type == "currency":
            return html.escape(self._fmt_currency(s) or s)

        if format_type == "date":
            return html.escape(self._fmt_date(s) or s)

        if format_type == "title":
            return html.escape(s.title())

        if format_type == "status_badge":
            resolved = STATUS_CODE_MAP.get(s.upper(), s)
            status_cls = resolved.lower().replace("/", "-").replace(" ", "-")
            return (
                f'<span class="{p}-status-badge {p}-status-{status_cls}">'
                f'{html.escape(resolved.upper())}</span>'
            )

        if format_type == "reject_codes" and s:
            return f'<span class="{p}-reject-badge">{html.escape(s)}</span>'

        return html.escape(s)

    # ------------------------------------------------------------------ #
    # Formatting helpers                                                    #
    # ------------------------------------------------------------------ #

    def _fmt_date(self, s: str) -> str:
        if not s:
            return ""
        s = str(s).strip()
        if _RE_DATE_8DIGIT.match(s):
            s = f"{s[:4]}-{s[4:6]}-{s[6:]}"
        if _RE_DATE_ISO.match(s):
            try:
                return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%m/%d/%Y")
            except ValueError:
                pass
        return s

    def _fmt_currency(self, val: Any) -> str:
        if val is None:
            return ""
        raw = str(val).strip()
        if not raw:
            return ""
        s = raw.lstrip("0") or "0"
        try:
            amount = float(s)
            return f"-${abs(amount):,.2f}" if amount < 0 else f"${amount:,.2f}"
        except (ValueError, TypeError):
            return raw

    # ------------------------------------------------------------------ #
    # No-data response (API succeeded but returned zero rows)              #
    # ------------------------------------------------------------------ #

    def _render_no_data(self) -> RenderingResult:
        """Return a friendly 'no data' HTML message — success=True, no table rows."""
        p = CSS_SCOPE
        html_content = (
            f'<div class="{p}-enhanced-business-container">'
            f'<div class="{p}-no-data-notice">'
            f'<span class="{p}-no-data-icon">&#9432;</span>'
            f'<span>No claims data found for this request.</span>'
            f'</div>'
            f'</div>'
        )
        return RenderingResult(
            render_format="html_table",
            html_content=html_content,
            css_content="",
            answer_header="",
            success=True,
        )

    # ------------------------------------------------------------------ #
    # Technical fallback (intent/status gating failure — not a data miss)  #
    # ------------------------------------------------------------------ #

    def _fallback(self, error: Optional[str] = None) -> RenderingResult:
        logger.warning("rendering fallback error=%s", error)
        return RenderingResult(
            render_format="text",
            html_content="",
            css_content="",
            answer_header="",
            success=False,
            error=error,
        )

    # ------------------------------------------------------------------ #
    # Render-mode resolution                                               #
    # ------------------------------------------------------------------ #

    def _should_skip_rendering(
        self,
        render_mode: Optional[str],
        render_dsl: Optional[dict],
        intent: str,
    ) -> bool:
        """Return True when rendering should be skipped (text-only response).

        Three tiers:
          Escape hatch  — LLM signalled data genuinely absent via suppress_table.
          Tier 1        — MUST_RENDER intent: always table unless escape hatch fires.
          Tier 2        — LLM decides: trust render_mode.
        """
        # Escape hatch checked FIRST — overrides even MUST_RENDER when data is genuinely absent
        if render_dsl and render_dsl.get("suppress_table"):
            logger.info("rendering: suppress_table escape hatch — text only intent=%s", intent)
            return True

        if intent in MUST_RENDER_INTENTS:
            return False        # Tier 1: always table, never suppress

        if render_mode == "text_only":
            return True         # Tier 2: LLM chose text-only

        if render_mode is None and not render_dsl:
            logger.info(
                "rendering: render_mode=None and no render_dsl — defaulting to text intent=%s",
                intent,
            )
            return True

        return False

    def _pick_visual_mode(
        self,
        section_results: List[Tuple[List[dict], List, str]],
        intent: str,
    ) -> str:
        """Decide text/table from actual extracted data shape — fully deterministic.

        Priority:
          1. No sections / no rows     → table (safe default; _execute guards fallback)
          2. Multiple sections         → always table
          3. Multiple rows             → always table
          4. Single row                → always text (LLM prose is sufficient)

        The MULTIPLE-ROWS RULE applies universally, including MUST_RENDER intents.
        MUST_RENDER's role is in _should_skip_rendering() — it ensures the pipeline
        runs even when LLM says text_only. Final format is always row-count driven.
        Card format has been removed — outputs are text or html_table only.
        """
        if not section_results:
            return "table"

        if len(section_results) > 1:
            return "table"

        rows, visible, _ = section_results[0]

        if not rows:
            return "table"

        if len(rows) > 1:
            return "table"

        # Single row → always text regardless of column count.
        # Multiple rows are required for a table to add value over prose.
        return "text"

    # CSS is handled by the frontend — no styles are generated here.

    @staticmethod
    def _css(p: str) -> str:
        return ""
