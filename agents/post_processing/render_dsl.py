"""
Render DSL — schema and validation for LLM-driven flexible rendering.

The LLM outputs a compact JSON block between ===RENDER_START=== / ===RENDER_END===
delimiters that describes WHAT to render and HOW to lay it out.  Deterministic
Python code then walks the actual tool_results JSON using dot-notation paths from
the DSL — the LLM never touches real values, so hallucination on financial data
is structurally impossible.

Frontend consumes render_dsl alongside html_content during the transition period,
then can move to pure DSL rendering once the Angular component is ready.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Allowlists — unknown values are silently defaulted, never rejected
# ---------------------------------------------------------------------------

VALID_LAYOUTS = frozenset({
    "table",          # multiple records in a list
    "pivot",          # categories as rows, comparison columns (e.g. pricing breakdown)
    "detail_card",    # single claim with many fields
    "split_panel",    # two side-by-side sections (COB primary vs secondary)
    "alert_card",     # denial / rejection requiring attention
    "metrics_card",   # summary metrics / KPIs
    "progress_card",  # deductible progress bar
    "timeline",       # audit trail / sequence of events
    "composite",      # mix of the above
})

VALID_SECTION_TYPES = frozenset({
    "table", "card", "metrics", "alert", "progress", "timeline_entry",
})

VALID_POSITIONS = frozenset({
    "main", "left", "right", "top", "bottom",
})

VALID_FORMATS = frozenset({
    "text", "date", "currency", "status_badge",
    "reject_codes", "title", "count", "progress",
})

_DEFAULT_LAYOUT  = "table"
_DEFAULT_SECTION = "table"
_DEFAULT_POSITION = "main"
_DEFAULT_FORMAT  = "text"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RenderColumn:
    """One column in a table section."""
    header: str
    field: str                   # flat field name in each record
    format: str = _DEFAULT_FORMAT

    @classmethod
    def from_dict(cls, d: dict) -> "RenderColumn":
        fmt = d.get("format", _DEFAULT_FORMAT)
        if fmt not in VALID_FORMATS:
            fmt = _DEFAULT_FORMAT
        return cls(
            header=str(d.get("header", "")),
            field=str(d.get("field") or d.get("path", "")),
            format=fmt,
        )

    def to_dict(self) -> dict:
        return {"header": self.header, "field": self.field, "format": self.format}


@dataclass
class RenderItem:
    """One label-value row in a card/detail section."""
    label: str
    path: str                    # dot-notation into the record
    format: str = _DEFAULT_FORMAT

    @classmethod
    def from_dict(cls, d: dict) -> "RenderItem":
        fmt = d.get("format", _DEFAULT_FORMAT)
        if fmt not in VALID_FORMATS:
            fmt = _DEFAULT_FORMAT
        return cls(
            label=str(d.get("label", "")),
            path=str(d.get("path", "")),
            format=fmt,
        )

    def to_dict(self) -> dict:
        return {"label": self.label, "path": self.path, "format": self.format}


@dataclass
class RenderPivotField:
    """One field inside a pivot group (e.g. the 'Primary' column)."""
    field: str
    format: str = _DEFAULT_FORMAT

    @classmethod
    def from_dict(cls, d: dict) -> "RenderPivotField":
        fmt = d.get("format", _DEFAULT_FORMAT)
        if fmt not in VALID_FORMATS:
            fmt = _DEFAULT_FORMAT
        return cls(field=str(d.get("field", "")), format=fmt)

    def to_dict(self) -> dict:
        return {"field": self.field, "format": self.format}


@dataclass
class RenderPivotGroup:
    """One row in a pivot layout (e.g. 'Ingredient Cost')."""
    label: str
    fields: Dict[str, RenderPivotField] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "RenderPivotGroup":
        fields = {}
        for col_name, col_def in (d.get("fields") or {}).items():
            if isinstance(col_def, dict):
                fields[col_name] = RenderPivotField.from_dict(col_def)
        return cls(label=str(d.get("label", "")), fields=fields)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
        }


@dataclass
class RenderSection:
    """One layout section — table, card, metrics panel, etc."""
    id: str
    type: str
    title: str = ""
    position: str = _DEFAULT_POSITION
    data_path: str = ""          # dot-notation to the record or list
    is_list: bool = False        # True → data_path points to a list of records
    columns: List[RenderColumn] = field(default_factory=list)
    items: List[RenderItem] = field(default_factory=list)
    groups: List[RenderPivotGroup] = field(default_factory=list)
    identifier_columns: List[RenderColumn] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "RenderSection":
        sec_type = d.get("type", _DEFAULT_SECTION)
        if sec_type not in VALID_SECTION_TYPES:
            sec_type = _DEFAULT_SECTION
        position = d.get("position", _DEFAULT_POSITION)
        if position not in VALID_POSITIONS:
            position = _DEFAULT_POSITION
        return cls(
            id=str(d.get("id", "")),
            type=sec_type,
            title=str(d.get("title", "")),
            position=position,
            data_path=str(d.get("data_path", "")),
            is_list=bool(d.get("is_list", False)),
            columns=[RenderColumn.from_dict(c) for c in (d.get("columns") or [])],
            items=[RenderItem.from_dict(i) for i in (d.get("items") or [])],
            groups=[RenderPivotGroup.from_dict(g) for g in (d.get("groups") or [])],
            identifier_columns=[RenderColumn.from_dict(c) for c in (d.get("identifier_columns") or [])],
        )

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "position": self.position,
            "data_path": self.data_path,
            "is_list": self.is_list,
            "columns": [c.to_dict() for c in self.columns],
            "items": [i.to_dict() for i in self.items],
        }
        if self.groups:
            d["groups"] = [g.to_dict() for g in self.groups]
        if self.identifier_columns:
            d["identifier_columns"] = [c.to_dict() for c in self.identifier_columns]
        return d


@dataclass
class RenderDSL:
    """Top-level render instruction produced by the LLM."""
    layout: str
    title: str
    sections: List[RenderSection]

    @classmethod
    def from_dict(cls, d: dict) -> "RenderDSL":
        layout = d.get("layout", _DEFAULT_LAYOUT)
        if layout not in VALID_LAYOUTS:
            layout = _DEFAULT_LAYOUT
        return cls(
            layout=layout,
            title=str(d.get("title", "")),
            sections=[RenderSection.from_dict(s) for s in (d.get("sections") or [])],
        )

    def to_dict(self) -> dict:
        return {
            "layout": self.layout,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
        }


# ---------------------------------------------------------------------------
# Public validation entry point
# ---------------------------------------------------------------------------

def validate_render_dsl(
    dsl_dict: Dict[str, Any],
) -> Optional[RenderDSL]:
    """
    Parse and validate a raw DSL dict.

    - Unknown layout types are silently defaulted to "table".
    - Unknown format types are silently defaulted to "text".
    - Returns None if the result has no usable sections.

    Designed to be called from routes.py inside a try/except — never raises.
    """
    try:
        dsl = RenderDSL.from_dict(dsl_dict)

        if not dsl.sections:
            return None

        return dsl

    except Exception:
        return None
