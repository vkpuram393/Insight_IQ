"""Rendering Models — data structures for HTML rendering results."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RenderingResult:
    render_format: str          # "html_table" | "text"
    html_content: str
    css_content: str            # always empty — styling handled by frontend
    answer_header: str
    success: bool
    render_time_ms: float = 0.0
    error: Optional[str] = None
    # Validated Render DSL from the response LLM (layout + field paths).
    # Populated by process_rendering() in routes.py; None when not applicable.
    render_dsl: Optional[Dict[str, Any]] = None
