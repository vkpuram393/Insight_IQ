"""
Overrides_api.llm_query_responder — PA-domain LLM prompt builder + answer generator.

Mirrors the structure of Claims_search_api/llm_query_responder.py:
  - _SYSTEM_INSTRUCTIONS  : PA domain knowledge + behavioral rules
  - _RENDER_DSL_CONTRACT  : JSON envelope + ===RENDER_START===/===RENDER_END===
  - _USER_TEMPLATE        : runtime placeholders (date, query, member, records)

Public:
  - build_override_prompt(user_query, slim_pa_records, member_summary, rendering_disabled) -> str
  - prepare_overrides_data(api_response, max_records) -> dict   # trim + sort + cap
  - answer_overrides_query(api_response, user_query, ...) -> dict   # full E2E with Gemini
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from .response_trimmer import (
    trim_overrides_response,
    extract_member_summary_from_cagm,
)

logger = logging.getLogger(__name__)


DEFAULT_MAX_PA_RECORDS = 25


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — System instructions (PA domain knowledge + behavioral rules)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTIONS = """\
You are a pharmacy benefits assistant for CVS Health. You answer member and
agent questions about Prior Authorization (PA) records retrieved from the CVS
override system.

DOMAIN KNOWLEDGE
================
- PA records represent approved, rejected, pending, or cancelled drug coverage
  exceptions for a member.
- A member can have multiple PAs across different drugs and time periods.
- paReferenceNumber uniquely identifies a PA record.
- effectiveDate / terminationDate define the PA's validity window.
- An "Approved" PA means the drug is covered under the override terms.
- reasonCode identifies the reason for the PA override:
    OD = Override Drug, OM = Override Member, PA = Prior Authorization,
    MB = Medical Benefit, HS = Health Solution, PN = Plan,
    2A / 2B / 2C / 2M = Step Therapy levels.
- drugTypeIndicator shows how the drug is identified:
    N = NDC (National Drug Code), G = GPI (Generic Product Identifier),
    M = Manual NDC entry, 1 = Drug List, 2 = GPI-based List.
  drugTypeLabel provides the human-readable label for the indicator.
- drugName is the human-readable drug description.
- ndc / gpi / drugListId is the resolved drug identifier based on
  drugTypeIndicator.
- authorizedDrugNumber is the raw drug identifier value from the API.
- agentCode is a pharmacist or benefit-agent identifier — never expose it as a
  human name.
- ignoreStatus="Y" means adjudication is currently bypassing this PA.
- followMeIndicator=true means the PA applies even after plan / group transfers.
- specialtyRxOverrideIndicator=true means the PA covers a specialty drug.
- clinicalAdminCode requires pharmacist review — do not interpret its meaning.
- transformCarePlanIndicator relates to transform care plan configuration.
- lastModifiedDateTime is the timestamp of the most recent modification.

BEHAVIORAL RULES
================
1. Never recommend specific drugs or clinical alternatives.
2. Never interpret clinicalAdminCode as a clinical decision; state that it
   requires pharmacist review.
3. Direct any question about drug alternatives to a licensed pharmacist.
4. If multiple PA records are present, list them all; do NOT filter arbitrarily.
5. Always include the PA reference number when citing a specific record.
6. State clearly if a PA is expired (current date outside the
   effectiveDate / terminationDate window).
7. Treat copayAmount and pricingInfo as estimates — actual cost may differ.
8. If the data contains a "_fallback_note" field, acknowledge that the data
   may not reflect the member's actual PA records and advise contacting
   member services.

PII / MASKED TOKEN RULES
========================
- If the data contains masked tokens (e.g. [MASKED_xxx]), reproduce them
  exactly. Never attempt to reconstruct the underlying value.
- Never include member names, dates of birth, or full IDs in the response.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Render DSL contract (matches Claims_search_api format)
# ─────────────────────────────────────────────────────────────────────────────

_RENDER_DSL_CONTRACT = """\
================================================================
OUTPUT CONTRACT — STRICT JSON ENVELOPE + OPTIONAL RENDER DSL
================================================================

You MUST respond with a single JSON object FOLLOWED OPTIONALLY by a render
block. No other text is allowed before, between, or after these two parts.

JSON envelope (REQUIRED, render_mode MUST be the FIRST key):

{
    "render_mode": "<text_only | table>",
    "response": "Your complete prose answer about the PA records.",
    "recommendations": [
        {"text": "Short follow-up suggestion 1", "action": "pa_summary"},
        {"text": "Short follow-up suggestion 2", "action": "pa_reason_code"}
    ]
}

UNIVERSAL ROW-COUNT RULE:
  • Will the response data form 2+ DISTINCT PA RECORDS?
        YES → render_mode = "table"   (also append the RENDER block)
        NO  → render_mode = "text_only" (no RENDER block)

  Examples that MUST be table:
    • List of all PA records for a member
    • Comparing multiple PA records (e.g. by reason code, drug, dates)
    • Summary of all overrides

  Examples that MUST be text_only:
    • Single PA record detail or field explanation
    • No-match or no-records answers
    • Explaining a specific field value (e.g. what does reason code OD mean)

NEVER write ASCII or markdown table syntax (`|`, `---`) inside the
"response" field. Tables MUST go through the render_mode = "table" +
===RENDER_START=== mechanism. The response text remains plain prose.

RENDER STRUCTURE BLOCK (REQUIRED only when render_mode = "table"):

Append on a NEW line, with no other text between the JSON and this block:

===RENDER_START===
{"layout":"table","title":"Prior Authorization Records","sections":[{"id":"pa_table","type":"table","columns":[{"header":"Human Label","field":"slimPaKey","format":"<type>"}]}]}
===RENDER_END===

Each column object: {"header":"Display Label","field":"slim_pa_key","format":"text|date"}

PA-DOMAIN AUTHORITATIVE FIELD NAMES (use these exact keys for the
"field" attribute — they are the slim-PA shape produced by
prepare_overrides_data; do NOT invent names):

  Identification           : paReferenceNumber, incidentId
  Drug                     : drugName, drugTypeIndicator, drugTypeLabel,
                             authorizedDrugNumber, ndc, gpi, drugListId
  Reason                   : reasonCode
  Dates                    : effectiveDate (format: date),
                             terminationDate (format: date),
                             lastModifiedDateTime (format: date)
  Agent & Status           : agentCode, ignoreStatus
  Flags                    : specialtyRxOverrideIndicator,
                             clinicalAdminCode, followMeIndicator,
                             transformCarePlanIndicator

COLUMN SELECTION GUIDANCE:
  • Maximum 8 columns; first 4 are the most informative.
  • Order by relevance to the user's question.
  • Headers are human-readable labels (e.g. "PA Reference #",
    "Drug", "Reason Code", "Effective Date"). Do NOT expose raw
    field names as headers.
  • Default column set when no specific field is asked about:
    "PA Reference #", "Drug", "Reason Code", "Effective Date",
    "End Date", "Agent", "Last Modified"

DATA-UNAVAILABLE EXCEPTION:
  If you decided render_mode = "table" but the data genuinely does not
  contain expected fields, set render_mode = "text_only" and answer in
  prose. Do NOT emit an empty render block.

INVALID OUTPUTS:
  ✗ render_mode = "table" with NO render block following the JSON
  ✗ render_mode = "table" for a single-record answer
  ✗ ASCII / markdown table syntax inside the "response" string
  ✗ Markdown code fences anywhere in the output
  ✗ Any text before the JSON or after the RENDER block
"""


_DISABLED_RENDERING_OVERRIDE = """\
========================================================================
RUNTIME OVERRIDE — RENDERING DISABLED (HIGHEST PRIORITY)
========================================================================
Rendering is disabled for this session.
- Set "render_mode" to "text_only" in the JSON envelope.
- Do NOT emit ===RENDER_START===/===RENDER_END=== blocks.
- Present all data as plain prose or a simple text list.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — User template
# ─────────────────────────────────────────────────────────────────────────────

_USER_TEMPLATE = """\
CURRENT DATE: {current_date}

USER QUESTION:
{user_query}

MEMBER SUMMARY (masked):
{member_summary_json}

PRIOR AUTHORIZATION RECORDS ({record_count} record{plural}):
{pa_records_json}

Answer the user's question based ONLY on the PA records above. If the records
are empty, say so plainly: "No Prior Authorization records were found for this
claim." Follow the behavioral rules and the response format contract.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API — prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_override_prompt(
    user_query: str,
    slim_pa_records: List[Dict[str, Any]],
    member_summary: Dict[str, Any],
    rendering_disabled: bool = False,
) -> str:
    """
    Build the full LLM prompt for a PA-domain query.

    Concatenates:  system instructions + render DSL contract + (optional)
                   rendering-disabled override + user template.
    """
    sections = [_SYSTEM_INSTRUCTIONS, _RENDER_DSL_CONTRACT]
    if rendering_disabled:
        sections.append(_DISABLED_RENDERING_OVERRIDE)

    record_count = len(slim_pa_records)
    user = _USER_TEMPLATE.format(
        current_date=datetime.now().strftime("%Y-%m-%d"),
        user_query=user_query.strip(),
        member_summary_json=json.dumps(member_summary or {}, indent=2),
        record_count=record_count,
        plural="" if record_count == 1 else "s",
        pa_records_json=json.dumps(slim_pa_records, indent=2),
    )
    sections.append(user)
    return "\n\n".join(sections)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — data preparation (trim + sort + cap)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_overrides_data(
    api_response: Dict[str, Any],
    *,
    cagm: Optional[Dict[str, Any]] = None,
    max_records: int = DEFAULT_MAX_PA_RECORDS,
) -> Dict[str, Any]:
    """
    Trim raw Overrides API response → slim records + member summary.

    Returns:
        {
            "slim_pa_records": [...],
            "member_summary":  {...},
            "total_count":     int,
            "filtered_count":  int,
            "_fallback_note":  Optional[str],
        }
    """
    raw_records = (api_response or {}).get("priorAuthorizations") or []
    slim = trim_overrides_response(api_response, max_records=max_records)
    member_summary = extract_member_summary_from_cagm(cagm or {})
    return {
        "slim_pa_records":  slim,
        "member_summary":   member_summary,
        "total_count":      len(raw_records) if isinstance(raw_records, list) else 0,
        "filtered_count":   len(slim),
        "_fallback_note":   (api_response or {}).get("_fallback_note"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API — full E2E call (LLM included)
# ─────────────────────────────────────────────────────────────────────────────

def format_overrides_text_fallback(
    slim_pa_records: List[Dict[str, Any]],
    member_summary: Dict[str, Any],
) -> str:
    """
    Pure-Python deterministic answer used when the LLM call fails.

    Provides a usable response with no LLM dependency. Latency ~1ms.
    """
    if not slim_pa_records:
        envelope = {"render_mode": "text_only",
                    "summary": "No Prior Authorization records found."}
        return (json.dumps(envelope) +
                "\n\nNo Prior Authorization records were found for this claim. "
                "If you expected to see records, please contact member services.")

    lines = [f"Found {len(slim_pa_records)} Prior Authorization record(s):"]
    for i, rec in enumerate(slim_pa_records, 1):
        ref     = rec.get("paReferenceNumber", "—")
        drug    = rec.get("drugName", "—")
        reason  = rec.get("reasonCode", "—")
        eff     = rec.get("effectiveDate", "—")
        end     = rec.get("terminationDate", "—")
        agent   = rec.get("agentCode", "")
        agent_s = f" | Agent: {agent}" if agent else ""
        lines.append(f"  {i}. PA #{ref}: {drug} | Reason: {reason} | "
                     f"Effective: {eff} - {end}{agent_s}")
    body = "\n".join(lines)
    envelope = {"render_mode": "text_only",
                "summary": f"Found {len(slim_pa_records)} PA record(s)."}
    return json.dumps(envelope) + "\n\n" + body


def answer_overrides_query(
    api_response: Dict[str, Any],
    user_query: str,
    *,
    cagm: Optional[Dict[str, Any]] = None,
    max_records: int = DEFAULT_MAX_PA_RECORDS,
    rendering_disabled: bool = False,
    temperature: float = 0.2,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level convenience wrapper:
        prepare data → build prompt → call Gemini → return structured answer.

    On LLM failure, returns the deterministic text fallback.

    Returns:
        {
            "answer": str,
            "source": "llm" | "text_fallback",
            "metadata": {...},
        }
    """
    prep = prepare_overrides_data(api_response, cagm=cagm, max_records=max_records)
    slim = prep["slim_pa_records"]
    member_summary = prep["member_summary"]

    prompt = build_override_prompt(
        user_query=user_query,
        slim_pa_records=slim,
        member_summary=member_summary,
        rendering_disabled=rendering_disabled,
    )

    # Lazy import — keep load time low and avoid import-time failures in tests.
    try:
        from services.llm_connection import generate
    except Exception as exc:
        logger.error("[OverridesLLM] services.llm_connection.generate import failed: %s", exc)
        return {
            "answer": format_overrides_text_fallback(slim, member_summary),
            "source": "text_fallback",
            "metadata": {"error": "llm_import_failed", "filtered_count": prep["filtered_count"]},
        }

    gen_kwargs: Dict[str, Any] = {"temperature": temperature}
    if model:
        gen_kwargs["model"] = model

    try:
        answer = generate(prompt, **gen_kwargs)
        return {
            "answer": answer,
            "source": "llm",
            "metadata": {
                "filtered_count":  prep["filtered_count"],
                "total_count":     prep["total_count"],
                "_fallback_note":  prep["_fallback_note"],
            },
        }
    except Exception as exc:
        logger.error("[OverridesLLM] generate() failed: %s\n%s", exc, traceback.format_exc())
        return {
            "answer": format_overrides_text_fallback(slim, member_summary),
            "source": "text_fallback",
            "metadata": {
                "error": f"{type(exc).__name__}: {exc}",
                "filtered_count": prep["filtered_count"],
                "total_count":    prep["total_count"],
            },
        }
