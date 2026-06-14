"""
Override Tool Node — LangGraph node for Prior Authorization queries.

Mirrors call_claims_tool_node / call_claims_search_node_v2 in shape:
  1. Resolve claim_id from state
  2. Extract auth headers from user_info
  3. Step 1: fetch member CAGM from existing claim-list API
  4. Step 2: fetch PA overrides from Overrides API
  5. Return {"tool_results": ToolResult(...).dict()}

The downstream response_agent generates the final answer from tool_results
using _get_override_domain_prompt() — no LLM call happens in this node.
"""

import re
import traceback
from typing import Any, Dict

from core.node_models import ToolResult, ToolExecutionStatus
from core.logger import get_logger
from core.logging_context import extract_logging_context, log_state_snapshot
from overrides.override_api_client import get_member_cagm_from_claim, get_overrides_by_member
from overrides.response_trimmer import trim_overrides_response

logger = get_logger(__name__)

_CLAIM_PREFIX = re.compile(r"^\s*claim\s+", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_claim_id(state: Dict[str, Any]) -> str:
    """
    Extract the first usable claim_id from the various state shapes.

    Checks claim_ids (list format used by entity extractor) FIRST, then
    scalar camelCase / snake_case variants, then user_info.claim_id.
    """
    entities = state.get("entities") or {}
    extracted_slots = state.get("extracted_slots") or {}
    user_info = state.get("user_info") or {}

    def _pick(d: dict) -> str:
        # List format: claim_ids (used by entity extractor + confidence router)
        v = d.get("claim_ids")
        if v:
            return str(v[0]).strip() if isinstance(v, list) else str(v).strip()
        # Scalar variants
        for key in ("claimNumber", "claimId", "claim_number", "claim_id"):
            val = d.get(key)
            if val:
                return str(val[0]).strip() if isinstance(val, list) else str(val).strip()
        return ""

    raw = _pick(entities) or _pick(extracted_slots) or (user_info.get("claim_id") or "")
    if not raw:
        return ""

    # Strip common "claim " prefix typos
    cleaned = _CLAIM_PREFIX.sub("", raw).strip()
    return cleaned


def _failure_payload(error: str, *, needs_clarification: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "tool_results": ToolResult(
            tool_name="override_api",
            status=ToolExecutionStatus.FAILURE,
            data={"error": error},
            error_message=error,
            api_endpoint=None,
            http_status_code=None,
            is_retryable=False,
        ).dict()
    }
    if needs_clarification:
        payload["needs_clarification"] = True
        payload["clarification_context"] = {
            "reason": "missing_entity",
            "domain": "override_domain",
            "missing_entities": ["claim_number"],
            "guidance": "Please provide a claim number to look up Prior Authorization records.",
        }
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Main node
# ─────────────────────────────────────────────────────────────────────────────

async def call_override_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: fetch Prior Authorization (PA) override records.

    Reads from state:
        entities / extracted_slots / user_info  →  claim_id
        user_info                               →  auth_token, x_api_key, x_clientrefid

    Writes to state:
        tool_results  →  ToolResult with PA data (domain tag = override_domain)
    """
    logger.info("🔧 Node: Call Override Tool (PA prior-authorization)")

    log_ctx = extract_logging_context(state)
    user_info = state.get("user_info") or {}

    # ------------------------------------------------------------------
    # 1. Resolve claim_id
    # ------------------------------------------------------------------
    claim_id = _coerce_claim_id(state)
    if not claim_id:
        logger.warning("[OverrideNode] No claim_id found — routing to clarification")
        result = _failure_payload(
            "No claim number provided for Prior Authorization lookup.",
            needs_clarification=True,
        )
        await log_state_snapshot(state, "call_override_tool", result)
        return result

    logger.info(f"[OverrideNode] Resolved claim_id={claim_id!r}")

    # ------------------------------------------------------------------
    # 2. Extract auth headers
    # ------------------------------------------------------------------
    bearer_token = user_info.get("auth_token", "")
    x_api_key = user_info.get("x_api_key", "")
    x_clientrefid = user_info.get("x_clientrefid", "")

    if not bearer_token:
        logger.warning("[OverrideNode] No auth token in user_info")
        result = _failure_payload("Missing authorization token.")
        await log_state_snapshot(state, "call_override_tool", result)
        return result

    # ------------------------------------------------------------------
    # 3. Step 1 — resolve member CAGM from the claim-list API
    # ------------------------------------------------------------------
    try:
        cagm = await get_member_cagm_from_claim(
            claim_number=claim_id,
            bearer_token=bearer_token,
            x_api_key=x_api_key,
            x_clientrefid=x_clientrefid,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[OverrideNode] CAGM resolution failed: {exc}\n{tb}")
        result = _failure_payload(f"Failed to resolve member information: {exc}")
        await log_state_snapshot(state, "call_override_tool", result)
        return result

    if not cagm:
        logger.warning(f"[OverrideNode] CAGM not found for claim_id={claim_id!r}")
        result = _failure_payload(
            f"Could not find member information for claim number {claim_id}."
        )
        await log_state_snapshot(state, "call_override_tool", result)
        return result

    # ------------------------------------------------------------------
    # 4. Step 2 — fetch PA override records
    # ------------------------------------------------------------------
    try:
        overrides_response = await get_overrides_by_member(
            cagm=cagm,
            bearer_token=bearer_token,
            x_api_key=x_api_key,
            x_clientrefid=x_clientrefid,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[OverrideNode] Overrides API failed: {exc}\n{tb}")
        result = _failure_payload(f"Failed to retrieve Prior Authorization records: {exc}")
        await log_state_snapshot(state, "call_override_tool", result)
        return result

    raw_count = len((overrides_response or {}).get("priorAuthorizations") or [])
    logger.info(f"[OverrideNode] Retrieved PA response — {raw_count} record(s)")

    # ------------------------------------------------------------------
    # 5. Slim the response to protect LLM context window
    # ------------------------------------------------------------------
    slim_overrides = trim_overrides_response(overrides_response)
    trimmed_count = len((slim_overrides or {}).get("priorAuthorizations") or [])
    if raw_count != trimmed_count:
        logger.info(
            f"[OverrideNode] Response trimmed: {raw_count} → {trimmed_count} records "
            f"(cap applied or {raw_count - trimmed_count} dropped)"
        )

    # ------------------------------------------------------------------
    # 6. Return tool_results (response_agent handles LLM generation)
    # ------------------------------------------------------------------
    tool_result = ToolResult(
        tool_name="override_api",
        status=ToolExecutionStatus.SUCCESS,
        data={
            "is_override_domain": True,
            "claim_id": claim_id,
            "cagm": cagm,
            "overrides": slim_overrides,        # trimmed — safe for LLM
            "overrides_raw_count": raw_count,   # for logging/telemetry only
        },
        api_endpoint=f"{getattr(__import__('config.config', fromlist=['settings']).settings, 'overrides_api_path', '/pss/myclaims/override/exp/v1/priorauth/search')}",
        http_status_code=200,
        is_retryable=False,
    )

    result = {
        "tool_results": tool_result.dict(),
        "metadata": {
            **state.get("metadata", {}),
            "tools_used": ["override_api"],
        },
    }
    await log_state_snapshot(state, "call_override_tool", result)
    return result
