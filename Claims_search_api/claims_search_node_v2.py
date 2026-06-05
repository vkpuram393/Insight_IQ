"""
Claims_search_api.claims_search_node_v2

ALTERNATE LangGraph node that uses the **generalized LLM-based** Q&A
(see `llm_query_responder.py`) instead of the deterministic
filter-then-summarize pipeline used by `claims_search_node.py`.

Key differences vs. claims_search_node.py
-----------------------------------------
  • No regex / keyword filtering on the Python side.
  • No call to `generalized_claims_query` from search.py.
  • The full (capped, newest-first) claim list is sent to the LLM,
    along with the user's original natural-language question, and the
    LLM does its own field-level lookup / filtering / summarization.
  • Writes the final user-facing answer directly to `state["response"]`
    AND mirrors the structure into `tool_results` so downstream
    PII / memory / cache nodes still work unchanged.

This node is **not** wired into `langgraph_agent.py` automatically.  To
use it instead of the existing claim-history node, swap the import in
`langgraph_agent.py`:

    # from Claims_search_api.claims_search_node import call_claims_search_node
    from Claims_search_api.claims_search_node_v2 import call_claims_search_node_v2 as call_claims_search_node

Both nodes share the same input/output contract on `tool_results`, so
downstream nodes don't need any changes.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict

from core.logger import get_logger
from Claims_search_api.api_utils import (
    extract_list_api_response_structure,
    extract_member_cagm_from_response,
)
from Claims_search_api.llm_query_responder import (
    prepare_claim_history_data,
    DEFAULT_MAX_CLAIMS,
)
from Claims_search_api.search import generalized_claims_query
from Claims_search_api.response_trimmer import trim_api_response

logger = get_logger(__name__)


def _coerce_claim_id(state: Dict[str, Any]) -> str:
    """Pull the first usable claim_id out of the various state shapes."""
    entities = state.get("entities") or {}
    extracted_slots = state.get("extracted_slots") or {}
    user_info = state.get("user_info") or {}

    candidates = (
        entities.get("claim_ids"),
        entities.get("claim_id"),
        entities.get("claimNumber"),
        entities.get("claim_number"),
        extracted_slots.get("claim_ids"),
        extracted_slots.get("claim_id"),
        extracted_slots.get("claimNumber"),
        extracted_slots.get("claim_number"),
        user_info.get("claim_id"),
    )
    for c in candidates:
        if not c:
            continue
        if isinstance(c, list):
            if c:
                return str(c[0]).strip()
        else:
            return str(c).strip()
    return ""


def _failure_payload(
    error: str,
    *,
    needs_clarification: bool = False,
    missing_entities: list = None,
) -> Dict[str, Any]:
    payload = {
        "tool_results": {
            "tool_name": "claims_search_v2",
            "status": "failure",
            "success": False,
            "error": error,
            "data": {
                "is_claim_history_search": True,
                "claims": [],
                "totalCount": 0,
                "filteredCount": 0,
            },
        },
    }
    if needs_clarification:
        payload["needs_clarification"] = True
        payload["clarification_context"] = {
            "reason": "missing_entity",
            "domain": "claim_history_search",
            "missing_entities": missing_entities or ["claim_number", "date_range"],
            "guidance": (
                "Please provide a claim ID (15 digits) or a date range "
                "to search your claim history."
            ),
        }
    return payload


async def call_claims_search_node_v2(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generalized LLM-driven LangGraph node for claim-history queries.

    Reads from state:
        - text / user_input / metadata.original_text : user question
        - entities / extracted_slots / user_info     : claim_id
        - user_info: auth_token, x_api_key, x_clientrefid

    Writes to state:
        - response       : str        — final user-facing answer
        - tool_results   : dict       — same envelope as v1 for compatibility
    """
    logger.info("🔍 Node: Call Claims Search v2 (LLM-driven, generalized)")

    # ------------------------------------------------------------------
    # 1. Resolve user query + claim_id + auth headers
    # ------------------------------------------------------------------
    metadata = state.get("metadata") or {}
    user_query = (
        metadata.get("original_text")
        or state.get("user_input")
        or state.get("text")
        or ""
    ).strip()

    user_info = state.get("user_info") or {}
    claim_id = _coerce_claim_id(state)

    if not claim_id:
        logger.warning("[ClaimsSearchV2] No claim_id found in state")
        return _failure_payload(
            "No claim ID provided",
            needs_clarification=True,
            missing_entities=["claim_number", "date_range"],
        )

    bearer_token = user_info.get("auth_token", "")
    x_api_key = user_info.get("x_api_key", "")
    x_clientrefid = user_info.get("x_clientrefid", "")

    if not bearer_token:
        logger.warning("[ClaimsSearchV2] No auth token in user_info")
        return _failure_payload("Missing authorization token")

    # ------------------------------------------------------------------
    # 2. Fetch the member's claim history (uses the same 2-step
    #    orchestration as v1 — no auth changes)
    # ------------------------------------------------------------------
    try:
        logger.info(f"[ClaimsSearchV2] Fetching member claim history (claim_id={claim_id})")
        api_response = await extract_list_api_response_structure(
            claim_id=claim_id,
            bearer_token=bearer_token,
            x_api_key=x_api_key,
            x_clientrefid=x_clientrefid,
        )
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[ClaimsSearchV2] API fetch failed: {e}\n{tb}")
        return _failure_payload(f"API error: {e}")

    claims = (api_response or {}).get("claims") or []
    total_count = len(claims)
    logger.info(f"[ClaimsSearchV2] Retrieved {total_count} claim(s) from API")

    if total_count == 0:
        return {
            "tool_results": {
                "tool_name": "claims_search_v2",
                "status": "success",
                "success": True,
                "data": {
                    "is_claim_history_search": True,
                    "claims": [],
                    "totalCount": 0,
                    "filteredCount": 0,
                    "memberInfo": {},
                    "_slim_claims": [],
                    "_member_summary": {},
                },
            },
        }

    # ------------------------------------------------------------------
    # 3. Filter claims by user query (drug, reject code, status, date…)
    # ------------------------------------------------------------------
    filtered_claims = generalized_claims_query(claims, user_query) or claims
    filtered_count = len(filtered_claims)
    logger.info(
        "[ClaimsSearchV2] Filtered %d → %d claim(s) for query: %r",
        total_count, filtered_count, user_query,
    )

    # ------------------------------------------------------------------
    # 4. Trim the filtered response and extract member info
    # ------------------------------------------------------------------
    filtered_response = {**api_response, "claims": filtered_claims, "totalCount": filtered_count}
    trimmed_response = trim_api_response(filtered_response, max_claims=DEFAULT_MAX_CLAIMS)
    member_info = extract_member_cagm_from_response(api_response)

    # ------------------------------------------------------------------
    # 5. Prepare claims data for response_agent (NO Gemini call here —
    #    response_agent owns all final-response LLM calls).
    # ------------------------------------------------------------------
    try:
        prepared = prepare_claim_history_data(trimmed_response, max_claims=DEFAULT_MAX_CLAIMS)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[ClaimsSearchV2] Data preparation failed: {e}\n{tb}")
        return _failure_payload(f"Data preparation error: {e}")

    logger.info(
        "[ClaimsSearchV2] Data prepared: total=%d, used=%d claims for LLM",
        prepared.get("total_claims", 0),
        prepared.get("used_claims", 0),
    )

    # ------------------------------------------------------------------
    # 6. Return tool_results with prepared data — response_agent will
    #    build the prompt and call Gemini.
    # ------------------------------------------------------------------
    return {
        "tool_results": {
            "tool_name": "claims_search_v2",
            "status": "success",
            "success": True,
            "error": "",
            "data": {
                "is_claim_history_search": True,
                "claims": [],
                "totalCount": total_count,
                "filteredCount": filtered_count,
                "memberInfo": member_info,
                "_slim_claims": prepared["slim_claims"],
                "_member_summary": prepared["member_summary"],
            },
        },
    }
