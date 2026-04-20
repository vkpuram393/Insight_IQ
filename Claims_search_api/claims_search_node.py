"""
Claims_search_api.claims_search_node

LangGraph node wrapper that integrates the claims search pipeline
into the existing graph flow.

This node is called instead of call_claims_tool when the intent
is 'claims_search'. It reads state the same way, calls the pipeline,
and writes results back to state in the same format as call_claims_tool.

Flow:
  confidence_checker → build_context → [router] → call_claims_search_node
    → response_safety_pii_precheck → response_agent → ...
"""

import traceback
from typing import Dict, Any

from core.logger import get_logger
from core.logging_context import extract_logging_context
from Claims_search_api.search import generalized_claims_query
from Claims_search_api.llm_formatter import format_claims_for_llm
from Claims_search_api.api_utils import (
    extract_list_api_response_structure,
    extract_member_cagm_from_response,
)
from Claims_search_api.claims_response_agent import ClaimsResponseAgent

logger = get_logger(__name__)

# Singleton agent
_claims_agent = None


def _get_claims_agent() -> ClaimsResponseAgent:
    global _claims_agent
    if _claims_agent is None:
        _claims_agent = ClaimsResponseAgent()
    return _claims_agent


async def call_claims_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: runs the claims search pipeline.

    Reads from state:
        - user_input:  The user's query text
        - entities:    Should contain claim_ids (list) from intent detection
        - user_info:   Contains auth_token, x-api-key, x-clientrefid headers

    Writes to state:
        - tool_results: Dict with the claims search results (same shape as call_claims_tool)
        - response:     The LLM-generated answer (so response_agent can pass through or enhance)
    """
    node_name = "call_claims_search"
    log_ctx = extract_logging_context(state) if isinstance(state, dict) else {}

    logger.info("🔍 Node: Call Claims Search (member history pipeline)")

    try:
        # ------------------------------------------------------------------
        # Extract inputs from state
        # ------------------------------------------------------------------
        user_query = state.get("user_input", "")
        entities = state.get("entities") or {}
        user_info = state.get("user_info") or {}
        extracted_slots = state.get("extracted_slots") or {}

        # Get claim ID from entities (current or extracted from history)
        claim_ids = (
            entities.get("claim_ids")
            or extracted_slots.get("claim_ids")
            or entities.get("claimNumber")
            or []
        )
        if isinstance(claim_ids, str):
            claim_ids = [claim_ids]
        claim_id = claim_ids[0] if claim_ids else None

        if not claim_id:
            logger.warning("[ClaimsSearch] No claim_id found in entities or slots")
            return {
                "tool_results": {
                    "success": False,
                    "error": "No claim ID provided",
                    "data": {},
                },
                "response": (
                    "I'd be happy to help search through the claims. "
                    "Could you please provide a claim number so I can look up the member's history?"
                ),
                "needs_clarification": True,
            }

        # Extract auth headers
        bearer_token = user_info.get("auth_token", "")
        x_api_key = user_info.get("x_api_key", "")
        x_clientrefid = user_info.get("x_clientrefid", "")

        if not bearer_token:
            logger.warning("[ClaimsSearch] No auth token in user_info")
            return {
                "tool_results": {
                    "success": False,
                    "error": "Missing authorization token",
                    "data": {},
                },
                "response": "I'm unable to access the claims system right now. Please ensure you're logged in and try again.",
            }

        # ------------------------------------------------------------------
        # Step 1: Fetch claims from upstream API
        # ------------------------------------------------------------------
        logger.info(f"[ClaimsSearch] Fetching claims for claim_id={claim_id}")
        api_response = await extract_list_api_response_structure(
            claim_id=claim_id,
            bearer_token=bearer_token,
            x_api_key=x_api_key,
            x_clientrefid=x_clientrefid,
        )

        claims = api_response.get("claims", [])
        total_count = len(claims)

        if not claims:
            return {
                "tool_results": {
                    "success": True,
                    "data": {"claims": [], "totalCount": 0},
                },
                "response": (
                    "I wasn't able to find any claims for the provided claim number. "
                    "Please double-check the claim number and try again."
                ),
            }

        logger.info(f"[ClaimsSearch] Fetched {total_count} claims from API")

        # ------------------------------------------------------------------
        # Step 2: Extract member info
        # ------------------------------------------------------------------
        member_info = extract_member_cagm_from_response(api_response)

        # ------------------------------------------------------------------
        # Step 3: Filter claims by user query
        # ------------------------------------------------------------------
        filtered_claims = generalized_claims_query(claims, user_query)
        filtered_count = len(filtered_claims)
        logger.info(
            f"[ClaimsSearch] Filtered {total_count} → {filtered_count} claims "
            f"for query: {user_query!r}"
        )

        # ------------------------------------------------------------------
        # Step 4: Format for LLM
        # ------------------------------------------------------------------
        filtered_response = {
            **api_response,
            "claims": filtered_claims,
            "totalCount": filtered_count,
        }
        llm_context = format_claims_for_llm(
            filtered_response,
            user_query=None,  # Already filtered
            is_member_history=True,
        )

        logger.info(f"[ClaimsSearch] LLM context: {len(llm_context)} chars")

        # ------------------------------------------------------------------
        # Step 5: Generate response via LLM
        # ------------------------------------------------------------------
        agent = _get_claims_agent()
        llm_response = agent.generate_response(
            user_query=user_query,
            claims_context=llm_context,
            member_info=member_info,
            total_claims=total_count,
            filtered_claims=filtered_count,
        )

        # ------------------------------------------------------------------
        # Write results to state
        # ------------------------------------------------------------------
        return {
            "tool_results": {
                "success": True,
                "data": {
                    "claims": filtered_claims,
                    "totalCount": total_count,
                    "filteredCount": filtered_count,
                    "memberInfo": member_info,
                },
            },
            "response": llm_response,
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[ClaimsSearch] Node error: {e}\n{tb}")
        return {
            "tool_results": {
                "success": False,
                "error": str(e),
                "data": {},
            },
            "response": (
                "I encountered an issue while searching the claims. "
                "Please try again shortly."
            ),
        }
