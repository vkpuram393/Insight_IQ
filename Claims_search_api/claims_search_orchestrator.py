"""
Claims_search_api.claims_search_orchestrator

End-to-end orchestrator that:
  1. Accepts a user query + auth context
  2. Fetches claims from the upstream API (by claim_id or member CAGM)
  3. Filters claims based on the user's natural-language query
  4. Formats filtered claims into a compact LLM-ready context
  5. Calls the claims response agent to generate an answer

This module is the single entry-point for the claims-search domain.
It mirrors the existing app flow (intent → context → tool → response)
but is self-contained under Claims_search_api/.
"""

import traceback
from typing import Any, Dict, Optional
from core.logger import get_logger

from Claims_search_api.api_utils import (
    extract_list_api_response_structure,
    extract_member_cagm_from_response,
)
from Claims_search_api.search import generalized_claims_query
from Claims_search_api.llm_formatter import format_claims_for_llm
from Claims_search_api.claims_response_agent import ClaimsResponseAgent

logger = get_logger(__name__)

# Singleton response agent (mirrors ResponseAgent singleton pattern)
_response_agent: Optional[ClaimsResponseAgent] = None


def _get_response_agent() -> ClaimsResponseAgent:
    global _response_agent
    if _response_agent is None:
        _response_agent = ClaimsResponseAgent()
    return _response_agent


async def run_claims_search_pipeline(
    user_query: str,
    claim_id: str,
    bearer_token: str,
    x_api_key: str,
    x_clientrefid: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: query → API fetch → filter → format → LLM → response.

    Args:
        user_query:      Natural language question from the user.
        claim_id:        Claim ID to search (used to fetch member's claims).
        bearer_token:    Authorization header value (Bearer <token>).
        x_api_key:       API key header.
        x_clientrefid:   Client reference ID header.
        session_id:      Optional session ID for tracing.

    Returns:
        Dict with keys:
            - response:       LLM-generated answer string
            - filtered_count: Number of claims passed to LLM
            - total_count:    Total claims from API
            - success:        Whether pipeline completed successfully
            - error:          Error message if success=False
    """
    try:
        # ---------------------------------------------------------------
        # Step 1: Fetch claims from upstream API
        # ---------------------------------------------------------------
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
                "response": (
                    "I wasn't able to find any claims for the provided claim number. "
                    "Please double-check the claim number and try again."
                ),
                "filtered_count": 0,
                "total_count": 0,
                "success": True,
                "error": None,
            }

        logger.info(f"[ClaimsSearch] Fetched {total_count} claims from API")

        # ---------------------------------------------------------------
        # Step 2: Extract member info (for context)
        # ---------------------------------------------------------------
        member_info = extract_member_cagm_from_response(api_response)

        # ---------------------------------------------------------------
        # Step 3: Filter claims based on user query
        # ---------------------------------------------------------------
        filtered_claims = generalized_claims_query(claims, user_query)
        filtered_count = len(filtered_claims)
        logger.info(
            f"[ClaimsSearch] Filtered {total_count} → {filtered_count} claims "
            f"for query: {user_query!r}"
        )

        # Build a filtered response dict for the formatter
        filtered_response = {
            **api_response,
            "claims": filtered_claims,
            "totalCount": filtered_count,
        }

        # ---------------------------------------------------------------
        # Step 4: Format for LLM (trim + compact text)
        # ---------------------------------------------------------------
        llm_context = format_claims_for_llm(
            filtered_response,
            user_query=None,  # Already filtered; don't re-filter
            is_member_history=True,
        )

        logger.info(
            f"[ClaimsSearch] LLM context size: {len(llm_context)} chars "
            f"({filtered_count} claims)"
        )

        # ---------------------------------------------------------------
        # Step 5: Generate response via LLM
        # ---------------------------------------------------------------
        agent = _get_response_agent()
        llm_response = agent.generate_response(
            user_query=user_query,
            claims_context=llm_context,
            member_info=member_info,
            total_claims=total_count,
            filtered_claims=filtered_count,
        )

        return {
            "response": llm_response,
            "filtered_count": filtered_count,
            "total_count": total_count,
            "success": True,
            "error": None,
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[ClaimsSearch] Pipeline error: {e}\n{tb}")
        return {
            "response": (
                "I encountered an issue while looking up the claims. "
                "Please try again shortly."
            ),
            "filtered_count": 0,
            "total_count": 0,
            "success": False,
            "error": str(e),
        }
