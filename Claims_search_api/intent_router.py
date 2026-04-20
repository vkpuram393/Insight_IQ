"""
Claims_search_api.intent_router

Determines whether a user query should be routed to the claims search
pipeline (member-level multi-claim queries) vs the existing single-claim
tool (call_claims_tool).

This module is used as a conditional router in the LangGraph graph,
inserted between build_context and the tool nodes.

No existing classifiers or config files are modified — this is an
additive routing layer that checks for patterns the existing intents
don't cover (member history, multi-claim filtering, etc.).
"""

import re
from typing import Dict, Any
from core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Patterns that indicate a member-level / multi-claim search query
# These are queries the existing single-claim API cannot answer
# ---------------------------------------------------------------------------

_CLAIMS_SEARCH_PATTERNS = [
    # Drug history across claims
    re.compile(r'when\s+was\s+.+\s+taken', re.IGNORECASE),
    re.compile(r'last\s+(claim|fill|prescription)\s+for', re.IGNORECASE),
    re.compile(r'claims?\s+(for|related\s+to|about)\s+(the\s+)?(drug|medication)', re.IGNORECASE),

    # Multi-claim filters
    re.compile(r'all\s+(the\s+)?claims', re.IGNORECASE),
    re.compile(r'show\s+(me\s+)?claims', re.IGNORECASE),
    re.compile(r'list\s+(all\s+)?claims', re.IGNORECASE),

    # Reject code search across claims
    re.compile(r'reject\s*(code|codes)?\s+\d+', re.IGNORECASE),
    re.compile(r'claims?\s+with\s+reject', re.IGNORECASE),

    # Status-based multi-claim queries
    re.compile(r'all\s+(rejected|paid|reversed|cancelled)', re.IGNORECASE),
    re.compile(r'(rejected|paid|reversed)\s+claims', re.IGNORECASE),

    # Month/date range queries
    re.compile(r'claims?\s+(in|for|from|during)\s+(january|february|march|april|may|june|july|august|september|october|november|december|this\s+month|last\s+month)', re.IGNORECASE),

    # Pharmacy / prescriber history
    re.compile(r'claims?\s+(filled|dispensed|from)\s+(at|by)', re.IGNORECASE),
    re.compile(r'(pharmacy|prescriber|doctor)\s+.+\s+claims', re.IGNORECASE),

    # Cost across claims
    re.compile(r'(how\s+much|total|cost|copay|patient\s+pay).+(all|member|history)', re.IGNORECASE),

    # Generic/brand filter
    re.compile(r'(generic|brand)\s+(drug|claim|medication)', re.IGNORECASE),

    # Days supply / quantity filter
    re.compile(r'\d+\s+day\s+supply', re.IGNORECASE),

    # Prior auth / specialty / compound across claims
    re.compile(r'(prior\s+auth|specialty|compound)\s+claims', re.IGNORECASE),
    re.compile(r'claims?\s+(with|using|used)\s+(prior\s+auth|authorization)', re.IGNORECASE),
    re.compile(r'(which|what)\s+claims?\s+(used|have|had|with)\s+(prior|auth)', re.IGNORECASE),

    # Settlement / diagnosis across claims
    re.compile(r'(settlement|diagnosis|icd)\s+(code)?\s*\w+\s+claims?', re.IGNORECASE),
    re.compile(r'claims?\s+with\s+(settlement|diagnosis|icd)', re.IGNORECASE),

    # NDC / GPI / manufacturer across claims
    re.compile(r'(ndc|gpi|manufactured|manufacturer)\s+', re.IGNORECASE),

    # Plan-based queries
    re.compile(r'claims?\s+(under|for|on)\s+plan', re.IGNORECASE),

    # Retail / mail order pharmacy type
    re.compile(r'(retail|mail\s*order)\s+(pharmacy\s+)?claims?', re.IGNORECASE),

    # Refill queries
    re.compile(r'(all\s+)?refills?\s+(for|of)', re.IGNORECASE),
    re.compile(r'show\s+(all\s+)?refills', re.IGNORECASE),
]


def is_claims_search_query(state: Dict[str, Any]) -> bool:
    """
    Determine if the user's query should route to the claims search pipeline.

    Args:
        state: LangGraph AgentState dict

    Returns:
        True if the query matches claims-search patterns
    """
    user_input = (state.get("user_input") or "").strip()
    if not user_input:
        return False

    for pattern in _CLAIMS_SEARCH_PATTERNS:
        if pattern.search(user_input):
            logger.info(
                f"🔍 Claims search pattern matched: {pattern.pattern!r} "
                f"for query: {user_input!r}"
            )
            return True

    return False


def route_after_build_context(state: Dict[str, Any]) -> str:
    """
    LangGraph conditional router: after build_context, decide which tool to call.

    Returns:
        'call_claims_search' if it's a member-level search query
        'call_claims_tool'   for standard single-claim queries
    """
    if is_claims_search_query(state):
        logger.info("🔀 Routing to claims_search pipeline (multi-claim query)")
        return "call_claims_search"
    else:
        logger.info("🔀 Routing to standard claims tool (single-claim query)")
        return "call_claims_tool"
