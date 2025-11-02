"""
Claims API Tool - Calls external API
"""

import asyncio
from typing import Dict, Any
from state.schema import AgentState
from core.logger import get_logger

logger = get_logger(__name__)

async def call_claims_tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Call Claims API

    🎓 CONCEPT:
    This simulates calling an external API to get data.
    In production, this would be a real HTTP call.

    INPUT (from state):
        - intent: What data to fetch
        - entities: Parameters (e.g., claim_number)

    OUTPUT (to state):
        - tool_results: API response data
    """

    logger.info("🔧 Node: Call Claims Tool")

    intent = state["intent"]
    entities = state.get("entities", {})
    claim_number = entities.get("claim_number", "12345")  # fallback

    # Simulate API call
    await asyncio.sleep(0.2)

    # Mock responses
    if intent == "claim_status":
        results = {
            "claim_id": claim_number,
            "status": "processing",
            "submitted_date": "2025-01-10",
            "expected_completion": "5-7 business days"
        }
    elif intent == "claim_rejection_reason":
        results = {
            "claim_id": claim_number,
            "status": "rejected",
            "reason": "Requires prior authorization",
            "action_needed": "Doctor must submit documentation"
        }
    else:
        results = {}

    logger.info(f"✅ Tool results: {results}")

    return {"tool_results": results}
