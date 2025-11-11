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
        - api_endpoint: Which API endpoint to call (NEW - from config!)
        - intent: What data to fetch
        - entities: Parameters (e.g., claim_number)

    OUTPUT (to state):
        - tool_results: API response data
    """

    logger.info("🔧 Node: Call Claims Tool")

    # ========== NEW: Read API endpoint from state ==========
    api_endpoint = state.get("api_endpoint")
    intent = state["intent"]
    entities = state.get("entities", {})
    claim_number = entities.get("claim_number", "12345")  # fallback
    
    # If no API needed (greeting, help, etc.), skip
    if not api_endpoint:
        logger.info(f"💬 No API call needed for intent '{intent}'")
        state["api_error"] = None
        return {"tool_results": None}
    
    logger.info(f"🔗 API Endpoint: {api_endpoint}")

    # ========== ERROR HANDLING: Catch API failures ==========
    try:
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
            # Default mock for other intents
            # Team will replace with real API calls
            results = {
                "claim_id": claim_number,
                "message": f"Mock data for intent: {intent}",
                "api_endpoint": api_endpoint
            }

        logger.info(f"✅ Tool results: {results}")
        
        # Clear error on success
        state["api_error"] = None
        return {"tool_results": results}
        
    except Exception as e:
        # API call failed (network error, 400, 500, wrong API, etc.)
        error_msg = str(e)
        logger.error(f"❌ API Error: {error_msg}")
        
        # Set error in state for fallback routing
        state["api_error"] = error_msg
        state["tool_results"] = {}
        
        return {
            "tool_results": {},
            "api_error": error_msg
        }
