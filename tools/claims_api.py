"""
Claims API Tool - Calls external API
"""

import asyncio
import traceback
from typing import Dict, Any
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger
from core.error_models import create_internal_error, create_api_error
from persistence import PersistenceStoreFactory

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
    node_name = "call_claims_tool"
    session_id = state.get("session_id", "unknown")
    request_id = state.get("uuid")
    user_id = state.get("user_info", {}).get("user_id")
    
    try:
        logger.info("🔧 Node: Call Claims Tool")

        # ========== NEW: Read API endpoint from state ==========
        api_endpoint = state.get("api_endpoint")
        intent = state["intent"]
        entities = state.get("entities", {})
        claim_number = entities.get("claim_number", "12345")  # fallback
        
        # If no API needed (greeting, help, etc.), skip
        if not api_endpoint:
            logger.info(f"💬 No API call needed for intent '{intent}'")
            return {"tool_results": None, "api_error": None}
        
        logger.info(f"🔗 API Endpoint: {api_endpoint}")

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
        return {"tool_results": results, "api_error": None}
        
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = str(e)
        
        # Check if it's an API-related error
        if "api" in str(e).lower() or "http" in str(e).lower() or "connection" in str(e).lower():
            error = create_api_error(
                api_name="claims_api",
                error_message=error_msg,
                session_id=session_id
            )
        else:
            error = create_internal_error(
                error_message=f"Claims API call failed: {error_msg}",
                stacktrace=tb,
                session_id=session_id,
                node_name=node_name
            )
        
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        await persistence_store.log_exception(
            error_code=error.error_code.value,
            category=error.category.value,
            severity=error.severity.value,
            message=error.message,
            user_message=error.user_message,
            session_id=session_id,
            request_id=request_id,
            node_name=node_name,
            stacktrace=error.stacktrace or tb,
            metadata=error.metadata,
            user_id=user_id
        )
        
        logger.error(f"🚨 Exception in claims tool: {e}\n{tb}")
        
        # Return error for fallback routing
        return {
            "error": error.user_message,
            "tool_results": {},
            "api_error": error_msg,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }
