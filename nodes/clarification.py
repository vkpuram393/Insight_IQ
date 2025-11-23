"""
Clarification Node - Ask questions when unsure
"""

import json
import traceback
from pathlib import Path
from typing import Dict, Any
from state.schema import AgentState
from config.config import settings
from core.logger import get_logger
from core.errors.models import create_internal_error
from core.logging_context import extract_logging_context, log_state_snapshot
from persistence import PersistenceStoreFactory

logger = get_logger(__name__)

def _load_config() -> Dict[str, Any]:
    """Load domain config from JSON file"""
    config_path = Path(__file__).parent.parent / "config" / "domain_config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Return defaults
        return {
            "clarification_messages": {
                "low_confidence": "I'm not quite sure what you're asking. Could you rephrase your question?",
                "missing_entity_template": "Could you provide your {missing_entity}?"
            }
        }

async def clarification_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate clarifying question

    🎓 CONCEPT:
    When confidence is low, we need more info from user.
    This generates an appropriate question to ask.

    FLOW:
        After this node → End graph, return question to user
        User answers → New request with more context
    """
    node_name = "clarification"
    log_ctx = extract_logging_context(state)
    
    try:
        logger.info("❓ Node: Clarification")

        # Load config for clarification templates
        config = _load_config()
        clarification_messages = config.get("clarification_messages", {})
        
        # Get state values
        missing_slots = state.get("missing_slots", [])
        metadata = state.get("metadata", {})
        clarification_reason = metadata.get("clarification_reason", "low_confidence")
        
        # Generate clarification question based on reason
        if clarification_reason == "missing_entity" and missing_slots:
            # Use template for missing entity
            template = clarification_messages.get("missing_entity_template", "Could you provide your {missing_entity}?")
            missing_slot_name = missing_slots[0].replace("_", " ")
            clarifying_question = template.format(missing_entity=missing_slot_name)
            logger.info(f"❓ Generated (missing entity): {clarifying_question}")
        else:
            # Use low confidence template
            clarifying_question = clarification_messages.get("low_confidence", "I'm not quite sure what you're asking. Could you rephrase your question?")
            logger.info(f"❓ Generated (low confidence): {clarifying_question}")

        result = {
            "needs_clarification": True,
            "clarifying_question": clarifying_question,
            "response": clarifying_question,
            "metadata": {
                **state.get("metadata", {}),
                "clarification": True
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result
        
    except Exception as e:
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Clarification failed: {str(e)}",
            stacktrace=tb,
            session_id=log_ctx["session_id"],
            node_name=node_name
        )
        
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        await persistence_store.log_exception(
            error_code=error.error_code.value,
            category=error.category.value,
            severity=error.severity.value,
            message=error.message,
            user_message=error.user_message,
            session_id=log_ctx["session_id"],
            request_id=log_ctx["request_id"],
            node_name=node_name,
            stacktrace=error.stacktrace,
            metadata=error.metadata,
            user_id=log_ctx["user_id"]
        )
        
        logger.error(f"🚨 Exception in clarification: {e}\n{tb}")
        
        result = {
            "error": error.user_message,
            "needs_clarification": True,
            "clarifying_question": error.user_message,
            "response": error.user_message,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value,
                "clarification": True
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result
