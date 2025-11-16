"""
Clarification Node - Ask questions when unsure
"""

import traceback
from typing import Dict, Any
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger
from core.error_models import create_internal_error
from persistence import PersistenceStoreFactory

logger = get_logger(__name__)

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
    session_id = state.get("session_id", "unknown")
    request_id = state.get("uuid")
    user_id = state.get("user_info", {}).get("user_id")
    
    try:
        logger.info("❓ Node: Clarification")

        intent = state.get("intent", "unknown")

        # Predefined questions for each intent
        questions = {
            "claim_status": "Could you provide your claim number?",
            "claim_rejection_reason": "Which claim are you asking about?",
            "unknown": "I'm not sure I understand. Are you asking about a claim?"
        }

        question = questions.get(intent, questions["unknown"])

        logger.info(f"❓ Generated: {question}")

        return {
            "needs_clarification": True,
            "clarifying_question": question,
            "response": question,
            "metadata": {**state.get("metadata", {}), "clarification": True}
        }
        
    except Exception as e:
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Clarification failed: {str(e)}",
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
            stacktrace=error.stacktrace,
            metadata=error.metadata,
            user_id=user_id
        )
        
        logger.error(f"🚨 Exception in clarification: {e}\n{tb}")
        
        return {
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
