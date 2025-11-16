"""
Confidence Check Node - Checks confidence and routes accordingly
"""

import json
import traceback
from pathlib import Path
from typing import Dict, Any, Literal
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger
from core.error_models import (
    AgentError,
    ErrorCode,
    ErrorCategory,
    ErrorSeverity,
    create_low_confidence_error,
    create_internal_error
)
from persistence import PersistenceStoreFactory

logger = get_logger(__name__)

# Load config once (will be reloaded each time as requested)
_config_cache = None

def _load_config() -> Dict[str, Any]:
    """Load domain config from JSON file"""
    global _config_cache
    config_path = Path(__file__).parent.parent / "config" / "domain_config.json"
    try:
        with open(config_path, 'r') as f:
            _config_cache = json.load(f)
        return _config_cache
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Return defaults
        return {
            "confidence_threshold": 0.7,
            "clarification_messages": {
                "low_confidence": "I'm not quite sure what you're asking. Could you rephrase your question?",
                "missing_entity_template": "Could you provide your {missing_entity}?"
            }
        }

def confidence_check_router(state: AgentState) -> Literal["clarification", "tool_call"]:
    """Route based on confidence and entity completeness (kept for backward compatibility).

    Rules:
      - If intent needs a claim_number and it's missing -> clarification
      - Else if confidence < threshold -> clarification
      - Else -> tool_call
    """
    config = _load_config()
    threshold = config.get("confidence_threshold", 0.7)
    
    intent = state.get("intent")
    entities = state.get("entities") or {}
    confidence = state.get("confidence", 0.0)

    # Entity completeness rule for claim rejection intent
    if intent == "claim_rejection_reason" and not entities.get("claim_number"):
        logger.info("⚠️ Missing claim_number for rejection intent -> Clarification")
        return "clarification"

    if confidence < threshold:
        logger.info(f"⚠️ Low confidence ({confidence:.2f}) < {threshold:.2f} -> Clarification")
        return "clarification"

    logger.info(f"✅ Confidence OK ({confidence:.2f}) -> Tool Call")
    return "tool_call"

async def confidence_checker_node(state: AgentState) -> Dict[str, Any]:
    """
    Confidence Checker Node - Checks confidence against threshold
    
    If low confidence:
        - Constructs clarification object
        - Returns clarification state
    
    If high confidence:
        - Constructs context builder input object
        - Logs to SQLite
        - Returns state to proceed to context builder
    """
    node_name = "confidence_checker"
    session_id = state.get("session_id", "unknown")
    request_id = state.get("uuid")
    user_id = state.get("user_info", {}).get("user_id")
    
    try:
        # Load config
        config = _load_config()
        threshold = config.get("confidence_threshold", 0.7)
        clarification_messages = config.get("clarification_messages", {})
        
        # Get state values
        intent = state.get("intent")
        entities = state.get("entities") or {}
        confidence = state.get("confidence", 0.0)
        
        logger.info(f"🔍 Confidence Check: intent={intent}, confidence={confidence:.2f}, threshold={threshold:.2f}")
        
        # Check for missing entities
        missing_entities = []
        if intent == "claim_rejection_reason" and not entities.get("claim_number"):
            missing_entities.append("claim_number")
        
        # Determine if confidence is low
        confidence_low = confidence < threshold
        
        # Get persistence store for logging
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        
        if confidence_low or missing_entities:
            # Low confidence or missing entities -> clarification
            logger.info(f"⚠️ Low confidence or missing entities -> Clarification")
            
            # Determine clarification reason
            if missing_entities:
                clarification_reason = "missing_entity"
                # Use template for missing entity
                template = clarification_messages.get("missing_entity_template", "Could you provide your {missing_entity}?")
                missing_entity_name = missing_entities[0].replace("_", " ")
                clarifying_question = template.format(missing_entity=missing_entity_name)
            else:
                clarification_reason = "low_confidence"
                clarifying_question = clarification_messages.get("low_confidence", "I'm not quite sure what you're asking. Could you rephrase your question?")
            
            # Construct clarification object (matching current structure)
            clarification_result = {
                "needs_clarification": True,
                "clarifying_question": clarifying_question,
                "response": clarifying_question,
                "metadata": {
                    **state.get("metadata", {}),
                    "clarification": True,
                    "clarification_reason": clarification_reason,
                    "missing_entities": missing_entities,
                    "confidence": confidence,
                    "threshold": threshold
                }
            }
            
            # Log confidence check decision
            await persistence_store.log_audit(
                session_id=session_id,
                node_name=node_name,
                event_type="confidence_check_decision",
                data={
                    "decision": "clarification",
                    "confidence": confidence,
                    "threshold": threshold,
                    "missing_entities": missing_entities,
                    "reason": clarification_reason
                },
                request_id=request_id,
                user_id=user_id
            )
            
            return clarification_result
        
        else:
            # High confidence -> proceed to context builder
            logger.info(f"✅ High confidence -> Context Builder")
            
            # Get conversation history from memory store
            from memory import MemoryStoreFactory
            memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
            chat_history = await memory_store.get_session_history(session_id)
            
            # Construct context builder input object
            context_builder_input = {
                "intent": intent,
                "confidence": confidence,
                "entities": entities,
                "domain": state.get("domain", "claims"),  # Placeholder
                "uuid": request_id,
                "user_profile": {
                    "user_id": user_id or state.get("user_info", {}).get("user_id")
                },
                "chat_history": chat_history
            }
            
            # Log confidence check decision
            await persistence_store.log_audit(
                session_id=session_id,
                node_name=node_name,
                event_type="confidence_check_decision",
                data={
                    "decision": "proceed",
                    "confidence": confidence,
                    "threshold": threshold,
                    "intent": intent
                },
                request_id=request_id,
                user_id=user_id
            )
            
            # Log context builder input
            await persistence_store.log_audit(
                session_id=session_id,
                node_name=node_name,
                event_type="context_builder_input",
                data=context_builder_input,
                request_id=request_id,
                user_id=user_id
            )
            
            # Return state to proceed (context builder will be called next)
            return {
                "metadata": {
                    **state.get("metadata", {}),
                    "confidence_check_passed": True,
                    "context_builder_input": context_builder_input
                }
            }
            
    except Exception as e:
        # Log exception
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Confidence checker failed: {str(e)}",
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
        
        logger.error(f"🚨 Exception in confidence checker: {e}\n{tb}")
        
        # Return error state (will stop graph)
        return {
            "error": error.user_message,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }
