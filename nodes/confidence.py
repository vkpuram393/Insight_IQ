"""
Confidence Check Node - Checks confidence and routes accordingly
"""

import json
import traceback
from pathlib import Path
from typing import Dict, Any, Literal
from state.schema import AgentState
from config.config import settings
from core.logger import get_logger
from core.errors.models import (
    AgentError,
    ErrorCode,
    ErrorCategory,
    ErrorSeverity,
    create_low_confidence_error,
    create_internal_error
)
from core.logging_context import extract_logging_context, log_state_snapshot
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

def confidence_check_router(state: AgentState) -> Literal["clarification", "build_context", "response_agent"]:
    """Route based on confidence, entity completeness, and query complexity.

    SIMPLIFIED THREE-WAY ROUTING:
      1. clarification: Missing entities (ask user for specific information)
      2. build_context → API: High confidence + has entities (clear, simple query)
      3. response_agent: Complex OR low confidence (will become "LLM Judge Agent")

    Rules (Priority Order):
      1. If is_complex=True → response_agent (will become LLM Judge Agent)
      2. If confidence < threshold → response_agent (will become LLM Judge Agent)
      3. If missing entities → clarification (existing clarification engine)
      4. If high confidence + has entities → build_context (standard API flow)
    """
    config = _load_config()
    threshold = config.get("confidence_threshold", 0.7)
    
    confidence = state.get("confidence", 0.0)
    needs_clarification = state.get("needs_clarification", False)
    is_complex = state.get("is_complex", False)
    embedding_failed = state.get("embedding_failed", False)

    # RULE 1: Complex query → Response Agent (will become LLM Judge Agent)
    # Complex queries like "summarize all my claims" need LLM reasoning
    if is_complex:
        logger.info(f"🧠 Complex query detected (confidence: {confidence:.2f}) -> Response Agent (LLM Judge)")
        logger.info("   Reason: Query contains aggregations, comparisons, or multiple conditions")
        return "response_agent"

    # RULE 2: Low confidence → Response Agent (will become LLM Judge Agent)
    # Let LLM Judge handle uncertain intent classification
    if confidence < threshold:
        logger.info(f"⚠️ Low confidence ({confidence:.2f}) < {threshold} -> Response Agent (LLM Judge)")
        logger.info("   Reason: Uncertain intent - route to LLM Judge for decision")
        return "response_agent"

    # RULE 3: Missing entities → Clarification Engine
    # Use existing clarification node to ask user for specific entities
    if needs_clarification:
        logger.info(f"⚠️ Missing required entities -> Clarification Engine")
        logger.info("   Reason: Classifier detected missing required slots")
        return "clarification"

    # RULE 4: High confidence + Has entities → Build Context (Standard API flow)
    logger.info(f"✅ High confidence ({confidence:.2f}) + ready for API -> Build Context")
    return "build_context"

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
    log_ctx = extract_logging_context(state)
    
    try:
        # Load config
        config = _load_config()
        threshold = config.get("confidence_threshold", 0.7)
        clarification_messages = config.get("clarification_messages", {})
        
        # Get state values
        intent = state.get("intent")
        entities = state.get("entities") or {}
        slots = state.get("slots") or {}
        required_slots = state.get("required_slots") or []
        missing_slots = state.get("missing_slots") or []
        confidence = state.get("confidence", 0.0)
        
        logger.info(f"🔍 Confidence Check: intent={intent}, confidence={confidence:.2f}, threshold={threshold:.2f}")
        logger.info(f"   Required slots: {required_slots}, Missing slots: {missing_slots}")
        
        # Determine if confidence is low
        confidence_low = confidence < threshold
        
        if confidence_low or missing_slots:
            # Low confidence or missing slots -> clarification
            logger.info(f"⚠️ Low confidence or missing slots -> Clarification")
            
            # Determine clarification reason
            if missing_slots:
                clarification_reason = "missing_entity"
                # Use template for missing slot
                template = clarification_messages.get("missing_entity_template", "Could you provide your {missing_entity}?")
                missing_slot_name = missing_slots[0].replace("_", " ")
                clarifying_question = template.format(missing_entity=missing_slot_name)
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
                    "missing_slots": missing_slots,
                    "confidence": confidence,
                    "threshold": threshold
                }
            }
            
            # Log full AgentState snapshot after this node
            await log_state_snapshot(state, node_name, clarification_result)
            
            return clarification_result
        
        else:
            # High confidence -> proceed to context builder
            logger.info(f"✅ High confidence -> Context Builder")
            
            # Get conversation history from memory store
            from memory import MemoryStoreFactory
            memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
            chat_history = await memory_store.get_session_history(log_ctx["session_id"])
            
            # Return state to proceed (context builder will be called next)
            proceed_result = {
                "metadata": {
                    **state.get("metadata", {}),
                    "confidence_check_passed": True
                }
            }
            
            # Log full AgentState snapshot after this node
            await log_state_snapshot(state, node_name, proceed_result)
            
            return proceed_result
            
    except Exception as e:
        # Log exception
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Confidence checker failed: {str(e)}",
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
