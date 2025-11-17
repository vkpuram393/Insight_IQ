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

def confidence_check_router(state: AgentState) -> Literal["clarification", "tool_call", "master_llm"]:
    """Route based on confidence, entity completeness, and query complexity.

    TWO-STAGE ROUTING:
      Stage 1 (Intent Classifier): Fast keyword-based classification
      Stage 2 (Master LLM Agent): Comprehensive LLM analysis for complex/unclear cases

    Rules:
      1. If Stage 1 set needs_clarification=True (missing slots) -> clarification
      2. CRITICAL: If query is_complex=True (aggregations, comparisons) -> master_llm (even if high confidence!)
      3. Else if confidence < threshold AND no entities -> master_llm (Stage 2 analysis)
      4. Else if confidence < threshold but has entities -> tool_call (trust entities)
      5. Else -> tool_call
    """
    confidence = state.get("confidence", 0.0)
    needs_clarification = state.get("needs_clarification", False)
    is_complex = state.get("is_complex", False)
    entities = state.get("entities", {})
    threshold = settings.confidence_threshold

    # RULE 1: Query is complex (CRITICAL: Route to LLM BEFORE checking slots!)
    # Complex queries like "summarize my claims" need LLM even if missing entities
    if is_complex:
        logger.info(f"🧠 Complex query detected (confidence: {confidence:.2f}) -> Master LLM Agent")
        logger.info("   Reason: Query contains aggregations, comparisons, or multiple conditions")
        return "master_llm"

    # RULE 2: Stage 1 detected missing slots (e.g., "show my claim" but no claim ID)
    if needs_clarification:
        logger.info("❓ Stage 1 detected missing required slots -> Clarification")
        return "clarification"

    # RULE 3 & 4: Low confidence routing
    if confidence < threshold:
        # NEW: Two-stage routing!
        # If confidence is low and no entities found, route to Master LLM Agent for comprehensive analysis
        has_any_entity = any(entities.values()) if isinstance(entities, dict) else False
        
        if not has_any_entity:
            # No entities found - can't call API
            # Route to Master LLM Agent to analyze from scratch
            logger.info(f"⚠️ Low confidence ({confidence:.2f}) + no entities -> Master LLM Agent (Stage 2)")
            return "master_llm"
        else:
            # Has entities but low confidence
            # Trust the entities and go to API anyway
            logger.info(f"⚠️ Low confidence ({confidence:.2f}) but has entities -> Tool Call")
            return "tool_call"

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


def route_after_api_call(state: AgentState) -> Literal["master_llm", "response_agent"]:
    """
    Route after API call with LLM fallback
    
    CRITICAL: When multiple APIs exist, wrong API call might return 400 error.
    This router catches errors and falls back to Master LLM Agent.
    
    Rules:
      - If api_error exists → route to master_llm (LLM figures it out!)
      - Else → route to response_agent (success)
    
    NOTE: Simplified version (no retry loop for now)
    Team can add retry logic later if needed
    """
    api_error = state.get("api_error")
    
    if api_error:
        logger.error(f"⚠️ API Error detected: {api_error}")
        logger.info("→ Routing to master_llm (API FAILED - LLM FALLBACK!)")
        return "master_llm"
    
    # Success
    logger.info("→ Routing to response_agent (API success)")
    return "response_agent"
