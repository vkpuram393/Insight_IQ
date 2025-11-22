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

def confidence_check_router(state: AgentState) -> Literal["clarification", "build_context", "response_agent"]:
    """Route based on confidence, entity completeness, and query complexity.

    THREE-WAY ROUTING:
      1. clarification: Missing required entities OR high confidence + no entities
      2. build_context → API: Simple query with entities
      3. response_agent: Complex queries OR low confidence + has entities OR embedding failed

    Rules:
      0. If embedding_failed=True -> response_agent (embedding classifier unavailable)
      1. If query is_complex=True -> response_agent (LLM handles directly)
      2. If needs_clarification=True -> clarification (missing required slots)
      3. If confidence >= threshold AND no entities -> clarification (needs more info)
      4. If confidence < threshold AND has entities -> response_agent (LLM with context)
      5. Else -> build_context
    """
    config = _load_config()
    threshold = config.get("confidence_threshold", 0.7)
    
    confidence = state.get("confidence", 0.0)
    missing_slots = state.get("missing_slots") or []
    needs_clarification = state.get("needs_clarification", False)
    is_complex = state.get("is_complex", False)
    embedding_failed = state.get("embedding_failed", False)
    entities = state.get("entities", {})
    confidence_check_passed = state.get("metadata", {}).get("confidence_check_passed", False)

    # RULE 0: Embedding classifier failed - route directly to LLM
    if embedding_failed:
        logger.info(f"❌ Embedding classifier failed -> Response Agent (LLM fallback)")
        logger.info("   Reason: .pkl cache corrupted or Azure OpenAI unavailable")
        return "response_agent"

    # RULE 1: Query is complex (CRITICAL: Route to LLM BEFORE checking slots!)
    # Complex queries like "summarize all my claims" need LLM reasoning, skip API
    if is_complex:
        logger.info(f"🧠 Complex query detected (confidence: {confidence:.2f}) -> Response Agent (LLM)")
        logger.info("   Reason: Query contains aggregations, comparisons, or multiple conditions")
        return "response_agent"

    # RULE 2: Missing required entities
    if needs_clarification:
        logger.info(f"⚠️ Needs clarification (missing entities) -> Clarification")
        return "clarification"
    
    # Check if confidence_checker_node already passed
    if confidence_check_passed:
        logger.info(f"✅ Confidence check passed (from confidence_checker) -> Build Context")
        return "build_context"

    # Check for missing required slots (from intent classifier)
    if missing_slots:
        logger.info(f"⚠️ Missing required slots: {missing_slots} -> Clarification")
        return "clarification"

    # RULE 3: High confidence + NO entities -> Clarification (user needs to provide more info)
    # RULE 4: Low confidence + HAS entities -> Response Agent (LLM with context)
    has_any_entity = any(entities.values()) if isinstance(entities, dict) else False
    
    if confidence >= threshold:
        # High confidence
        if not has_any_entity:
            logger.info(f"✅ High confidence ({confidence:.2f}) but NO entities -> Clarification")
            logger.info("   Reason: Need entity (claim ID, member ID, etc.) to proceed")
            return "clarification"
        else:
            logger.info(f"✅ High confidence ({confidence:.2f}) + has entities -> Build Context")
            return "build_context"
    else:
        # Low confidence
        if has_any_entity:
            logger.info(f"⚠️ Low confidence ({confidence:.2f}) but HAS entities -> Response Agent (LLM)")
            logger.info("   Reason: Let LLM interpret ambiguous query with entity context")
            return "response_agent"
        else:
            logger.info(f"⚠️ Low confidence ({confidence:.2f}) + no entities -> Response Agent (LLM)")
            logger.info("   Reason: LLM fallback for unclear query")
            return "response_agent"

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
