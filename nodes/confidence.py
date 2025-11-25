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

def confidence_check_router(state: AgentState) -> Literal["clarification", "build_context", "llm_judge"]:
    """Route based on confidence, entity completeness, query complexity, and intent_reclassified flag.

    UPDATED ROUTING WITH BOOLEAN FLAG:
      1. llm_judge: Low confidence/complex AND intent_reclassified == False (needs re-classification)
      2. clarification: Missing entities OR low confidence after LLM judge (template-based)
      3. build_context: High confidence + has entities (standard API flow)

    Rules (Priority Order):
      IF intent_reclassified == False (initial classifier result):
        1. If is_complex=True → llm_judge (needs expert review)
        2. If confidence < threshold → llm_judge (needs expert review)
        3. If high confidence + has entities → build_context (proceed directly)
      
      IF intent_reclassified == True (LLM judge already ran):
        1. If confidence >= threshold AND entities present → build_context (expert is confident)
        2. If missing entities OR confidence < threshold → clarification (template-based, no LLM call)
    """
    config = _load_config()
    threshold = config.get("confidence_threshold", 0.7)
    
    # TEMPORARY: Check if LLM judge path is disabled (for testing)
    disable_llm_judge = settings.temporarily_disable_llm_judge_path_for_testing
    
    confidence = state.get("confidence", 0.0)
    intent = state.get("intent", "")  # ✅ Get intent from state
    needs_clarification = state.get("needs_clarification", False)
    is_complex = state.get("is_complex", False)
    intent_reclassified = state.get("intent_reclassified", False)
    entities = state.get("entities") or {}

    # INTENTS_WITHOUT_ENTITIES: Intents that don't require entity extraction
    INTENTS_WITHOUT_ENTITIES = {'out_of_scope', 'greeting', 'help'}

    # Check if we have required entities (direct entity existence check)
    has_entities = bool(entities and any(v is not None for v in entities.values()))

    # DECISION LOGIC: Check intent_reclassified flag first
    if not intent_reclassified:
        # Initial classifier result - can route to LLM judge if not disabled
        # RULE 1: Complex query → LLM Judge (if enabled) OR Clarification (if disabled)
        if is_complex:
            if disable_llm_judge:
                logger.info(f"🧠 Complex query detected (confidence: {confidence:.2f}) -> Clarification (LLM judge path temporarily disabled for testing)")
                logger.info("   Reason: LLM judge path is disabled - routing to clarification instead")
                return "clarification"
            else:
                logger.info(f"🧠 Complex query detected (confidence: {confidence:.2f}) -> LLM Judge")
                logger.info("   Reason: Query contains aggregations, comparisons, or multiple conditions")
                logger.info("   Flag: intent_reclassified=False (initial classification)")
                return "llm_judge"

        # RULE 2: Low confidence → LLM Judge (if enabled) OR Clarification (if disabled)
        if confidence < threshold:
            if disable_llm_judge:
                logger.info(f"⚠️ Low confidence ({confidence:.2f}) < {threshold} -> Clarification (LLM judge path temporarily disabled for testing)")
                logger.info("   Reason: LLM judge path is disabled - routing to clarification instead")
                return "clarification"
            else:
                logger.info(f"⚠️ Low confidence ({confidence:.2f}) < {threshold} -> LLM Judge")
                logger.info("   Reason: Uncertain intent - route to LLM Judge for re-classification")
                logger.info("   Flag: intent_reclassified=False (initial classification)")
                return "llm_judge"

        # RULE 3: No entities → Clarification (except for whitelisted intents)
        if not has_entities and intent not in INTENTS_WITHOUT_ENTITIES:
            logger.info(f"⚠️ No entities extracted for intent '{intent}' -> Clarification")
            logger.info(f"   Reason: Intent requires entities but none were found")
            logger.info("   Flag: intent_reclassified=False (initial classification)")
            return "clarification"

        # RULE 4: High confidence + Has entities (or doesn't need them) → Build Context (direct path)
        if confidence >= threshold and (has_entities or intent in INTENTS_WITHOUT_ENTITIES):
            if has_entities:
                logger.info(f"✅ High confidence ({confidence:.2f}) + entities present -> Build Context")
            else:
                logger.info(f"✅ Intent '{intent}' doesn't require entities -> Build Context")
            logger.info("   Reason: Initial classifier is confident enough")
            logger.info("   Flag: intent_reclassified=False (initial classification)")
            return "build_context"

    else:
        # LLM judge already ran - route based on updated intent
        # RULE 1: High confidence + Has entities (or doesn't need them) → Build Context
        if confidence >= threshold and (has_entities or intent in INTENTS_WITHOUT_ENTITIES):
            if has_entities:
                logger.info(f"✅ High confidence ({confidence:.2f}) + entities present -> Build Context")
            else:
                logger.info(f"✅ Intent '{intent}' doesn't require entities -> Build Context")
            logger.info("   Reason: LLM Judge is confident enough")
            logger.info("   Flag: intent_reclassified=True (LLM judge already ran)")
            return "build_context"

        # RULE 2: Missing entities (for intents that need them) OR low confidence → Clarification (template-based)
        if (not has_entities and intent not in INTENTS_WITHOUT_ENTITIES) or confidence < threshold:
            logger.info(f"⚠️ Missing entities or low confidence -> Clarification (template)")
            logger.info(f"   Confidence: {confidence:.2f}, Has entities: {has_entities}")
            logger.info("   Reason: LLM Judge still uncertain - use template clarification")
            logger.info("   Flag: intent_reclassified=True (LLM judge already ran, won't route to LLM judge again)")
            return "clarification"

    # Fallback: Default to build_context if all else fails
    logger.info(f"✅ Default routing -> Build Context")
    return "build_context"

async def confidence_checker_node(state: AgentState) -> Dict[str, Any]:
    """
    Confidence Checker Node - Checks confidence against threshold and routes accordingly
    
    If low confidence:
        - Sets needs_clarification flag
        - Routes to clarification node (which will generate the question)
    
    If high confidence:
        - Constructs context builder input object
        - Logs to SQLite
        - Returns state to proceed to context builder
    
    Note: Entity checking is handled by the router, not this node.
    """
    node_name = "confidence_checker"
    log_ctx = extract_logging_context(state)
    
    try:
        # Load config
        config = _load_config()
        threshold = config.get("confidence_threshold", 0.7)
        
        # Get state values
        intent = state.get("intent")
        confidence = state.get("confidence", 0.0)
        
        logger.info(f"🔍 Confidence Check: intent={intent}, confidence={confidence:.2f}, threshold={threshold:.2f}")
        
        # Determine if confidence is low
        confidence_low = confidence < threshold
        
        if confidence_low:
            # Low confidence -> route to clarification node
            logger.info(f"⚠️ Low confidence -> Routing to Clarification")
            
            # Just set flags - clarification node will generate the question
            result = {
                "needs_clarification": True,
                "metadata": {
                    **state.get("metadata", {}),
                    "clarification_reason": "low_confidence",
                    "confidence": confidence,
                    "threshold": threshold
                }
            }
            
            # Log full AgentState snapshot after this node
            await log_state_snapshot(state, node_name, result)
            
            return result
        
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
        result = {
            "error": error.user_message,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result
