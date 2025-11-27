"""
LLM Judge Node - Re-classifies intent using LLM when initial classifier has low confidence

This node acts as an "expert reviewer" that takes a second look at uncertain intent classifications.
It uses an LLM (or mock implementation) to re-classify the intent with higher accuracy.
"""

import json
import traceback
from pathlib import Path
from typing import Dict, Any
from state.schema import AgentState
from core.logger import get_logger
from core.errors.models import create_internal_error
from core.logging_context import extract_logging_context, log_state_snapshot
from persistence import PersistenceStoreFactory
from config.config import settings

logger = get_logger(__name__)

# Load config cache
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
            "llm_judge_mock_high_confidence": True,
            "confidence_threshold": 0.7
        }

async def llm_judge_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM Judge Node - Re-classifies intent using LLM when initial classifier has low confidence
    
    This node:
    1. Takes the original intent classification (low confidence)
    2. Uses LLM (or mock) to re-classify with more context
    3. Updates confidence and potentially intent/entities
    4. Sets intent_reclassified = True (prevents infinite loops)
    
    INPUT (from state):
        - intent: Original intent from classifier
        - confidence: Original confidence (low)
        - entities: Original entities
        - text: User's query
        - intent_reclassified: Should be False (initial classification)
    
    OUTPUT (to state):
        - intent: Re-classified intent (or kept original)
        - confidence: Updated confidence (high or low based on mock config)
        - entities: Updated entities (or kept original)
        - intent_reclassified: True (KEY - prevents infinite loop)
    """
    node_name = "llm_judge"
    log_ctx = extract_logging_context(state)
    
    try:
        logger.info("\n" + "="*70)
        logger.info("⚖️  LLM JUDGE NODE - Re-classifying Intent")
        logger.info("="*70)
        
        # Load config
        config = _load_config()
        mock_high_conf = config.get("llm_judge_mock_high_confidence", True)
        threshold = config.get("confidence_threshold", 0.7)
        
        # Get current state values
        original_intent = state.get("intent", "unknown")
        original_confidence = state.get("confidence", 0.0)
        original_entities = state.get("entities") or {}
        text = state.get("text", "")
        intent_reclassified = state.get("intent_reclassified", False)
        
        logger.info(f"📥 Input - Intent: {original_intent}, Confidence: {original_confidence:.2f}")
        logger.info(f"   Entities: {original_entities}")
        logger.info(f"   Text: {text[:100]}...")
        logger.info(f"   intent_reclassified flag: {intent_reclassified}")
        
        # MOCK IMPLEMENTATION (for testing flow)
        # In production, this would call Gemini LLM to re-classify
        if mock_high_conf:
            new_confidence = 0.95  # High confidence
            logger.info("🎭 Mock: Returning HIGH confidence (0.95)")
            logger.info("   Config: llm_judge_mock_high_confidence = true")
        else:
            new_confidence = 0.3  # Low confidence
            logger.info("🎭 Mock: Returning LOW confidence (0.3)")
            logger.info("   Config: llm_judge_mock_high_confidence = false")
        
        # For mock, keep original intent and entities
        # In production, LLM might update these based on deeper analysis
        new_intent = original_intent
        new_entities = original_entities
        
        # IMPORTANT: Clear needs_clarification flag if confidence is now high
        # This allows confidence_checker to re-evaluate and proceed normally
        needs_clarification = new_confidence < threshold
        
        logger.info(f"📤 Output - Intent: {new_intent}, Confidence: {new_confidence:.2f}")
        logger.info(f"   Entities: {new_entities}")
        logger.info(f"   needs_clarification: {needs_clarification} (based on new confidence)")
        logger.info(f"   Setting: intent_reclassified = True (prevents infinite loop)")
        
        # Construct result
        result = {
            "intent": new_intent,
            "confidence": new_confidence,
            "entities": new_entities,
            "intent_reclassified": True,  # KEY: Set flag to True to prevent infinite loops
            "needs_clarification": needs_clarification  # Clear if high confidence
        }
        
        # Log state snapshot
        await log_state_snapshot(state, node_name, result)
        
        logger.info("✅ LLM Judge completed - returning to confidence_checker for re-evaluation")
        return result
        
    except Exception as e:
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"LLM Judge failed: {str(e)}",
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
        
        logger.error(f"🚨 Exception in LLM Judge: {e}\n{tb}")
        
        # Return error state - but still set flag to prevent loops
        result = {
            "intent_reclassified": True,  # Still set flag even on error
            "error": error.user_message,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value,
                "llm_judge_failed": True
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result

