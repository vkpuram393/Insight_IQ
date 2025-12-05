"""
Extended Intent Agent Node (NO LLM REQUIRED)
Alternative to intent_agent.py that uses keyword-based or embedding-based classification

🚀 BENEFITS:
- NO LLM costs for intent classification
- Faster (< 10ms for keyword, ~200ms for embeddings vs 500ms+ for LLM)
- 30+ CVS-specific intents
- Production-ready with conversation continuity
- Can run side-by-side with existing LLM agent

🔧 TO USE:
In langgraph_agent.py, replace:
    workflow.add_node("intent_agent", intent_agent_node)
With:
    workflow.add_node("intent_agent", extended_intent_agent_node)
"""

from typing import Dict, Any
from state.schema import AgentState
from core.logger import get_logger
from core.logging_context import log_state_snapshot
from classifiers.intent_classifier_wrapper import classify_intent_unified, extract_entities_unified
from config.api_routing_config import get_api_config  # NEW: Get API endpoint from config
from config.config import settings  # NEW: For classifier_type metadata

logger = get_logger(__name__)


async def extended_intent_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Classify user intent using CVS keyword-based classifier
    NO LLM required - fast and cost-effective
    """
    logger.info("🤖 CVS AGENT: Intent Classification (NO LLM)")
    
    text = state["text"]
    
    # Classify intent using wrapper (respects settings.use_embedding_classifier)
    intent_result = classify_intent_unified(text)
    
    # ========== CHECK 1: Handle embedding classifier failure ==========
    if intent_result.get('embedding_failed', False):
        logger.warning("❌ Embedding classifier failed - setting flag to route to response_agent (LLM)")
        fallback_reason = intent_result.get('fallback_reason', 'Unknown')
        
        result = {
            "intent": None,
            "confidence": 0.0,
            "entities": {},
            "embedding_failed": True,  # KEY: Router will check this flag
            "is_complex": False,
            "needs_clarification": False,
            "metadata": {
                "embedding_failure": True,
                "fallback_reason": fallback_reason
            }
        }
        
        # Log to telemetry
        await log_state_snapshot(state, "intent_agent", result)
        
        logger.info("🔄 Routing to response_agent (LLM) due to embedding failure")
        return result
    
    # ========== Normal flow (embedding succeeded) ==========
    intent = intent_result['intent']
    confidence = intent_result['confidence']
    
    logger.info(f"🎯 Intent: {intent} ({confidence:.2f})")
    
    # Extract entities
    entity_result = extract_entities_unified(text, intent=intent)
    entities = entity_result['entities']
    
    logger.info(f"📦 Entities: {entities}")
    
    # ========== NEW: Get API configuration from config file ==========
    api_config = get_api_config(intent)
    api_endpoint = api_config.get("api_endpoint")
    required_entities_list = api_config.get("required_entities", [])
    requires_llm = api_config.get("requires_llm", False)
    
    # Log API routing decision
    if api_endpoint:
        logger.info(f"🔗 API Endpoint: {api_endpoint}")
        logger.info(f"📋 Required Entities: {required_entities_list}")
    else:
        logger.info(f"💬 No API needed - Intent will use {'LLM' if requires_llm else 'FAQ/Knowledge Base'}")
    
    # ========== USE CLASSIFIER'S COMPLEXITY DETECTION (no duplicate calculation) ==========
    is_complex = intent_result.get('is_complex', False)
    
    if is_complex:
        logger.info(f"🧠 Complex query detected by classifier (is_complex=True)")
        logger.info("   Query contains aggregations, comparisons, date ranges, or multiple conditions")
    
    # Check if clarification needed (only for missing slots, not low confidence)
    needs_clarification = False
    clarifying_question = None
    
    # Check for missing required slots (e.g., "show my claim" but no claim ID)
    if 'slot_validation' in entity_result:
        slot_validation = entity_result['slot_validation']
        if not slot_validation['has_all_slots']:
            needs_clarification = True
            missing = slot_validation['missing_slots']
            
            # Generate slot request message
            if 'claim_id' in missing:
                clarifying_question = "I need your claim ID to look that up. Could you provide it?"
            elif 'member_id' in missing:
                clarifying_question = "I need your member ID to retrieve that information. Could you provide it?"
            elif 'prescription_id' in missing:
                clarifying_question = "I need your prescription number. Could you provide it?"
            else:
                clarifying_question = f"I need more information to help you. Could you provide: {', '.join(missing)}?"
    
    # ========== BUILD RESULT WITH METADATA ==========
    # Extract missing_slots from slot_validation if available
    missing_slots_list = []
    if 'slot_validation' in entity_result:
        missing_slots_list = entity_result['slot_validation'].get('missing_slots', [])
    
    result = {
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
        "is_complex": is_complex,  # NEW: Complexity detection for LLM routing
        "needs_clarification": needs_clarification,
        "clarifying_question": clarifying_question,
        "missing_slots": missing_slots_list,  # Required for clarification_node
        # NEW: API routing info from config
        "api_endpoint": api_endpoint,
        "required_entities_list": required_entities_list,
        "requires_llm": requires_llm,
        # NEW: Add observability metadata
        "metadata": {
            **state.get("metadata", {}),  # Preserve existing metadata
            "all_scores": intent_result.get('all_scores', {}),  # All intent similarities for debugging
            "is_complex": is_complex,  # Complex query flag (single source of truth)
            "classifier_type": "embedding" if settings.use_embedding_classifier else "keyword",  # Which classifier was used
            "intent_classification_metadata": {
                "top_intent": intent,
                "top_confidence": confidence,
                "num_intents_evaluated": len(intent_result.get('all_scores', {})),
            }
        }
    }
    
    # Log to telemetry database (same as remote MVP-1's intent_agent.py)
    await log_state_snapshot(state, "intent_agent", result)
    
    return result

