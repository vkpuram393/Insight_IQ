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
from agents.intent_classifier_wrapper import classify_intent_unified, extract_entities_unified
from config.api_routing_config import get_api_config  # NEW: Get API endpoint from config

logger = get_logger(__name__)


async def extended_intent_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Classify user intent using CVS keyword-based classifier
    NO LLM required - fast and cost-effective
    """
    logger.info("🤖 CVS AGENT: Intent Classification (NO LLM)")
    
    text = state["text"]
    
    # Classify intent using wrapper (respects settings.use_cvs_intent_classifier)
    intent_result = classify_intent_unified(text)
    
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
    
    # Check query complexity (aggregations, comparisons, date ranges)
    query_lower = text.lower()
    complexity_keywords = {
        'aggregation': ['all', 'total', 'sum', 'count', 'average', 'summarize', 'list all', 'show all'],
        'comparison': ['compare', 'difference', 'versus', 'vs', 'better', 'worse', 'more than', 'less than'],
        'range': ['between', 'from', 'to', 'since', 'until', 'last', 'past', 'recent'],
        'multiple': [' and ', 'both', 'all of', 'each']
    }
    
    is_complex = False
    complexity_reason = []
    
    for category, keywords in complexity_keywords.items():
        if any(kw in query_lower for kw in keywords):
            is_complex = True
            complexity_reason.append(category)
    
    if is_complex:
        logger.info(f"🧠 Complex query detected: {', '.join(complexity_reason)}")
    
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
    
    result = {
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
        "is_complex": is_complex,  # NEW: Complexity detection for LLM routing
        "needs_clarification": needs_clarification,
        "clarifying_question": clarifying_question,
        # NEW: API routing info from config
        "api_endpoint": api_endpoint,
        "required_entities_list": required_entities_list,
        "requires_llm": requires_llm,
    }
    
    # Log to telemetry database (same as remote MVP-1's intent_agent.py)
    await log_state_snapshot(state, "intent_agent", result)
    
    return result

