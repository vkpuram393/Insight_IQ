"""
Confidence Check - Conditional router
"""

from typing import Literal
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

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
    intent = state.get("intent")
    entities = state.get("entities") or {}
    confidence = state.get("confidence", 0.0)
    needs_clarification = state.get("needs_clarification", False)
    is_complex = state.get("is_complex", False)
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
