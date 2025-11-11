"""
Master LLM Agent Node - STAGE 2 ROUTING
This is the comprehensive LLM analysis that catches misclassifications from Stage 1
"""

from typing import Dict, Any
from state.schema import AgentState
from core.logger import get_logger
import re

logger = get_logger(__name__)


async def master_llm_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Master LLM Agent: Comprehensive query analysis (Stage 2)
    
    This node is called when Stage 1 (Intent Classifier) routes to RAG path.
    The LLM analyzes the query from scratch and decides:
    - call_api: Reroute to API path (Stage 1 misclassified)
    - search_faq: Confirm general question (search knowledge base)
    - ask_clarification: Need more info from user
    - general_response: Simple conversational response
    """
    logger.info("=" * 80)
    logger.info("🧠 Master LLM Agent (STAGE 2 - Comprehensive Analysis)")
    logger.info("=" * 80)
    
    query = state.get("text", "").strip()
    
    # Safety check: empty query
    if not query:
        logger.warning("⚠️ Empty query received")
        state["response"] = "I didn't receive a question. How can I help you?"
        return state
    
    # ========== LLM COMPREHENSIVE ANALYSIS ==========
    # In production, this would call actual LLM
    # For now, using rule-based logic that simulates LLM decision
    
    try:
        llm_decision = _analyze_query_comprehensive(query, state.get("entities", {}))
        
        logger.info(f"✅ LLM Decision: {llm_decision['action']} (confidence: {llm_decision['confidence']:.2f})")
        logger.info(f"   Reasoning: {llm_decision['reasoning']}")
    except Exception as e:
        logger.error(f"❌ Master LLM Agent error: {e}")
        # Fallback to safe default
        llm_decision = {
            'action': 'search_faq',
            'confidence': 0.50,
            'reasoning': f'Error during analysis: {str(e)} - defaulting to FAQ search',
            'entities': state.get("entities", {})
        }
    
    # Store LLM decision in state
    state["llm_action"] = llm_decision["action"]
    state["llm_confidence"] = llm_decision["confidence"]
    state["llm_reasoning"] = llm_decision["reasoning"]
    
    action = llm_decision["action"]
    
    # ========== EXECUTE LLM DECISION ==========
    
    if action == "call_api":
        # LLM detected this needs API (Stage 1 misclassified!)
        logger.info("🔄 LLM REROUTE: Stage 1 said no API, but LLM detected API query!")
        
        # Update entities with LLM-extracted ones
        llm_entities = llm_decision.get("entities", {})
        
        # Ensure state["entities"] exists before updating
        if "entities" not in state or state["entities"] is None:
            state["entities"] = {}
        
        for key, value in llm_entities.items():
            if value:
                state["entities"][key] = value
        
        # Check if we have entities or need clarification
        if not any(state["entities"].values()):
            # No entities found - ask for clarification
            logger.info("   ⚠️ LLM detected API query but no entities - needs clarification")
            state["llm_action"] = "ask_clarification"
            state["needs_clarification"] = True
            clarification = llm_decision.get("clarification_prompt", "Could you provide more details?")
            state["response"] = clarification
        else:
            # Entities found! Reroute to API path
            logger.info("   ✅ Rerouting to API path with entities!")
            state["llm_rerouted"] = True
            state["needs_api_reroute"] = True  # Signal to router
            
            # Update intent if needed
            if not state.get("intent") or state.get("intent") == "unknown":
                state["intent"] = "llm_detected_api"
    
    elif action == "search_faq":
        # LLM confirmed general question - would search FAQ/knowledge base
        logger.info("📚 LLM confirmed: General knowledge question")
        state["needs_faq"] = True
        # Note: Actual FAQ search would happen in a separate node
        # For now, just mark that FAQ is needed
        state["response"] = "I can help with general questions about pharmacy benefits. What would you like to know?"
    
    elif action == "ask_clarification":
        # LLM needs more info
        logger.info("❓ LLM decision: Need clarification")
        state["needs_clarification"] = True
        clarification = llm_decision.get("clarification_prompt", "Could you provide more details?")
        state["response"] = clarification
    
    elif action == "general_response":
        # LLM can answer directly
        logger.info("💬 LLM decision: Direct response")
        response = llm_decision.get("response", "How can I help you today?")
        state["response"] = response
    
    return state


def _analyze_query_comprehensive(query: str, entities_found: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate LLM comprehensive analysis
    In production, this would be an actual LLM call
    
    Returns decision with:
    - action: 'call_api' | 'search_faq' | 'ask_clarification' | 'general_response'
    - confidence: 0.0-1.0
    - reasoning: Why this action was chosen
    - entities: Extracted entities
    - clarification_prompt: What to ask (if action=ask_clarification)
    - response: Direct answer (if action=general_response)
    """
    query_lower = query.lower()
    
    # ========== RULE 1: Check for entity patterns ==========
    claim_id_pattern = r'\b(CLM\d{3,10})\b'
    member_id_pattern = r'\b(MEM\d{3,10})\b'
    prescription_id_pattern = r'\b(RX\d{3,10})\b'
    
    claim_match = re.search(claim_id_pattern, query, re.IGNORECASE)
    member_match = re.search(member_id_pattern, query, re.IGNORECASE)
    rx_match = re.search(prescription_id_pattern, query, re.IGNORECASE)
    
    extracted_entities = {
        'claim_id': claim_match.group(1).upper() if claim_match else entities_found.get('claim_id'),
        'member_id': member_match.group(1).upper() if member_match else entities_found.get('member_id'),
        'prescription_id': rx_match.group(1).upper() if rx_match else entities_found.get('prescription_id')
    }
    
    # ========== RULE 2: Personal data keywords ==========
    personal_indicators = [
        'my claim', 'my prescription', 'my medication', 'my drug', 'my member',
        'my benefits', 'my coverage', 'my copay', 'my deductible',
        'did i get', 'do i have', 'was i', 'am i', 'can i get my'
    ]
    
    has_personal_indicator = any(indicator in query_lower for indicator in personal_indicators)
    
    # ========== RULE 3: API-specific action verbs ==========
    api_verbs = [
        'show', 'check', 'track', 'find', 'look up', 'get', 'view',
        'status of', 'details of', 'information about', 'rejected', 'denied', 'pending'
    ]
    
    has_api_verb = any(verb in query_lower for verb in api_verbs)
    
    # ========== RULE 4: General knowledge keywords ==========
    general_keywords = [
        'what is', 'how do', 'how does', 'can you explain', 'tell me about',
        'what are the', 'how to', 'when should', 'where can'
    ]
    
    has_general_keyword = any(keyword in query_lower for keyword in general_keywords)
    is_truly_general = has_general_keyword and not has_personal_indicator
    
    # ========== RULE 5: Simple conversational ==========
    simple_responses = {
        'hello': 'greeting',
        'hi': 'greeting',
        'hey': 'greeting',
        'thank': 'acknowledgment',
        'thanks': 'acknowledgment',
        'bye': 'farewell',
        'goodbye': 'farewell'
    }
    
    for word, response_type in simple_responses.items():
        if word in query_lower:
            return {
                'action': 'general_response',
                'confidence': 0.95,
                'reasoning': f'Simple {response_type} - can respond directly',
                'entities': extracted_entities,
                'response': _get_simple_response(response_type)
            }
    
    # ========== DECISION LOGIC ==========
    
    # DECISION 1: Has entity codes → Definitely API
    if any(extracted_entities.values()):
        return {
            'action': 'call_api',
            'confidence': 0.98,
            'reasoning': f'Found entity codes: {extracted_entities}',
            'entities': extracted_entities
        }
    
    # DECISION 2: Personal indicator + API verb → API (but need clarification)
    if has_personal_indicator and has_api_verb:
        # Determine what entity is missing
        if 'claim' in query_lower or 'reject' in query_lower:
            missing_entity = 'claim number'
        elif 'prescription' in query_lower or 'medication' in query_lower or 'drug' in query_lower:
            missing_entity = 'prescription ID or member ID'
        elif 'member' in query_lower or 'benefit' in query_lower:
            missing_entity = 'member ID'
        else:
            missing_entity = 'claim number, member ID, or prescription ID'
        
        return {
            'action': 'ask_clarification',
            'confidence': 0.90,
            'reasoning': f'User asking for personal data but missing: {missing_entity}',
            'entities': extracted_entities,
            'clarification_prompt': f"I can help you with that. Could you provide your {missing_entity}?"
        }
    
    # DECISION 3: Truly general question → FAQ
    if is_truly_general:
        return {
            'action': 'search_faq',
            'confidence': 0.85,
            'reasoning': 'General knowledge question about policies/procedures',
            'entities': extracted_entities
        }
    
    # DECISION 4: Default to FAQ search (safer than guessing API)
    return {
        'action': 'search_faq',
        'confidence': 0.60,
        'reasoning': 'Unable to confidently classify - defaulting to general knowledge',
        'entities': extracted_entities
    }


def _get_simple_response(response_type: str) -> str:
    """Get simple conversational responses"""
    responses = {
        'greeting': "Hi! I'm here to help with your pharmacy benefits questions. How can I assist you today?",
        'acknowledgment': "You're welcome! Is there anything else I can help you with?",
        'farewell': "Goodbye! Feel free to come back if you have more questions."
    }
    return responses.get(response_type, "How can I help you today?")

