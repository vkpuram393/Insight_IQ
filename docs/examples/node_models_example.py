"""
Node Models Usage Examples

Demonstrates how to integrate the node result models into your
conversational agent system nodes.
"""

from typing import Dict, Any
import time
from core.node_models import (
    IntentResult,
    IntentComplexity,
    EntityExtractionResult,
    SafetyResult,
    SafetyCheckType,
    SafetyViolationType,
    ToolResult,
    ToolExecutionStatus,
    ContextResult,
    ConversationMessage,
    SessionFact,
    ClarificationResult,
    ClarificationType,
    ResponsePayload,
    ResponseType,
    ResponseSource,
    CacheResult,
    CacheStatus,
    ConfidenceCheckResult,
    ConfidenceCheckDecision,
    create_intent_result,
    create_safety_result,
    create_tool_result,
    create_response_payload,
)
from state.schema import AgentState

# Import generic serialization helpers
from utils.serialization import (
    to_dict,
    from_dict,
    to_dict_list,
    from_dict_list,
)


# ============================================================================
# Example 1: Using IntentResult in Intent Agent
# ============================================================================

async def intent_agent_node_enhanced(state: AgentState) -> Dict[str, Any]:
    """
    Enhanced intent agent using IntentResult model
    """
    from agents.intent_classifier import get_intent_classifier
    
    classifier = get_intent_classifier()
    start_time = time.time()
    
    # Classify intent
    result = classifier.classify(state["text"])
    processing_time_ms = (time.time() - start_time) * 1000
    
    # Extract entities (simplified example)
    entities = None
    if "CLM-" in state["text"].upper():
        # Extract claim number
        import re
        match = re.search(r'CLM-\d+', state["text"].upper())
        if match:
            entities = EntityExtractionResult(claim_number=match.group(0))
    
    # Create structured result
    intent_result = IntentResult(
        intent=result["intent"],
        confidence=result["confidence"],
        needs_clarification=result.get("needs_clarification", False),
        all_scores=result.get("all_scores", {}),
        top_candidates=result.get("top_candidates", []),
        is_simple=result.get("is_simple", False),
        is_complex=result.get("is_complex", False),
        entities=entities,
        reasoning=f"Classified using keyword matching with score {result['confidence']:.2f}",
        classification_method="keyword_matching",
        processing_time_ms=processing_time_ms
    )
    
    # Return structured data to state
    # Note: You can use to_dict() helper or model.model_dump() directly
    return {
        "intent": intent_result.intent,
        "confidence": intent_result.confidence,
        "entities": to_dict(intent_result.entities) if intent_result.entities else None,
        "needs_clarification": intent_result.needs_clarification,
        "metadata": {
            **state.get("metadata", {}),
            "intent_result": to_dict(intent_result)  # Generic helper
        }
    }


# ============================================================================
# Example 2: Using SafetyResult in Safety Nodes
# ============================================================================

async def safety_precheck_node_enhanced(state: AgentState) -> Dict[str, Any]:
    """
    Enhanced safety precheck using SafetyResult model
    """
    from core.config import settings
    
    if not settings.enable_safety_precheck:
        result = create_safety_result(
            check_type=SafetyCheckType.PRECHECK,
            passed=True,
            suggested_action="allow"
        )
        return {"safety_precheck_passed": True}
    
    text = state["text"].lower()
    start_time = time.time()
    
    # Check for harmful content
    harmful_keywords = {
        "self_harm": ["kill", "suicide"],
        "violence": ["bomb", "hurt"],
        "hate_speech": ["hate"]
    }
    
    for category, keywords in harmful_keywords.items():
        detected = [kw for kw in keywords if kw in text]
        if detected:
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = SafetyResult(
                check_type=SafetyCheckType.PRECHECK,
                passed=False,
                violation_type=SafetyViolationType(category),
                block_reason=f"Detected {category} related content",
                detected_keywords=detected,
                confidence_score=0.95,
                suggested_action="block",
                user_message="I cannot process that request.",
                processing_time_ms=processing_time_ms
            )
            
            return {
                "safety_precheck_passed": False,
                "safety_block_reason": result.block_reason,
                "response": result.user_message,
                "metadata": {
                    **state.get("metadata", {}),
                    "safety_result": to_dict(result)  # Generic helper
                }
            }
    
    # Passed safety check
    processing_time_ms = (time.time() - start_time) * 1000
    result = create_safety_result(
        check_type=SafetyCheckType.PRECHECK,
        passed=True,
        violation_type=SafetyViolationType.NONE,
        suggested_action="allow",
        processing_time_ms=processing_time_ms
    )
    
    return {
        "safety_precheck_passed": True,
        "metadata": {
            **state.get("metadata", {}),
            "safety_result": result.model_dump()
        }
    }


# ============================================================================
# Example 3: Using ToolResult in API Calls
# ============================================================================

async def call_claims_tool_node_enhanced(state: AgentState) -> Dict[str, Any]:
    """
    Enhanced claims tool node using ToolResult model
    """
    import asyncio
    
    intent = state["intent"]
    entities = state.get("entities", {})
    claim_number = entities.get("claim_number", "12345")
    
    start_time = time.time()
    
    try:
        # Simulate API call
        await asyncio.sleep(0.2)
        
        # Mock API response
        if intent == "claim_status":
            data = {
                "claim_id": claim_number,
                "status": "processing",
                "submitted_date": "2025-01-10",
                "expected_completion": "5-7 business days"
            }
        else:
            data = {}
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        result = ToolResult(
            tool_name="claims_api",
            status=ToolExecutionStatus.SUCCESS,
            data=data,
            execution_time_ms=execution_time_ms,
            api_endpoint=f"/api/v1/claims/{claim_number}",
            http_status_code=200,
            is_retryable=False,
            from_cache=False
        )
        
        return {
            "tool_results": result.data,
            "metadata": {
                **state.get("metadata", {}),
                "tool_result": to_dict(result),  # Generic helper
                "tools_used": [result.tool_name]
            }
        }
        
    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        
        result = ToolResult(
            tool_name="claims_api",
            status=ToolExecutionStatus.FAILURE,
            data={},
            error_message=str(e),
            error_code="API_ERROR",
            execution_time_ms=execution_time_ms,
            is_retryable=True
        )
        
        return {
            "tool_results": {},
            "error": f"Tool execution failed: {e}",
            "metadata": {
                **state.get("metadata", {}),
                "tool_result": result.model_dump()
            }
        }


# ============================================================================
# Example 4: Using ContextResult in Context Building
# ============================================================================

async def build_context_node_enhanced(state: AgentState) -> Dict[str, Any]:
    """
    Enhanced context building using ContextResult model
    """
    from memory import MemoryStoreFactory
    from core.config import settings
    
    memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
    session_id = state["session_id"]
    
    start_time = time.time()
    
    # Get conversation history
    history_data = await memory_store.get_session_history(session_id)
    facts_data = await memory_store.get_session_facts(session_id)
    
    retrieval_time_ms = (time.time() - start_time) * 1000
    
    # Convert to structured models
    conversation_history = [
        ConversationMessage(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            timestamp=msg.get("timestamp"),
            metadata=msg.get("metadata", {})
        )
        for msg in history_data
    ]
    
    relevant_facts = [
        SessionFact(
            fact_type=fact.get("fact_type", "unknown"),
            data=fact.get("data", {}),
            extracted_at=fact.get("extracted_at")
        )
        for fact in facts_data
    ]
    
    result = ContextResult(
        conversation_history=conversation_history,
        relevant_facts=relevant_facts,
        history_length=len(conversation_history),
        facts_count=len(relevant_facts),
        context_window_size=10,
        memory_source=settings.memory_store_type,
        retrieval_time_ms=retrieval_time_ms
    )
    
    return {
        "conversation_history": to_dict_list(result.conversation_history),  # Generic list helper
        "relevant_facts": to_dict_list(result.relevant_facts),  # Generic list helper
        "metadata": {
            **state.get("metadata", {}),
            "context_result": to_dict(result)
        }
    }


# ============================================================================
# Example 5: Using ClarificationResult in Clarification Node
# ============================================================================

async def clarification_node_enhanced(state: AgentState) -> Dict[str, Any]:
    """
    Enhanced clarification using ClarificationResult model
    """
    intent = state.get("intent", "unknown")
    confidence = state.get("confidence", 0.0)
    entities = state.get("entities", {})
    
    # Determine clarification type
    missing_entities = []
    if intent == "claim_status" and not entities.get("claim_number"):
        missing_entities.append("claim_number")
        clarification_type = ClarificationType.MISSING_ENTITY
        question = "Could you provide your claim number?"
    elif confidence < 0.5:
        clarification_type = ClarificationType.LOW_CONFIDENCE
        question = "I'm not sure I understand. Are you asking about a claim?"
    else:
        clarification_type = ClarificationType.AMBIGUOUS_INTENT
        question = "Could you provide more details about your request?"
    
    result = ClarificationResult(
        needs_clarification=True,
        clarifying_question=question,
        clarification_type=clarification_type,
        original_intent=intent,
        missing_entities=missing_entities,
        question_template="Could you provide your {entity_name}?",
        confidence_score=0.90,
        expected_entity_types=missing_entities,
        current_attempt=1
    )
    
    return {
        "needs_clarification": True,
        "clarifying_question": result.clarifying_question,
        "response": result.clarifying_question,
        "metadata": {
            **state.get("metadata", {}),
            "clarification_result": result.model_dump()
        }
    }


# ============================================================================
# Example 6: Using ResponsePayload in Response Agent
# ============================================================================

async def response_agent_node_enhanced(state: AgentState) -> Dict[str, Any]:
    """
    Enhanced response generation using ResponsePayload model
    """
    from core.config import settings
    
    intent = state.get("intent", "unknown")
    tool_results = state.get("tool_results", {})
    
    start_time = time.time()
    
    # Generate response (simplified)
    if intent == "claim_status" and tool_results:
        response_text = f"Your claim #{tool_results.get('claim_id', 'unknown')} is currently {tool_results.get('status', 'being processed')}."
        response_type = ResponseType.DIRECT_ANSWER
    else:
        response_text = "I'm not sure how to help with that."
        response_type = ResponseType.FALLBACK
    
    generation_time_ms = (time.time() - start_time) * 1000
    
    result = ResponsePayload(
        response=response_text,
        response_type=response_type,
        response_source=ResponseSource.LLM_GENERATED if not settings.use_mock_llm else ResponseSource.TEMPLATE,
        llm_model=settings.llm_model,
        temperature=settings.llm_temperature,
        input_tokens=150,
        output_tokens=45,
        total_tokens=195,
        estimated_cost_usd=0.0001,
        confidence_score=0.92,
        completeness_score=0.95,
        context_used=bool(state.get("conversation_history")),
        tools_used=state.get("metadata", {}).get("tools_used", []),
        generation_time_ms=generation_time_ms,
        safety_checked=state.get("safety_postcheck_passed", False),
        safety_passed=state.get("safety_postcheck_passed", True)
    )
    
    return {
        "response": result.response,
        "metadata": {
            **state.get("metadata", {}),
            "response_payload": result.model_dump(),
            "tokens_used": result.total_tokens,
            "estimated_cost": result.estimated_cost_usd
        }
    }


# ============================================================================
# Example 7: Using CacheResult in Cache Nodes
# ============================================================================

async def check_cache_node_enhanced(state: AgentState) -> Dict[str, Any]:
    """
    Enhanced cache check using CacheResult model
    """
    from core.config import settings
    from memory import MemoryStoreFactory
    import hashlib
    import json
    
    if not settings.enable_semantic_cache:
        result = CacheResult(
            cache_hit=False,
            status=CacheStatus.DISABLED
        )
        return {"cache_hit": False}
    
    # Generate cache key
    cache_key = f"cache:{hashlib.md5(state['text'].lower().encode()).hexdigest()}"
    
    memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
    start_time = time.time()
    
    cached_value = await memory_store.get(cache_key)
    retrieval_time_ms = (time.time() - start_time) * 1000
    
    if cached_value:
        cached = json.loads(cached_value) if isinstance(cached_value, str) else cached_value
        
        result = CacheResult(
            cache_hit=True,
            status=CacheStatus.HIT,
            cached_response=cached.get("response"),
            cached_intent=cached.get("intent"),
            cached_confidence=cached.get("confidence"),
            cache_key=cache_key,
            ttl_seconds=3600,
            retrieval_time_ms=retrieval_time_ms,
            cache_backend=settings.memory_store_type
        )
        
        return {
            "response": result.cached_response,
            "intent": result.cached_intent,
            "confidence": result.cached_confidence,
            "cache_hit": True,
            "metadata": {
                **state.get("metadata", {}),
                "cache_result": result.model_dump()
            }
        }
    
    result = CacheResult(
        cache_hit=False,
        status=CacheStatus.MISS,
        cache_key=cache_key,
        retrieval_time_ms=retrieval_time_ms,
        cache_backend=settings.memory_store_type
    )
    
    return {
        "cache_hit": False,
        "metadata": {
            **state.get("metadata", {}),
            "cache_result": result.model_dump()
        }
    }


# ============================================================================
# Example 8: Using ConfidenceCheckResult in Routing
# ============================================================================

def confidence_check_router_enhanced(state: AgentState) -> str:
    """
    Enhanced confidence router using ConfidenceCheckResult model
    """
    from core.config import settings
    
    intent = state.get("intent")
    entities = state.get("entities") or {}
    confidence = state.get("confidence", 0.0)
    threshold = settings.confidence_threshold
    
    # Check entity completeness
    required_entities = []
    missing_entities = []
    
    if intent == "claim_rejection_reason":
        required_entities = ["claim_number"]
        if not entities.get("claim_number"):
            missing_entities.append("claim_number")
    
    entities_complete = len(missing_entities) == 0
    passed_threshold = confidence >= threshold
    
    # Make decision
    if not entities_complete:
        decision = ConfidenceCheckDecision.CLARIFY
        decision_reason = f"Missing required entities: {', '.join(missing_entities)}"
        next_node = "clarification"
    elif not passed_threshold:
        decision = ConfidenceCheckDecision.CLARIFY
        decision_reason = f"Confidence {confidence:.2f} below threshold {threshold:.2f}"
        next_node = "clarification"
    else:
        decision = ConfidenceCheckDecision.PROCEED
        decision_reason = f"Confidence {confidence:.2f} meets threshold"
        next_node = "tool_call"
    
    result = ConfidenceCheckResult(
        decision=decision,
        confidence=confidence,
        threshold=threshold,
        passed_threshold=passed_threshold,
        required_entities=required_entities,
        missing_entities=missing_entities,
        entities_complete=entities_complete,
        decision_reason=decision_reason,
        next_node=next_node,
        intent_checked=intent
    )
    
    # Store result in state metadata
    state["metadata"] = {
        **state.get("metadata", {}),
        "confidence_check_result": result.model_dump()
    }
    
    return next_node


# ============================================================================
# Example 9: Complete Node Integration
# ============================================================================

async def complete_enhanced_node_example(state: AgentState) -> Dict[str, Any]:
    """
    Example showing complete integration with multiple model types
    """
    from core.errors import create_internal_error
    
    try:
        # 1. Intent classification
        intent_result = await intent_agent_node_enhanced(state)
        
        # 2. Safety check
        safety_result = await safety_precheck_node_enhanced(state)
        if not safety_result.get("safety_precheck_passed"):
            return safety_result
        
        # 3. Tool call
        tool_result = await call_claims_tool_node_enhanced(state)
        
        # 4. Response generation
        response_result = await response_agent_node_enhanced(state)
        
        # Combine all results
        return {
            **intent_result,
            **tool_result,
            **response_result,
            "metadata": {
                "node_results": {
                    "intent": intent_result.get("metadata", {}).get("intent_result"),
                    "tool": tool_result.get("metadata", {}).get("tool_result"),
                    "response": response_result.get("metadata", {}).get("response_payload")
                }
            }
        }
        
    except Exception as e:
        error = create_internal_error(
            error_message=str(e),
            session_id=state.get("session_id"),
            node_name="complete_enhanced_node"
        )
        return {"error": error.model_dump_json()}

