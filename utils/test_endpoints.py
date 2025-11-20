"""
Utility Test Endpoints

Individual endpoints for testing each component independently.
Each developer can test their functionality without running the full application.

Endpoints:
- /utils/test-intent - Test intent classifier
- /utils/test-intent-agent - Test full intent agent node
- /utils/test-cache - Test in-memory cache
- /utils/test-persistence - Test SQLite persistence
- /utils/test-memory-store - Test memory store operations
- /utils/test-session-history - Test session management
- /utils/test-context-building - Test context retrieval
- /utils/test-orchestrator - Test orchestrator node normalization
- /utils/test-safety-precheck - Test safety precheck
- /utils/test-safety-postcheck - Test safety postcheck
- /utils/test-confidence-checker - Test confidence scoring and routing
- /utils/test-clarification - Test clarification logic
- /utils/test-claims-api - Test claims API orchestrator
- /utils/test-response-agent - Test response generation agent (Mock/Real LLM)
- /utils/test-exception-handling - Test exception handling across nodes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid
import time
from datetime import datetime, timezone

from agents.intent_classifier import classify_intent
from agents.intent_agent import intent_agent_node
from agents.response_agent import response_agent_node
from nodes.safety import (
    safety_precheck_node,
    response_safety_pii_precheck_node,
    response_safety_pii_postcheck_node
)
from nodes.confidence import confidence_check_router
from nodes.clarification import clarification_node
from nodes.context import build_context_node, update_memory_node
from nodes.cache import check_cache_node, cache_response_node
from tools.claims_api import call_claims_tool_node
from memory import MemoryStoreFactory
from persistence import PersistenceStoreFactory, EventType
from core.config import settings
from core.logger import get_logger
from core.telemetry import log_event, log_request_response
from state.schema import AgentState

router = APIRouter(prefix="/utils", tags=["Testing Utils"])
logger = get_logger(__name__)

# ==================== Request/Response Models ====================

class IntentTestRequest(BaseModel):
    text: str
    user_info: Optional[Dict[str, Any]] = None

class IntentTestResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    reasoning: Optional[str] = None
    timestamp: str

class CacheTestRequest(BaseModel):
    key: str
    value: Optional[Dict[str, Any]] = None
    ttl_seconds: Optional[int] = 3600

class CacheTestResponse(BaseModel):
    operation: str
    key: str
    value: Optional[Any] = None
    success: bool
    timestamp: str

class PersistenceTestRequest(BaseModel):
    event_type: str
    session_id: str
    data: Dict[str, Any]

class SessionTestRequest(BaseModel):
    session_id: str
    role: str
    content: str

class SafetyTestRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

class ContextTestRequest(BaseModel):
    session_id: str
    text: str

class OrchestratorTestRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None

# ==================== Intent Classifier Tests ====================

@router.post("/test-intent", response_model=IntentTestResponse)
async def test_intent_classifier(request: IntentTestRequest):
    """
    Test the intent classifier independently

    Tests: agents/intent_classifier.py
    """
    try:
        result = await classify_intent(
            text=request.text,
            user_info=request.user_info or {}
        )

        return IntentTestResponse(
            text=request.text,
            intent=result.get("intent", "unknown"),
            confidence=result.get("confidence", 0.0),
            reasoning=result.get("reasoning"),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Intent classifier test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-intent-agent")
async def test_intent_agent(request: IntentTestRequest):
    """
    Test the full intent agent node

    Tests: agents/intent_agent.py
    """
    try:
        state = AgentState(
            text=request.text,
            session_id=str(uuid.uuid4()),
            user_info=request.user_info or {},
            uuid=None,
            domain=None,
            conversation_history=[],
            relevant_facts=[]
        )

        result = await intent_agent_node(state)

        return {
            "input": request.text,
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "entities": result.get("entities"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Intent agent test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Cache Tests ====================

@router.post("/test-cache", response_model=CacheTestResponse)
async def test_cache_operations(request: CacheTestRequest):
    """
    Test in-memory cache operations

    Tests: nodes/cache.py, memory/inmemory_store.py
    """
    try:
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)

        if request.value is not None:
            # SET operation
            success = await memory_store.set(
                request.key,
                request.value,
                ttl_seconds=request.ttl_seconds
            )
            return CacheTestResponse(
                operation="set",
                key=request.key,
                value=request.value,
                success=success,
                timestamp=datetime.now().isoformat()
            )
        else:
            # GET operation
            value = await memory_store.get(request.key)
            return CacheTestResponse(
                operation="get",
                key=request.key,
                value=value,
                success=value is not None,
                timestamp=datetime.now().isoformat()
            )
    except Exception as e:
        logger.error(f"Cache test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/test-cache/{key}")
async def test_cache_delete(key: str):
    """Delete a cache entry"""
    try:
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        success = await memory_store.delete(key)
        return {
            "operation": "delete",
            "key": key,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache delete test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-cache/{key}")
async def test_cache_get(key: str):
    """Get a cache entry"""
    try:
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        value = await memory_store.get(key)
        return {
            "operation": "get",
            "key": key,
            "value": value,
            "exists": value is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache get test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Persistence Tests ====================

@router.post("/test-persistence")
async def test_persistence_logging(request: PersistenceTestRequest):
    """
    Test SQLite persistence/telemetry logging

    Tests: persistence/sqlite_store.py
    """
    try:
        event_type = EventType[request.event_type.upper()]
        event_id = await log_event(
            event_type=event_type,
            session_id=request.session_id,
            data=request.data
        )

        return {
            "event_id": event_id,
            "event_type": request.event_type,
            "session_id": request.session_id,
            "success": event_id is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Persistence test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-persistence/events/{session_id}")
async def test_get_session_events(session_id: str):
    """Get all events for a session from persistence"""
    try:
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        events = await persistence_store.get_session_events(session_id)

        return {
            "session_id": session_id,
            "event_count": len(events),
            "events": events,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Get session events test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Session Memory Tests ====================

@router.post("/test-session-history")
async def test_session_history(request: SessionTestRequest):
    """
    Test session history storage and retrieval

    Tests: nodes/context.py, memory/inmemory_store.py
    """
    try:
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)

        # Add message to session
        await memory_store.append_to_session(
            session_id=request.session_id,
            role=request.role,
            content=request.content
        )

        # Retrieve history
        history = await memory_store.get_session_history(request.session_id)

        return {
            "session_id": request.session_id,
            "message_added": {"role": request.role, "content": request.content},
            "total_messages": len(history),
            "history": history,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Session history test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test-session-history/{session_id}")
async def test_get_session_history(session_id: str):
    """Get session history"""
    try:
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        history = await memory_store.get_session_history(session_id)
        facts = await memory_store.get_session_facts(session_id)

        return {
            "session_id": session_id,
            "message_count": len(history),
            "history": history,
            "facts": facts,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Get session history test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Context Building Tests ====================

class ContextBuilderTestRequest(BaseModel):
    """Request for context builder test endpoint - simulates what confidence checker passes"""
    text: str
    intent: str
    confidence: float
    entities: Optional[Dict[str, Any]] = None
    slots: Optional[Dict[str, Any]] = None
    required_slots: Optional[List[str]] = None
    missing_slots: Optional[List[str]] = None
    session_id: Optional[str] = None
    uuid: Optional[str] = None
    domain: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None

@router.post("/test-context-building")
async def test_context_building(request: ContextBuilderTestRequest):
    """
    Test context building node with full payload from confidence checker

    Tests: nodes/context.py - build_context_node
    
    This endpoint simulates what confidence checker passes to context builder:
    - Intent, confidence, entities, slots from intent classifier
    - Required slots and missing slots
    - Session ID, UUID, domain, user info
    
    Returns:
    - Conversation history (last N messages, configurable)
    - Relevant facts from session
    - Extracted slots from conversation history
    - Complete planner_context object for planner/executor
    """
    try:
        from nodes.context import build_context_node
        
        session_id = request.session_id or str(uuid.uuid4())
        
        # Create state as confidence checker would pass it
        state = AgentState(
            text=request.text,
            session_id=session_id,
            user_info=request.user_info or {},
            uuid=request.uuid,
            domain=request.domain,
            intent=request.intent,
            confidence=request.confidence,
            entities=request.entities or {},
            slots=request.slots or {},
            required_slots=request.required_slots or [],
            missing_slots=request.missing_slots or [],
            needs_clarification=False,
            clarifying_question=None,
            conversation_history=[],
            relevant_facts=[],
            tool_results=None,
            safety_precheck_passed=True,
            safety_postcheck_passed=True,
            safety_block_reason=None,
            response="",
            metadata={},
            cache_hit=False,
            error=None,
            messages=[],
            extracted_slots=None,
            planner_context=None
        )

        result = await build_context_node(state)

        return {
            "session_id": session_id,
            "conversation_history": result.get("conversation_history", []),
            "relevant_facts": result.get("relevant_facts", []),
            "extracted_slots": result.get("extracted_slots", {}),
            "planner_context": result.get("planner_context", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Context building test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Orchestrator Tests ====================

@router.post("/test-orchestrator")
async def test_orchestrator(request: OrchestratorTestRequest):
    """
    Test orchestrator node normalization
    
    Tests: nodes/orchestrator.py
    
    This endpoint tests the orchestrator's input normalization logic without
    running the full LangGraph workflow. Useful for verifying text processing.
    
    Example cURL command:
    ```bash
    curl -X POST "http://localhost:8000/utils/test-orchestrator" \
      -H "Content-Type: application/json" \
      -d '{
        "text": "  Why was MY claim REJECTED?  ",
        "session_id": "test-orch-1"
      }'
    ```
    
    Example PowerShell command (Windows):
    ```powershell
    Invoke-RestMethod -Uri "http://localhost:8000/utils/test-orchestrator" `
      -Method POST `
      -ContentType "application/json" `
      -Body '{"text": "  Why was MY claim REJECTED?  ", "session_id": "ps-test-1"}'
    ```
    """
    try:
        from nodes.orchestrator import orchestrator_node
        
        logger.info(f"🧪 TEST ORCHESTRATOR: Received test request with {len(request.text)} chars")
        
        # Create initial state (mimics what comes from api/routes.py)
        # Use create_initial_state helper for consistency with orchestrator requirements
        from state.schema import create_initial_state
        
        state = create_initial_state(
            text=request.text,
            session_id=request.session_id or str(uuid.uuid4()),
            user_info=request.user_info or {}
        )
        
        # Run orchestrator node
        result = await orchestrator_node(state)
        
        # Extract orchestrator metadata
        orchestrator_meta = result.get("metadata", {}).get("orchestrator_metadata", {})
        original_text = result.get("metadata", {}).get("original_text", request.text)
        
        return {
            "success": not orchestrator_meta.get("error", False),
            "uuid": result.get("uuid"),
            "input": {
                "raw_text": request.text,
                "length": len(request.text),
                "session_id": request.session_id or str(uuid.uuid4())
            },
            "output": {
                "normalized_text": result.get("text", ""),
                "original_text": original_text,
                "length": len(result.get("text", "")),
                "uuid": result.get("uuid"),
                "metadata": orchestrator_meta
            },
            "comparison": {
                "chars_removed": len(request.text) - len(result.get("text", "")),
                "was_normalized": orchestrator_meta.get("normalization_applied", False)
            },
            "error": {
                "occurred": orchestrator_meta.get("error", False),
                "type": orchestrator_meta.get("error_type"),
                "message": orchestrator_meta.get("error_message")
            } if orchestrator_meta.get("error") else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"🧪 TEST ORCHESTRATOR ERROR: {str(e)}")
        import traceback
        logger.error(f"🔍 STACK TRACE:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Orchestrator test failed: {str(e)}"
        )



# ==================== Claims API Tests ====================

class ClaimsAPITestRequest(BaseModel):
    """Request model for claims API test endpoint"""
    text: str = Field(..., description="User's original query text")
    intent: str = Field(..., description="Detected intent (e.g., find_claim, claim_details)")
    entities: Dict[str, Any] = Field(..., description="Extracted entities with raw_entities support")
    session_id: Optional[str] = Field(None, description="Session identifier")
    uuid: Optional[str] = Field(None, description="Request UUID")
    domain: Optional[str] = Field(default="claims", description="Domain context")
    user_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User metadata")

@router.post("/test-claims-api")
async def test_claims_api(request: ClaimsAPITestRequest):
    """
    Test Claims API Orchestrator Node
    
    Tests: tools/claims_api.py - call_claims_tool_node()
    
    This endpoint tests the complete claims API orchestration flow:
    - Entity normalization (snake_case → camelCase)
    - Dynamic API matching based on intent + entities
    - Request body building
    - External API call with retry logic
    - ToolResult creation and state update
    
    The endpoint validates that tool_results properly updates AgentState
    for downstream LangGraph node consumption.
    
    Example requests:
    
    1. Find Claim by ID:
    ```json
    {
      "text": "Show me claim 253152732536005",
      "intent": "find_claim",
      "entities": {
        "raw_entities": {
          "claim_id": "253152732536005"
        }
      }
    }
    ```
    
    2. Get Claim Details:
    ```json
    {
      "text": "What are the details for claim 12345 sequence 1?",
      "intent": "claim_details",
      "entities": {
        "claim_number": "253152732536005",
        "raw_entities": {
          "claim_sequence": "1"
        }
      }
    }
    ```
    
    3. Test Missing Entities Error:
    ```json
    {
      "text": "Show me claim details",
      "intent": "claim_details",
      "entities": {
        "claim_number": "12345"
      }
    }
    ```
    
    cURL Example:
    ```bash
    curl -X POST "http://localhost:8000/utils/test-claims-api" \
      -H "Content-Type: application/json" \
      -d '{
        "text": "Show claim 253152732536005",
        "intent": "find_claim",
        "entities": {"raw_entities": {"claim_id": "253152732536005"}}
      }'
    ```
    
    PowerShell Example:
    ```powershell
    $body = @{
        text = "Show claim 253152732536005"
        intent = "find_claim"
        entities = @{
            raw_entities = @{
                claim_id = "253152732536005"
            }
        }
    } | ConvertTo-Json -Depth 5
    
    Invoke-RestMethod -Uri "http://localhost:8000/utils/test-claims-api" `
      -Method POST `
      -ContentType "application/json" `
      -Body $body
    ```
    """
    try:
        from tools.claims_api import call_claims_tool_node
        
        session_id = request.session_id or str(uuid.uuid4())
        request_uuid = request.uuid or str(uuid.uuid4())
        
        logger.info(f"🧪 TEST CLAIMS API: intent={request.intent}, entities={request.entities}")
        
        # Create complete AgentState matching production flow
        state = AgentState(
            text=request.text,
            session_id=session_id,
            user_info=request.user_info,
            uuid=request_uuid,
            domain=request.domain,
            intent=request.intent,
            confidence=0.9,  # High confidence for direct test
            entities=request.entities,
            slots=None,
            required_slots=None,
            missing_slots=None,
            needs_clarification=False,
            clarifying_question=None,
            conversation_history=[],
            relevant_facts=[],
            extracted_slots=None,
            planner_context=None,
            tool_results=None,
            safety_precheck_passed=True,
            safety_postcheck_passed=False,
            safety_block_reason=None,
            response="",
            metadata={
                "request_id": request_uuid,
                "user_id": request.user_info.get("user_id") if request.user_info else None
            },
            cache_hit=False,
            error=None,
            messages=[]
        )
        
        # Log test start to telemetry database
        if settings.enable_telemetry:
            persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
            await persistence_store.log_audit(
                session_id=session_id,
                request_id=request_uuid,
                user_id=request.user_info.get("user_id") if request.user_info else None,
                node_name="test_claims_api",
                event_type="test_start",
                data={
                    "intent": request.intent,
                    "entities": request.entities,
                    "test_type": "claims_api_orchestrator"
                }
            )
        
        # Call the claims tool node
        logger.info("🔄 Calling call_claims_tool_node...")
        start_time = time.time()
        result = await call_claims_tool_node(state)
        elapsed_ms = (time.time() - start_time) * 1000.0
        
        # Extract tool_results from result
        tool_results = result.get("tool_results") if isinstance(result, dict) and "tool_results" in result else result
        
        # Determine success
        is_success = tool_results.get("status") == "success" if isinstance(tool_results, dict) else False
        
        # Log test completion
        if settings.enable_telemetry:
            await persistence_store.log_audit(
                session_id=session_id,
                request_id=request_uuid,
                user_id=request.user_info.get("user_id") if request.user_info else None,
                node_name="test_claims_api",
                event_type="test_complete",
                data={
                    "success": is_success,
                    "status": tool_results.get("status") if isinstance(tool_results, dict) else None,
                    "total_test_time_ms": elapsed_ms
                }
            )
        
        logger.info(f"✅ Test completed: success={is_success}, time={elapsed_ms:.2f}ms")
        
        # Build comprehensive response
        return {
            "test_metadata": {
                "session_id": session_id,
                "request_uuid": request_uuid,
                "test_duration_ms": elapsed_ms,
                "timestamp": datetime.now().isoformat()
            },
            "input": {
                "text": request.text,
                "intent": request.intent,
                "entities": request.entities,
                "domain": request.domain
            },
            "tool_execution": {
                "tool_name": tool_results.get("tool_name") if isinstance(tool_results, dict) else None,
                "status": tool_results.get("status") if isinstance(tool_results, dict) else None,
                "api_endpoint": tool_results.get("api_endpoint") if isinstance(tool_results, dict) else None,
                "execution_time_ms": tool_results.get("execution_time_ms") if isinstance(tool_results, dict) else None,
                "http_status_code": tool_results.get("http_status_code") if isinstance(tool_results, dict) else None,
                "retry_count": tool_results.get("retry_count") if isinstance(tool_results, dict) else None,
                "is_retryable": tool_results.get("is_retryable") if isinstance(tool_results, dict) else None
            },
            "result": {
                "success": is_success,
                "has_data": bool(tool_results.get("data")) if isinstance(tool_results, dict) else False,
                "data_preview": {
                    "keys": list(tool_results.get("data", {}).keys()) if isinstance(tool_results, dict) and tool_results.get("data") else [],
                    "total_count": tool_results.get("data", {}).get("totalCount") if isinstance(tool_results, dict) else None,
                    "message": tool_results.get("data", {}).get("message") if isinstance(tool_results, dict) else None
                },
                "error_message": tool_results.get("error_message") if isinstance(tool_results, dict) else None,
                "error_code": tool_results.get("error_code") if isinstance(tool_results, dict) else None
            },
            "agent_error": tool_results.get("agent_error") if isinstance(tool_results, dict) else None,
            "state_update": {
                "tool_results_field_updated": "tool_results" in result,
                "structure_correct": isinstance(result, dict) and "tool_results" in result,
                "ready_for_langgraph": "tool_results" in result,
                "note": "Result properly wrapped for AgentState.tool_results update" if "tool_results" in result else "Warning: Result not wrapped properly"
            },
            "full_tool_results": tool_results,
            "performance": {
                "total_time_ms": elapsed_ms,
                "api_time_ms": tool_results.get("execution_time_ms") if isinstance(tool_results, dict) else None,
                "overhead_ms": elapsed_ms - ((tool_results.get("execution_time_ms") or 0) if isinstance(tool_results, dict) else 0)
            }
        }
        
    except Exception as e:
        logger.error(f"🧪 TEST CLAIMS API ERROR: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logger.error(f"🔍 STACK TRACE:\n{tb}")
        
        # Log exception to database
        if settings.enable_telemetry:
            try:
                persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
                await persistence_store.log_exception(
                    session_id=session_id,
                    request_id=request_uuid,
                    user_id=request.user_info.get("user_id") if request.user_info else None,
                    node_name="test_claims_api",
                    error_code="E9001",
                    category="system",
                    severity="error",
                    message=f"Test endpoint exception: {str(e)}",
                    stacktrace=tb
                )
            except Exception as log_error:
                logger.error(f"Failed to log exception: {log_error}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": tb,
                "test": "claims_api_orchestrator"
            }
        )


# ==================== Safety Tests ====================

@router.post("/test-safety-precheck")
async def test_safety_precheck(request: SafetyTestRequest):
    """
    Test safety precheck node

    Tests: nodes/safety.py
    """
    try:
        state = AgentState(
            text=request.text,
            session_id=request.session_id or str(uuid.uuid4()),
            user_info={},
            uuid=None,
            domain=None,
            conversation_history=[],
            relevant_facts=[]
        )

        result = await safety_precheck_node(state)

        return {
            "text": request.text,
            "safety_precheck_passed": result.get("safety_precheck_passed"),
            "blocked_reason": result.get("blocked_reason"),
            "safety_score": result.get("safety_score"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Safety precheck test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-safety-postcheck")
async def test_safety_postcheck(request: SafetyTestRequest):
    """
    Test safety postcheck node

    Tests: nodes/safety.py
    """
    try:
        state = AgentState(
            text=request.text,
            session_id=request.session_id or str(uuid.uuid4()),
            user_info={},
            uuid=None,
            domain=None,
            conversation_history=[],
            relevant_facts=[],
            response="This is a test response"
        )

        result = await response_safety_pii_postcheck_node(state)

        return {
            "response": state.get("response", "This is a test response"),
            "safety_postcheck_passed": result.get("safety_postcheck_passed"),
            "blocked_reason": result.get("blocked_reason"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Safety postcheck test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Clarification Tests ====================

@router.post("/test-clarification")
async def test_clarification(request: IntentTestRequest):
    """
    Test clarification logic

    Tests: nodes/clarification.py
    """
    try:
        state = AgentState(
            text=request.text,
            session_id=str(uuid.uuid4()),
            user_info=request.user_info or {},
            uuid=None,
            domain=None,
            conversation_history=[],
            relevant_facts=[],
            intent="claim_status",
            entities={}
        )

        result = await clarification_node(state)

        return {
            "text": request.text,
            "needs_clarification": result.get("needs_clarification"),
            "clarifying_question": result.get("clarifying_question"),
            "missing_info": result.get("missing_info"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Clarification test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Confidence Checker Tests ====================

class ConfidenceCheckRequest(BaseModel):
    """Request for confidence checker endpoint"""
    text: str
    intent: str
    confidence: float
    entities: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    uuid: Optional[str] = None
    domain: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None

@router.post("/test-confidence-checker")
async def test_confidence_checker(request: ConfidenceCheckRequest):
    """
    Test confidence checker node

    Tests: nodes/confidence.py - confidence_checker_node
    
    This endpoint:
    - Takes intent classifier output (intent, confidence, entities)
    - Checks confidence against threshold from config
    - If low confidence: returns clarification object
    - If high confidence: calls context builder and logs to SQLite, returns context builder output
    """
    try:
        from nodes.confidence import confidence_checker_node
        from nodes.context import build_context_node
        
        session_id = request.session_id or str(uuid.uuid4())
        
        # Create state from intent classifier output
        state = AgentState(
            text=request.text,
            session_id=session_id,
            user_info=request.user_info or {},
            uuid=request.uuid,
            domain=request.domain or "claims",
            intent=request.intent,
            confidence=request.confidence,
            entities=request.entities or {},
            conversation_history=[],
            relevant_facts=[],
            needs_clarification=False,
            clarifying_question=None,
            tool_results=None,
            safety_precheck_passed=True,
            safety_postcheck_passed=False,
            safety_block_reason=None,
            response="",
            metadata={},
            cache_hit=False,
            error=None,
            messages=[]
        )
        
        # Call confidence checker node
        result = await confidence_checker_node(state)
        
        # Check if clarification needed
        if result.get("needs_clarification"):
            return {
                "decision": "clarification",
                "needs_clarification": True,
                "clarifying_question": result.get("clarifying_question"),
                "response": result.get("response"),
                "metadata": result.get("metadata", {}),
                "timestamp": datetime.now().isoformat()
            }
        
        # High confidence - call context builder
        if result.get("metadata", {}).get("confidence_check_passed"):
            context_builder_input = result.get("metadata", {}).get("context_builder_input")
            
            # Update state with context builder input
            state.update({
                "conversation_history": context_builder_input.get("chat_history", []) if context_builder_input else []
            })
            
            # Call context builder
            context_result = await build_context_node(state)
            
            # Log context builder output
            persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
            await persistence_store.log_audit(
                session_id=session_id,
                node_name="confidence_checker",
                event_type="context_builder_output",
                data=context_result,
                request_id=request.uuid,
                user_id=request.user_info.get("user_id") if request.user_info else None
            )
            
            return {
                "decision": "proceed",
                "confidence_check_passed": True,
                "context_builder_input": context_builder_input,
                "context_builder_output": context_result,
                "timestamp": datetime.now().isoformat()
            }
        
        # Error case
        return {
            "decision": "error",
            "error": result.get("error"),
            "metadata": result.get("metadata", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Confidence checker test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Response Agent Tests ====================

class ResponseAgentTestRequest(BaseModel):
    """Request model for response agent test endpoint"""
    text: str = Field(..., description="User's original query text")
    intent: str = Field(default="find_claim", description="Intent from intent classifier")
    confidence: float = Field(default=0.9, description="Confidence score")
    entities: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extracted entities")
    tool_results: Optional[Dict[str, Any]] = Field(None, description="Tool results from Claims API (ToolResult structure)")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="Conversation history")
    session_id: Optional[str] = Field(None, description="Session identifier")
    uuid: Optional[str] = Field(None, description="Request UUID")
    user_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User metadata")
    use_mock_llm: Optional[bool] = Field(None, description="Force mock LLM (None = use settings.use_mock_llm)")

def _validate_claim_data_in_response(response_text: str, tool_results: Optional[Dict[str, Any]]) -> bool:
    """
    Validate if response contains actual claim data from tool_results.
    
    Checks for presence of key claim-related information in the response:
    - Drug/product names
    - Member names
    - Claim numbers
    - Pricing information
    - Pharmacy names
    
    Args:
        response_text: Generated response from LLM
        tool_results: Tool results containing claim data
        
    Returns:
        bool: True if response contains claim data, False otherwise
    """
    if not tool_results or not response_text:
        return False
    
    # Extract claim data from tool_results
    data = tool_results.get("data", {})
    claims = data.get("claims", [])
    
    if not claims:
        return False
    
    # Check if any claim data appears in response
    response_upper = response_text.upper()
    
    for claim in claims:
        # Check drug name
        drug = claim.get("drug", {})
        if drug.get("productName") and drug["productName"].upper() in response_upper:
            return True
        
        # Check member info
        member = claim.get("member", {})
        if member.get("firstName") and member["firstName"].upper() in response_upper:
            return True
        if member.get("lastName") and member["lastName"].upper() in response_upper:
            return True
        if member.get("memberId") and member["memberId"] in response_text:
            return True
        
        # Check claim number
        claim_info = claim.get("claimInformation", {})
        if claim_info.get("claimNumber") and claim_info["claimNumber"] in response_text:
            return True
        
        # Check pharmacy
        prescription = claim.get("prescription", {})
        if prescription.get("pharmacyName") and prescription["pharmacyName"].upper() in response_upper:
            return True
        
        # Check pricing (any mention of dollar amounts)
        pricing = claim.get("pricing", {})
        if pricing.get("patientPay") and pricing["patientPay"] in response_text:
            return True
    
    return False

@router.post("/test-response-agent")
async def test_response_agent(request: ResponseAgentTestRequest):
    """
    Test Response Generation Agent (Node)
    
    Tests: agents/response_agent.py - response_agent_node()
    
    This endpoint tests the complete response agent flow:
    - System prompt loading (pharmacy claims assistant)
    - User prompt building with ChatPromptTemplate
    - Tool results formatting (ToolResult structure from Claims API)
    - Conversation history formatting
    - LLM invocation (Mock or Real Gemini)
    - Error handling and telemetry logging
    
    The endpoint validates that the response agent:
    - Follows llm_connection.py patterns for Gemini calls
    - Uses ChatPromptTemplate for structured prompts
    - Properly formats ToolResult data for LLM
    - Handles both Mock and Real LLM modes
    - Logs telemetry correctly
    - Returns graceful error states
    
    Example requests:
    
    1. Test with Dummy Claim Data (Mock LLM or Real LLM):
    ```json
    {
      "text": "Show me my claim details",
      "intent": "find_claim",
      "confidence": 0.95,
      "entities": {"claim_number": "CLM1234567890001"},
      "tool_results": {
        "tool_name": "get_claim_list",
        "status": "success",
        "data": {
          "claims": [{
            "claimInformation": {
              "claimNumber": "CLM1234567890001",
              "claimStatus": "P",
              "claimStatusDescription": "Paid",
              "fillDate": "2025-05-01",
              "quantity": "10.0",
              "daysSupplied": "90"
            },
            "drug": {
              "productName": "ATORVASTATIN",
              "productNdc": "12345-678-90"
            },
            "pricing": {
              "patientPay": "10.00"
            },
            "member": {
              "firstName": "JOHN",
              "lastName": "DOE",
              "memberId": "MBR123456789"
            },
            "prescription": {
              "pharmacyName": "MAIN STREET PHARMACY"
            }
          }],
          "totalCount": 1
        }
      }
    }
    ```
    
    2. Test with Rejected Claim:
    ```json
    {
      "text": "Why was my claim rejected?",
      "intent": "rejection_reasons",
      "tool_results": {
        "status": "success",
        "data": {
          "claims": [{
            "claimInformation": {
              "claimStatus": "R",
              "claimStatusDescription": "Rejected"
            },
            "messages": {
              "rejectCodes": ["79"],
              "messages": ["Refill Too Soon"]
            }
          }]
        }
      }
    }
    ```
    
    3. Test with Conversation History:
    ```json
    {
      "text": "What was the copay?",
      "intent": "claim_details",
      "conversation_history": [
        {"role": "user", "content": "Show me claim 12345"},
        {"role": "assistant", "content": "Here's claim 12345..."}
      ],
      "tool_results": {
        "data": {
          "claims": [{
            "pricing": {"patientPay": "25.00"}
          }]
        }
      }
    }
    ```
    
    4. Test Error Handling:
    ```json
    {
      "text": "Test error handling",
      "intent": "unknown",
      "tool_results": null
    }
    ```
    
    cURL Example:
    ```bash
    curl -X POST "http://localhost:8000/utils/test-response-agent" \
      -H "Content-Type: application/json" \
      -d '{
        "text": "Show claim details",
        "intent": "find_claim",
        "tool_results": {
          "status": "success",
          "data": {"claims": [{"claimInformation": {"claimNumber": "CLM1234567890001"}}]}
        }
      }'
    ```
    
    PowerShell Example:
    ```powershell
    $body = @{
        text = "Show claim CLM1234567890001"
        intent = "find_claim"
        tool_results = @{
            status = "success"
            data = @{
                claims = @(
                    @{
                        claimInformation = @{
                            claimNumber = "CLM1234567890001"
                            claimStatus = "P"
                            fillDate = "2025-05-01"
                        }
                        drug = @{productName = "ATORVASTATIN"}
                        pricing = @{patientPay = "10.00"}
                    }
                )
            }
        }
    } | ConvertTo-Json -Depth 10
    
    Invoke-RestMethod -Uri "http://localhost:8000/utils/test-response-agent" `
      -Method POST `
      -ContentType "application/json" `
      -Body $body
    ```
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        request_uuid = request.uuid or str(uuid.uuid4())
        
        logger.info(f"🧪 TEST RESPONSE AGENT: intent={request.intent}, has_tool_results={request.tool_results is not None}")
        
        # Override USE_MOCK_LLM if specified
        original_use_mock = settings.use_mock_llm
        if request.use_mock_llm is not None:
            settings.use_mock_llm = request.use_mock_llm
            logger.info(f"🔧 Overriding USE_MOCK_LLM: {original_use_mock} → {request.use_mock_llm}")
        
        # Create complete AgentState matching production flow
        state = AgentState(
            text=request.text,
            session_id=session_id,
            user_info=request.user_info,
            uuid=request_uuid,
            domain="claims",
            intent=request.intent,
            confidence=request.confidence,
            entities=request.entities,
            slots=None,
            required_slots=None,
            missing_slots=None,
            needs_clarification=False,
            clarifying_question=None,
            conversation_history=request.conversation_history,
            relevant_facts=[],
            extracted_slots=None,
            planner_context=None,
            tool_results=request.tool_results,
            safety_precheck_passed=True,
            safety_postcheck_passed=False,
            safety_block_reason=None,
            response="",
            metadata={
                "request_id": request_uuid,
                "user_id": request.user_info.get("user_id") if request.user_info else None
            },
            cache_hit=False,
            error=None,
            messages=[]
        )
        
        # Log test start to telemetry database
        if settings.enable_telemetry:
            persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
            await persistence_store.log_audit(
                session_id=session_id,
                request_id=request_uuid,
                user_id=request.user_info.get("user_id") if request.user_info else None,
                node_name="test_response_agent",
                event_type="test_start",
                data={
                    "intent": request.intent,
                    "has_tool_results": request.tool_results is not None,
                    "has_conversation_history": len(request.conversation_history) > 0,
                    "use_mock_llm": settings.use_mock_llm,
                    "test_type": "response_agent_node"
                }
            )
        
        # Call the response agent node
        logger.info("🔄 Calling response_agent_node...")
        start_time = time.time()
        result = await response_agent_node(state)
        elapsed_ms = (time.time() - start_time) * 1000.0
        
        # Restore original setting
        if request.use_mock_llm is not None:
            settings.use_mock_llm = original_use_mock
            logger.info(f"🔧 Restored USE_MOCK_LLM: {original_use_mock}")
        
        # Extract response
        response_text = result.get("response", "")
        error_occurred = result.get("error") is not None
        
        # Log test completion
        if settings.enable_telemetry:
            await persistence_store.log_audit(
                session_id=session_id,
                request_id=request_uuid,
                user_id=request.user_info.get("user_id") if request.user_info else None,
                node_name="test_response_agent",
                event_type="test_complete",
                data={
                    "success": not error_occurred,
                    "response_length": len(response_text),
                    "total_test_time_ms": elapsed_ms,
                    "error": result.get("error")
                }
            )
        
        logger.info(f"✅ Test completed: success={not error_occurred}, response_length={len(response_text)}, time={elapsed_ms:.2f}ms")
        
        # Build comprehensive response
        return {
            "test_metadata": {
                "session_id": session_id,
                "request_uuid": request_uuid,
                "test_duration_ms": elapsed_ms,
                "use_mock_llm": original_use_mock if request.use_mock_llm is None else request.use_mock_llm,
                "timestamp": datetime.now().isoformat()
            },
            "input": {
                "text": request.text,
                "intent": request.intent,
                "confidence": request.confidence,
                "entities": request.entities,
                "has_tool_results": request.tool_results is not None,
                "tool_results_keys": list(request.tool_results.keys()) if request.tool_results else [],
                "conversation_history_length": len(request.conversation_history),
                "conversation_history": request.conversation_history
            },
            "output": {
                "success": not error_occurred,
                "response": response_text,
                "response_length": len(response_text),
                "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                "error": result.get("error"),
                "metadata": result.get("metadata", {})
            },
            "validation": {
                "response_generated": len(response_text) > 0,
                "follows_format": any(keyword in response_text for keyword in ["SUMMARY", "DRUG", "MEMBER", "FINANCIAL", "REJECTION"]) if not error_occurred else None,
                "contains_claim_data": _validate_claim_data_in_response(response_text, request.tool_results) if not error_occurred else None,
                "appropriate_length": 50 < len(response_text) < 5000 if not error_occurred else None,
                "is_mock_response": "MOCK" in response_text.upper()
            },
            "performance": {
                "total_time_ms": elapsed_ms,
                "avg_chars_per_sec": len(response_text) / (elapsed_ms / 1000) if elapsed_ms > 0 and not error_occurred else None
            },
            "agent_state_update": {
                "response_field_updated": "response" in result,
                "error_field_present": "error" in result,
                "metadata_present": "metadata" in result,
                "structure_correct": isinstance(result, dict) and "response" in result
            }
        }
        
    except Exception as e:
        logger.error(f"🧪 TEST RESPONSE AGENT ERROR: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logger.error(f"🔍 STACK TRACE:\n{tb}")
        
        # Restore original setting if it was changed
        if request.use_mock_llm is not None:
            settings.use_mock_llm = original_use_mock
        
        # Log exception to database
        if settings.enable_telemetry:
            try:
                persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
                await persistence_store.log_exception(
                    session_id=session_id,
                    request_id=request_uuid,
                    user_id=request.user_info.get("user_id") if request.user_info else None,
                    node_name="test_response_agent",
                    error_code="E9001",
                    category="system",
                    severity="error",
                    message=f"Test endpoint exception: {str(e)}",
                    stacktrace=tb
                )
            except Exception as log_error:
                logger.error(f"Failed to log exception: {log_error}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": tb,
                "test": "response_agent_node"
            }
        )


@router.post("/test-response-agent-final-prompt")
async def test_response_agent_final_prompt(request: ResponseAgentTestRequest):
    """
    Test Response Agent Prompt Building (No LLM Call)
    
    Tests: agents/response_agent.py - ResponseAgent prompt building methods
    
    This endpoint shows exactly what prompts are sent to the LLM without actually
    calling the LLM. Perfect for debugging and understanding what the LLM sees.
    
    Returns:
    - system_prompt: Complete system instructions (pharmacy claims assistant)
    - user_prompt: Formatted user prompt with query, intent, claim data, history
    - prompt_metadata: Statistics about prompt size and components
    - formatted_components: Individual formatted components (claim data, history)
    
    Use Cases:
    1. Verify prompt formatting is correct
    2. Debug why LLM responses might be incorrect
    3. Understand how claim data is presented to LLM
    4. Check conversation history formatting
    5. Validate ChatPromptTemplate output
    
    Example Request (with Dummy Data):
    ```json
    {
      "text": "Show me my claim details",
      "intent": "find_claim",
      "confidence": 0.95,
      "entities": {"claim_number": "CLM1234567890001"},
      "tool_results": {
        "data": {
          "claims": [{
            "claimInformation": {
              "claimNumber": "CLM1234567890001",
              "claimStatus": "P"
            },
            "drug": {"productName": "ATORVASTATIN"},
            "pricing": {"patientPay": "10.00"}
          }]
        }
      },
      "conversation_history": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"}
      ]
    }
    ```
    
    cURL Example:
    ```bash
    curl -X POST "http://localhost:8000/utils/test-response-agent-final-prompt" \
      -H "Content-Type: application/json" \
      -d '{
        "text": "Show claim CLM1234567890001",
        "intent": "find_claim",
        "tool_results": {
          "data": {
            "claims": [{
              "claimInformation": {"claimNumber": "CLM1234567890001"},
              "drug": {"productName": "ATORVASTATIN"}
            }]
          }
        }
      }'
    ```
    
    PowerShell Example:
    ```powershell
    $body = @{
        text = "Show claim CLM1234567890001"
        intent = "find_claim"
        tool_results = @{
            data = @{
                claims = @(
                    @{
                        claimInformation = @{claimNumber = "CLM1234567890001"}
                        drug = @{productName = "ATORVASTATIN"}
                    }
                )
            }
        }
    } | ConvertTo-Json -Depth 10
    
    Invoke-RestMethod -Uri "http://localhost:8000/utils/test-response-agent-final-prompt" `
      -Method POST `
      -ContentType "application/json" `
      -Body $body
    ```
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        request_uuid = request.uuid or str(uuid.uuid4())
        
        logger.info(f"🧪 TEST RESPONSE AGENT FINAL PROMPT: intent={request.intent}, has_tool_results={request.tool_results is not None}")
        
        # Create complete AgentState matching production flow
        state = AgentState(
            text=request.text,
            session_id=session_id,
            user_info=request.user_info,
            uuid=request_uuid,
            domain="claims",
            intent=request.intent,
            confidence=request.confidence,
            entities=request.entities,
            slots=None,
            required_slots=None,
            missing_slots=None,
            needs_clarification=False,
            clarifying_question=None,
            conversation_history=request.conversation_history,
            relevant_facts=[],
            extracted_slots=None,
            planner_context=None,
            tool_results=request.tool_results,
            safety_precheck_passed=True,
            safety_postcheck_passed=False,
            safety_block_reason=None,
            response="",
            metadata={
                "request_id": request_uuid,
                "user_id": request.user_info.get("user_id") if request.user_info else None
            },
            cache_hit=False,
            error=None,
            messages=[]
        )
        
        # Initialize Response Agent
        from agents.response_agent import ResponseAgent
        agent = ResponseAgent()
        
        # Build prompts (same as actual generation)
        logger.info("🔧 Building system prompt...")
        system_prompt = agent._get_system_prompt()
        
        logger.info("🔧 Building user prompt...")
        user_prompt = agent._build_user_prompt(state)
        
        # Get individual components for detailed inspection
        logger.info("🔧 Extracting formatted components...")
        claim_data_formatted = agent._format_tool_results(request.tool_results) if request.tool_results else "No claim data available"
        history_formatted = agent._format_conversation_history(request.conversation_history) if request.conversation_history else "No previous conversation"
        
        # Calculate metadata
        system_prompt_lines = system_prompt.count('\n') + 1
        user_prompt_lines = user_prompt.count('\n') + 1
        claim_data_lines = claim_data_formatted.count('\n') + 1
        history_lines = history_formatted.count('\n') + 1
        
        # Estimate token counts (rough: 4 chars ≈ 1 token for English text)
        system_prompt_tokens_est = len(system_prompt) // 4
        user_prompt_tokens_est = len(user_prompt) // 4
        total_tokens_est = system_prompt_tokens_est + user_prompt_tokens_est
        
        logger.info(f"✅ Prompts built: system={len(system_prompt)} chars, user={len(user_prompt)} chars")
        
        return {
            "success": True,
            "test": "response_agent_final_prompt",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": {
                "text": request.text,
                "intent": request.intent,
                "confidence": request.confidence,
                "entities": request.entities,
                "has_tool_results": request.tool_results is not None,
                "has_conversation_history": len(request.conversation_history) > 0 if request.conversation_history else False,
                "history_messages_count": len(request.conversation_history) if request.conversation_history else 0
            },
            "system_prompt": {
                "content": system_prompt,
                "length_chars": len(system_prompt),
                "length_lines": system_prompt_lines,
                "tokens_estimate": system_prompt_tokens_est,
                "preview_first_200": system_prompt[:200],
                "preview_last_200": system_prompt[-200:] if len(system_prompt) > 200 else system_prompt
            },
            "user_prompt": {
                "content": user_prompt,
                "length_chars": len(user_prompt),
                "length_lines": user_prompt_lines,
                "tokens_estimate": user_prompt_tokens_est,
                "preview_first_300": user_prompt[:300],
                "preview_last_300": user_prompt[-300:] if len(user_prompt) > 300 else user_prompt
            },
            "formatted_components": {
                "claim_data": {
                    "content": claim_data_formatted,
                    "length_chars": len(claim_data_formatted),
                    "length_lines": claim_data_lines,
                    "preview": claim_data_formatted[:500] + "..." if len(claim_data_formatted) > 500 else claim_data_formatted
                },
                "conversation_history": {
                    "content": history_formatted,
                    "length_chars": len(history_formatted),
                    "length_lines": history_lines,
                    "messages_included": len(request.conversation_history) if request.conversation_history else 0,
                    "note": "Last 5 messages only (as per agent logic)"
                }
            },
            "prompt_metadata": {
                "total_chars": len(system_prompt) + len(user_prompt),
                "total_lines": system_prompt_lines + user_prompt_lines,
                "total_tokens_estimate": total_tokens_est,
                "cost_estimate_usd": total_tokens_est * 0.00001,  # Rough estimate for Gemini Flash
                "breakdown": {
                    "system_prompt_percent": round((len(system_prompt) / (len(system_prompt) + len(user_prompt))) * 100, 1),
                    "user_prompt_percent": round((len(user_prompt) / (len(system_prompt) + len(user_prompt))) * 100, 1)
                }
            },
            "llm_config": {
                "model": settings.llm_model,
                "temperature": settings.llm_temperature,
                "top_p": settings.top_p,
                "max_output_tokens": settings.max_output_tokens,
                "note": "These would be used if LLM was actually called"
            },
            "validation": {
                "system_prompt_loaded": len(system_prompt) > 0,
                "system_prompt_expected_size": 5000 < len(system_prompt) < 6000,  # Should be ~5080 chars
                "user_prompt_loaded": len(user_prompt) > 0,
                "user_prompt_contains_query": request.text in user_prompt,
                "user_prompt_contains_intent": request.intent in user_prompt,
                "claim_data_formatted": "claims" in claim_data_formatted.lower() or "no claim data" in claim_data_formatted.lower(),
                "total_tokens_under_limit": total_tokens_est < 30000  # Well under typical limits
            },
            "usage_tips": [
                "💡 Copy system_prompt.content to see full system instructions",
                "💡 Copy user_prompt.content to see exactly what LLM receives",
                "💡 Use formatted_components to inspect individual pieces",
                "💡 Check validation section for any issues",
                "💡 Estimated tokens are approximate (4 chars ≈ 1 token)",
                "💡 This endpoint does NOT call the LLM - it only shows what would be sent"
            ]
        }
        
    except Exception as e:
        logger.error(f"🧪 TEST RESPONSE AGENT FINAL PROMPT ERROR: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logger.error(f"🔍 STACK TRACE:\n{tb}")
        
        # Log exception to database
        if settings.enable_telemetry:
            try:
                persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
                await persistence_store.log_exception(
                    session_id=session_id if 'session_id' in locals() else "unknown",
                    request_id=request_uuid if 'request_uuid' in locals() else "unknown",
                    user_id=request.user_info.get("user_id") if request.user_info else None,
                    node_name="test_response_agent_final_prompt",
                    error_code="E9002",
                    category="system",
                    severity="error",
                    message=f"Test endpoint exception: {str(e)}",
                    stacktrace=tb
                )
            except Exception as log_error:
                logger.error(f"Failed to log exception: {log_error}")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": tb,
                "test": "response_agent_final_prompt"
            }
        )


# ==================== Exception Handling Tests ====================

class ExceptionTestRequest(BaseModel):
    """Request for exception handling test"""
    node_name: str
    error_type: Optional[str] = "generic"  # "generic", "llm", "api"
    session_id: Optional[str] = None
    uuid: Optional[str] = None

@router.post("/test-exception-handling")
async def test_exception_handling(request: ExceptionTestRequest):
    """
    Test exception handling in specific nodes
    
    This endpoint intentionally triggers exceptions to test:
    1. Exception catching
    2. Exception logging to SQLite
    3. Graceful error responses
    
    Node names: safety_precheck, safety_postcheck, check_cache, cache_response,
                build_context, update_memory, clarification, confidence_checker,
                intent_agent, response_agent, call_claims_tool
    """
    try:
        from nodes.safety import (
            safety_precheck_node,
            response_safety_pii_precheck_node,
            response_safety_pii_postcheck_node
        )
        from nodes.cache import check_cache_node, cache_response_node
        from nodes.context import build_context_node, update_memory_node
        from nodes.clarification import clarification_node
        from nodes.confidence import confidence_checker_node
        from agents.intent_agent import intent_agent_node
        from agents.response_agent import response_agent_node
        from tools.claims_api import call_claims_tool_node
        
        session_id = request.session_id or str(uuid.uuid4())
        
        # Create a state that will trigger an exception based on node type
        # For most nodes, we'll use a state that causes errors in processing
        # We'll use a dict that's missing required fields or has invalid data
        problematic_state_dict = {
            "text": "",  # Empty text - valid but might cause issues
            "session_id": session_id,
            "user_info": {},
            "uuid": request.uuid,
            "domain": None,
            "intent": None,
            "confidence": None,
            "entities": None,
            "needs_clarification": False,
            "clarifying_question": None,
            "conversation_history": [],
            "relevant_facts": [],
            "tool_results": None,
            "safety_precheck_passed": False,
            "safety_postcheck_passed": False,
            "safety_block_reason": None,
            "response": "",
            "metadata": {},
            "cache_hit": False,
            "error": None,
            "messages": []
        }
        
        # For specific nodes, modify state to trigger errors
        if request.node_name == "check_cache":
            # Use a state that will cause error when hashing
            problematic_state_dict["text"] = object()  # Can't hash object
        elif request.node_name in ["build_context", "update_memory"]:
            # Use invalid session_id that might cause memory store errors
            problematic_state_dict["session_id"] = None  # None session_id
        elif request.node_name == "safety_precheck":
            # Use state without text field (will cause KeyError)
            problematic_state_dict.pop("text", None)
        elif request.node_name == "safety_postcheck":
            # Use state without response field
            problematic_state_dict.pop("response", None)
        elif request.node_name == "call_claims_tool":
            # Use state without intent field
            problematic_state_dict.pop("intent", None)
        
        # Convert to AgentState (this might fail for invalid states, which is fine)
        try:
            problematic_state = AgentState(**problematic_state_dict)
        except (TypeError, KeyError):
            # If we can't create valid AgentState, create a dict that will cause errors
            problematic_state = problematic_state_dict
        
        # Map node names to their functions
        node_map = {
            "safety_precheck": safety_precheck_node,
            "response_safety_pii_precheck": response_safety_pii_precheck_node,
            "response_safety_pii_postcheck": response_safety_pii_postcheck_node,
            "check_cache": check_cache_node,
            "cache_response": cache_response_node,
            "build_context": build_context_node,
            "update_memory": update_memory_node,
            "clarification": clarification_node,
            "confidence_checker": confidence_checker_node,
            "intent_agent": intent_agent_node,
            "response_agent": response_agent_node,
            "call_claims_tool": call_claims_tool_node
        }
        
        if request.node_name not in node_map:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown node name: {request.node_name}. Available: {list(node_map.keys())}"
            )
        
        node_func = node_map[request.node_name]
        
        # Call the node - this should trigger an exception
        result = await node_func(problematic_state)
        
        # Check if exception was handled
        exception_occurred = result.get("error") is not None or result.get("error_occurred", False)
        
        # Check database for logged exception
        from persistence import PersistenceStoreFactory
        from core.config import settings
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        
        # Get recent exceptions for this node
        import aiosqlite
        async with aiosqlite.connect(settings.telemetry_db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM exceptions 
                WHERE node_name = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
                """,
                (request.node_name,)
            ) as cursor:
                row = await cursor.fetchone()
                logged_exception = dict(row) if row else None
        
        return {
            "node_name": request.node_name,
            "exception_handled": exception_occurred,
            "error_in_response": result.get("error") is not None,
            "error_message": result.get("error"),
            "exception_logged": logged_exception is not None,
            "logged_exception": {
                "error_code": logged_exception.get("error_code") if logged_exception else None,
                "category": logged_exception.get("category") if logged_exception else None,
                "severity": logged_exception.get("severity") if logged_exception else None,
                "message": logged_exception.get("message")[:200] if logged_exception else None
            } if logged_exception else None,
            "result_metadata": result.get("metadata", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Exception handling test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Health Check ====================

@router.get("/health")
async def utils_health():
    """Health check for utils endpoints"""
    return {
        "status": "healthy",
        "endpoints": [
            "/utils/test-intent",
            "/utils/test-intent-agent",
            "/utils/test-cache",
            "/utils/test-persistence",
            "/utils/test-session-history",
            "/utils/test-context-building",
            "/utils/test-orchestrator",
            "/utils/test-safety-precheck",
            "/utils/test-safety-postcheck",
            "/utils/test-clarification",
            "/utils/test-confidence-checker",
            "/utils/test-claims-api",
            "/utils/test-response-agent",
            "/utils/test-exception-handling"
        ],
        "timestamp": datetime.now().isoformat()
    }

