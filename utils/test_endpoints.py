"""
Utility Test Endpoints

Individual endpoints for testing each component independently.
Each developer can test their functionality without running the full application.

Endpoints:
- /utils/test-intent - Test intent classifier
- /utils/test-cache - Test in-memory cache
- /utils/test-persistence - Test SQLite persistence
- /utils/test-memory-store - Test memory store operations
- /utils/test-session-history - Test session management
- /utils/test-context-building - Test context retrieval
- /utils/test-safety - Test safety checks
- /utils/test-confidence - Test confidence scoring
- /utils/test-clarification - Test clarification logic
- /utils/test-claims-api - Test claims API mock
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime

from classifiers.intent_classifier import classify_intent
from agents import intent_agent_node, response_agent_node  # Use configured classifier based on settings
from nodes.safety import safety_precheck_node, response_safety_pii_postcheck_node as safety_postcheck_node
from nodes.confidence import confidence_check_router
from nodes.clarification import clarification_node
from nodes.context import build_context_node, update_memory_node
from nodes.cache import check_cache_node, cache_response_node
from tools.claims_api import call_claims_tool_node
from memory import MemoryStoreFactory
from persistence import PersistenceStoreFactory, EventType
from config.config import settings
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

# ==================== Intent Classifier Tests ====================

@router.post("/test-intent", response_model=IntentTestResponse)
async def test_intent_classifier(request: IntentTestRequest):
    """
    Test the intent classifier independently

    Tests: classifiers/intent_classifier.py
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
            # NEW: Show API routing from config
            "api_endpoint": result.get("api_endpoint"),
            "required_entities_list": result.get("required_entities_list"),
            "requires_llm": result.get("requires_llm"),
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

        result = await safety_postcheck_node(state)

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
        
        # Check for low confidence (updated logic - confidence_checker no longer sets needs_clarification)
        metadata = result.get("metadata", {})
        confidence_low_detected = metadata.get("confidence_low_detected", False)
        
        if confidence_low_detected:
            # Low confidence detected - would route to llm_judge or clarification
            return {
                "decision": "low_confidence",
                "confidence_low_detected": True,
                "confidence": metadata.get("confidence"),
                "threshold": metadata.get("threshold"),
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
        
        # Check if clarification explicitly set (for backward compatibility)
        if result.get("needs_clarification"):
            return {
                "decision": "clarification",
                "needs_clarification": True,
                "clarifying_question": result.get("clarifying_question"),
                "response": result.get("response"),
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
        
        # High confidence - call context builder
        if metadata.get("confidence_check_passed"):
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


# ==================== Claims API Tests ====================

@router.post("/test-claims-api")
async def test_claims_api(request: Dict[str, Any]):
    """
    Test claims API mock

    Tests: tools/claims_api.py
    """
    try:
        state = AgentState(
            text=request.get("text", ""),
            session_id=str(uuid.uuid4()),
            user_info={},
            uuid=None,
            domain=None,
            conversation_history=[],
            relevant_facts=[],
            intent=request.get("intent", "claim_status"),
            entities=request.get("entities", {})
        )

        result = await call_claims_tool_node(state)

        return {
            "tool_called": True,
            "tool_result": result.get("tool_result"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Claims API test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Response Agent Tests ====================

@router.post("/test-response-agent")
async def test_response_agent(request: IntentTestRequest):
    """
    Test response generation agent

    Tests: agents/response_agent.py
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
            confidence=0.9,
            entities={"claim_number": "12345"},
            tool_results={"status": "approved", "amount": "$500"}
        )

        result = await response_agent_node(state)

        return {
            "input": request.text,
            "response": result.get("response"),
            "intent": state.get("intent", "claim_status"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Response agent test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        from nodes.safety import safety_precheck_node, response_safety_pii_postcheck_node as safety_postcheck_node
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
            "safety_postcheck": safety_postcheck_node,
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
        from config.config import settings
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

