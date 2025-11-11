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

from agents.intent_classifier import classify_intent
from agents import intent_agent_node, response_agent_node  # Use configured classifier based on settings
from nodes.safety import safety_precheck_node, safety_postcheck_node
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

@router.post("/test-context-building")
async def test_context_building(request: ContextTestRequest):
    """
    Test context building node

    Tests: nodes/context.py
    """
    try:
        state = AgentState(
            text=request.text,
            session_id=request.session_id,
            user_info={},
            conversation_history=[],
            relevant_facts=[]
        )

        result = await build_context_node(state)

        return {
            "session_id": request.session_id,
            "conversation_history": result.get("conversation_history", []),
            "relevant_facts": result.get("relevant_facts", []),
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
            conversation_history=[],
            relevant_facts=[],
            intent="claim_status",
            confidence=0.9,
            entities={"claim_number": "12345"},
            tool_result={"status": "approved", "amount": "$500"}
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
            "/utils/test-claims-api",
            "/utils/test-response-agent"
        ],
        "timestamp": datetime.now().isoformat()
    }

