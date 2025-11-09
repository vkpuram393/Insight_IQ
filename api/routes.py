"""
API Routes - HTTP endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import traceback

from langgraph_agent import run_graph
from core.logger import get_logger
from core.telemetry import log_event, log_request_response, RequestTimer
from persistence import EventType

router = APIRouter()
logger = get_logger(__name__)

class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint

    📍 BREAKPOINT: Set here (line 33) to debug incoming requests
    """
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_info.get("user_id") if request.user_info else None

    # Log incoming request
    await log_event(
        EventType.REQUEST_RECEIVED,
        session_id,
        {"text": request.text, "user_info": request.user_info},
        user_id
    )

    try:
        # Time the request processing
        async with RequestTimer(session_id, EventType.RESPONSE_GENERATED) as timer:
            final_state = await run_graph(
                text=request.text,
                session_id=session_id,
                user_info=request.user_info or {}
            )

        if not isinstance(final_state, dict):
            logger.error(f"Final state not dict: {type(final_state)} -> {final_state}")
            raise HTTPException(status_code=500, detail="Invalid final state type")

        logger.debug(f"Final state keys: {list(final_state.keys())}")

        # Extract response details
        response_text = final_state.get("response", "")
        intent = final_state.get("intent")
        confidence = final_state.get("confidence")
        metadata = final_state.get("metadata", {})
        metadata["duration_ms"] = timer.duration_ms
        metadata["user_id"] = user_id

        # Log complete request-response cycle
        await log_request_response(
            session_id=session_id,
            user_text=request.text,
            intent=intent,
            confidence=confidence,
            response=response_text,
            metadata=metadata
        )

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            intent=intent,
            confidence=confidence,
            needs_clarification=final_state.get("needs_clarification", False),
            clarifying_question=final_state.get("clarifying_question"),
            metadata=metadata,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Chat endpoint error: {e}\n{tb}")

        # Log error event
        await log_event(
            EventType.ERROR_OCCURRED,
            session_id,
            {"error": str(e), "traceback": tb},
            user_id
        )

        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_analytics_data():
    """
    Get analytics data

    Returns telemetry statistics including:
    - Total requests
    - Intent distribution
    - Cache hit rate
    - Error rate
    - Average response time
    """
    try:
        from core.telemetry import get_analytics
        analytics = await get_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    try:
        from memory import MemoryStoreFactory
        from core.config import settings

        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        history = await memory_store.get_session_history(session_id)
        facts = await memory_store.get_session_facts(session_id)

        return {
            "session_id": session_id,
            "history": history,
            "facts": facts
        }
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear all data for a session"""
    try:
        from memory import MemoryStoreFactory
        from core.config import settings

        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        cleared = await memory_store.clear_session(session_id)

        return {
            "session_id": session_id,
            "cleared": cleared
        }
    except Exception as e:
        logger.error(f"Failed to clear session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
