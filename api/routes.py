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
    try:
        final_state = await run_graph(
            text=request.text,
            session_id=session_id,
            user_info=request.user_info or {}
        )
        if not isinstance(final_state, dict):
            logger.error(f"Final state not dict: {type(final_state)} -> {final_state}")
            raise HTTPException(status_code=500, detail="Invalid final state type")
        logger.debug(f"Final state keys: {list(final_state.keys())}")
        return ChatResponse(
            response=final_state.get("response", ""),
            session_id=session_id,
            intent=final_state.get("intent"),
            confidence=final_state.get("confidence"),
            needs_clarification=final_state.get("needs_clarification", False),
            clarifying_question=final_state.get("clarifying_question"),
            metadata=final_state.get("metadata", {}),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Chat endpoint error: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=str(e))
