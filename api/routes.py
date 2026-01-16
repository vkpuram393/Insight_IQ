"""
API Routes - HTTP endpoints
"""

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncIterator
import uuid
from datetime import datetime, timezone
import traceback
import json

from langgraph_agent import run_graph, run_graph_stream
from core.logger import get_logger
from core.telemetry import log_event, log_request_response, RequestTimer
from persistence import EventType
from config.config import settings
from utils.jwt_utils import extract_user_info_from_jwt

router = APIRouter()
logger = get_logger(__name__)

class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    response_id: Optional[str] = None  # ✅ UUID for feedback tracking
    intent: Optional[str] = None
    confidence: Optional[float] = None
    entities: Optional[Dict[str, Any]] = None  # ✅ Extracted entities
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None  # DEPRECATED: Use 'response' field instead. Kept for backward compatibility and internal tracing only.
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """
    Main chat endpoint

    📍 BREAKPOINT: Set here (line 33) to debug incoming requests
    """
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_info.get("user_id") if request.user_info else None
    
    # Capture auth token and add to user_info for downstream use
    user_info = request.user_info.copy() if request.user_info else {}
    user_info["auth_token"] = http_request.headers.get("Authorization", "")
    
    # Extract user info from JWT for compliance audit logging (email, name, etc.)
    jwt_user_info = extract_user_info_from_jwt(user_info.get("auth_token", ""))
    user_info.update(jwt_user_info)

    # Log incoming request
    await log_event(
        EventType.REQUEST_RECEIVED,
        session_id,
        {"text": request.text, "user_info": user_info},
        user_id
    )

    try:
        # Time the request processing
        async with RequestTimer(session_id, EventType.RESPONSE_GENERATED) as timer:
            final_state = await run_graph(
                text=request.text,
                session_id=session_id,
                user_info=user_info
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
            response=response_text,  # ✅ Always contains the answer or clarification question
            session_id=session_id,
            response_id=final_state.get("response_id"),  # ✅ Include response_id for feedback
            intent=intent,
            confidence=confidence,
            entities=final_state.get("entities"),  # ✅ Include extracted entities
            needs_clarification=final_state.get("needs_clarification", False),  # ✅ If True, 'response' contains a question
            clarifying_question=None,  # DEPRECATED: Always null. Use 'response' + 'needs_clarification' instead. Kept for tracing/backward compatibility only.
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


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    """
    Streaming chat endpoint using Server-Sent Events (SSE)
    
    📍 BREAKPOINT: Set here to debug streaming requests
    
    🎯 PURPOSE:
    This endpoint provides real-time streaming of:
    1. Node status updates (for observability/debugging)
    2. Response chunks (after safety validation)
    3. Final metadata (intent, confidence, etc.)
    
    🔒 SECURITY FLOW:
    - Graph executes normally through all nodes
    - Response agent generates FULL response (with masked PII)
    - Safety postcheck validates & unmasks (COMPLETES FULLY)
    - Only then do we stream the validated response to user
    - Memory/cache updates happen asynchronously
    
    📡 SSE EVENT TYPES:
    - "node_start": Node execution began (e.g., "Checking safety...")
    - "node_complete": Node execution finished
    - "response_chunk": Piece of final response (after postcheck)
    - "complete": Full response sent, includes metadata
    - "error": Error occurred
    
    🔌 FRONTEND INTEGRATION (for Angular team):
    
    Example JavaScript/TypeScript:
    ```typescript
    const response = await fetch('/api/v1/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: 'Hello', session_id: 'xyz'})
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    // Parse SSE events and handle accordingly
    ```
    
    Returns:
        StreamingResponse with text/event-stream content type
    """
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_info.get("user_id") if request.user_info else None
    
    # Capture auth token and add to user_info for downstream use
    user_info = request.user_info.copy() if request.user_info else {}
    user_info["auth_token"] = http_request.headers.get("Authorization", "")
    
    # Extract user info from JWT for compliance audit logging (email, name, etc.)
    jwt_user_info = extract_user_info_from_jwt(user_info.get("auth_token", ""))
    user_info.update(jwt_user_info)

    # Log incoming request (follows existing pattern)
    await log_event(
        EventType.REQUEST_RECEIVED,
        session_id,
        {"text": request.text, "user_info": user_info, "streaming": True},
        user_id
    )

    async def event_generator() -> AsyncIterator[str]:
        """
        Generate SSE events from graph execution
        
        SSE Format:
            event: <event_type>\n
            data: <json_data>\n\n
        """
        try:
            start_time = datetime.now(timezone.utc)
            
            # Stream events from graph execution
            async for event in run_graph_stream(
                text=request.text,
                session_id=session_id,
                user_info=user_info
            ):
                event_type = event.get("type")
                event_data = event.get("data")
                event_metadata = event.get("metadata", {})
                
                # Format as SSE event
                # Note: SSE requires "event:" and "data:" lines, ending with double newline
                sse_message = f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
                yield sse_message
                
                # Log event for telemetry (optional, configurable)
                if settings.enable_telemetry and event_type in ["node_start", "error"]:
                    await log_event(
                        EventType.STREAM_EVENT,
                        session_id,
                        {"event_type": event_type, "node": event_metadata.get("node")},
                        user_id
                    )
                
                # If error or complete, finalize and stop
                if event_type in ["error", "complete"]:
                    end_time = datetime.now(timezone.utc)
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                    
                    if event_type == "complete":
                        # Log complete request-response cycle (follows existing pattern)
                        response_data = event_data
                        await log_request_response(
                            session_id=session_id,
                            user_text=request.text,
                            intent=response_data.get("intent"),
                            confidence=response_data.get("confidence"),
                            response=response_data.get("response"),
                            metadata={
                                **response_data.get("metadata", {}),
                                "duration_ms": duration_ms,
                                "user_id": user_id,
                                "streaming": True
                            }
                        )
                    elif event_type == "error":
                        # Log error (follows existing pattern)
                        await log_event(
                            EventType.ERROR_OCCURRED,
                            session_id,
                            {"error": event_data, "streaming": True},
                            user_id
                        )
                    
                    logger.info(f"✅ Streaming completed for session {session_id} in {duration_ms:.2f}ms")
                    break
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"🚨 Streaming error: {e}\n{tb}")
            
            # Log error event (follows existing pattern)
            await log_event(
                EventType.ERROR_OCCURRED,
                session_id,
                {"error": str(e), "traceback": tb, "streaming": True},
                user_id
            )
            
            # Send error event to client
            error_event = f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            yield error_event

    # Return SSE response
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering if behind proxy
            "Access-Control-Allow-Origin": "*",  # CORS for Angular frontend
        }
    )


class BatchTestRequest(BaseModel):
    """Request model for batch testing - same as ChatRequest"""
    text: str
    session_id: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None

class BatchTestResponse(BaseModel):
    """Response model for batch testing"""
    user_prompt: str
    session_id: str  # Return session ID so caller can reuse it for conversational testing
    agent_response: Optional[str] = None
    embedding_classifier_confidence: Optional[float] = None
    llm_judge_confidence: Optional[float] = None
    intent: Optional[str] = None
    clarification_called: bool = False
    exception: Optional[Dict[str, Any]] = None

@router.post("/test/batch", response_model=BatchTestResponse)
async def batch_test(request: BatchTestRequest):
    """
    Batch testing endpoint for processing prompts without logging.
    
    This endpoint:
    - Processes requests through the same flow as /chat
    - Returns structured response with all required fields
    - Does NOT log to telemetry (no logs, no exceptions table)
    - Returns immediately with full response
    
    Session Management:
    - If session_id is provided in request, it will be reused (conversational testing)
    - If session_id is not provided, a new session is created for each request
    - Session ID is returned in response for reuse in subsequent requests
    
    Use Cases:
    1. Independent testing: Don't provide session_id (each request gets new session)
    2. Conversational testing: Provide same session_id in all requests to maintain context
    
    Use this endpoint with Postman Collection Runner to test multiple prompts.
    """
    # Use provided session_id for conversational testing, or generate new one
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_info.get("user_id") if request.user_info else None
    
    # Store original telemetry setting
    original_telemetry_setting = settings.enable_telemetry
    
    try:
        # Disable telemetry for this request (no logging requirement)
        settings.enable_telemetry = False
        
        # Track embedding classifier confidence (before LLM judge potentially overwrites it)
        embedding_classifier_confidence = None
        llm_judge_confidence = None
        intent_reclassified = False
        
        # Run graph through same flow as chat endpoint
        try:
            final_state = await run_graph(
                text=request.text,
                session_id=session_id,
                user_info=request.user_info or {}
            )
            
            if not isinstance(final_state, dict):
                raise HTTPException(status_code=500, detail="Invalid final state type")
            
            # Extract response details
            response_text = final_state.get("response", "")
            intent = final_state.get("intent")
            confidence = final_state.get("confidence")
            needs_clarification = final_state.get("needs_clarification", False)
            metadata = final_state.get("metadata", {})
            
            # Determine which confidence scores we have
            # If intent_reclassified is True, LLM judge ran and overwrote confidence
            intent_reclassified = final_state.get("intent_reclassified", False)
            
            if intent_reclassified:
                # LLM judge ran - get both confidence scores from metadata
                embedding_classifier_confidence = metadata.get("embedding_classifier_confidence")
                llm_judge_confidence = metadata.get("llm_judge_confidence") or confidence
            else:
                # LLM judge did not run - current confidence is embedding classifier confidence
                embedding_classifier_confidence = confidence
                llm_judge_confidence = None
            
            return BatchTestResponse(
                user_prompt=request.text,
                session_id=session_id,  # Return session ID for reuse
                agent_response=response_text,
                embedding_classifier_confidence=embedding_classifier_confidence,
                llm_judge_confidence=llm_judge_confidence,
                intent=intent,
                clarification_called=needs_clarification,
                exception=None
            )
            
        except HTTPException:
            raise
        except Exception as e:
            # Exception occurred - return error in response (don't log)
            tb = traceback.format_exc()
            logger.error(f"Batch test error: {e}\n{tb}")
            
            return BatchTestResponse(
                user_prompt=request.text,
                session_id=session_id,  # Return session ID even on error
                agent_response=None,
                embedding_classifier_confidence=None,
                llm_judge_confidence=None,
                intent=None,
                clarification_called=False,
                exception={
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "stacktrace": tb
                }
            )
    
    finally:
        # Restore original telemetry setting
        settings.enable_telemetry = original_telemetry_setting
        
        # Cleanup memory after test execution to prevent accumulation
        try:
            from utils.memory_cleanup import cleanup_after_tests
            cleanup_results = await cleanup_after_tests()
            logger.debug(f"🧹 Post-test cleanup: {cleanup_results}")
        except Exception as e:
            logger.warning(f"⚠️ Post-test cleanup failed: {e}")


@router.post("/cleanup/memory")
async def cleanup_memory(aggressive: bool = Query(False, description="Perform aggressive cleanup (1 hour sessions, 1 day checkpoints)")):
    """
    Manual memory cleanup endpoint.
    
    Use this endpoint to clean up memory after test runs or when memory usage is high.
    
    Query Parameters:
    - aggressive (bool, default=False): If True, performs more aggressive cleanup
        - Sessions older than 1 hour (instead of 24 hours)
        - Checkpoints older than 1 day (instead of 7 days)
        - Multiple GC passes
    
    Returns:
        Dict with cleanup statistics
    """
    try:
        from utils.memory_cleanup import cleanup_after_tests, force_memory_cleanup
        
        if aggressive:
            results = await force_memory_cleanup()
            logger.info(f"🧹 Aggressive memory cleanup completed: {results}")
        else:
            results = await cleanup_after_tests()
            logger.info(f"🧹 Memory cleanup completed: {results}")
        
        return {
            "status": "success",
            "cleanup_type": "aggressive" if aggressive else "normal",
            "results": results
        }
    except Exception as e:
        logger.error(f"❌ Memory cleanup error: {e}")
        raise HTTPException(status_code=500, detail=f"Memory cleanup failed: {str(e)}")


@router.get("/cleanup/memory/stats")
async def get_memory_stats():
    """
    Get memory statistics for monitoring and debugging.
    
    Returns:
        Dict with memory usage statistics including:
        - Memory store stats (sessions, cache keys)
        - Checkpoint count (approximate)
        - GC statistics
    """
    try:
        import gc
        import psutil
        import os
        
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "process_memory": {},
            "memory_store": {},
            "gc_stats": {}
        }
        
        # Process memory
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            stats["process_memory"] = {
                "rss_mb": round(mem_info.rss / 1024 / 1024, 2),
                "vms_mb": round(mem_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2)
            }
        except Exception as e:
            stats["process_memory"] = {"error": str(e)}
        
        # Memory store stats
        try:
            from memory import MemoryStoreFactory
            memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
            if hasattr(memory_store, 'get_stats'):
                stats["memory_store"] = await memory_store.get_stats()
            else:
                stats["memory_store"] = {"note": "Stats not available for this store type"}
        except Exception as e:
            stats["memory_store"] = {"error": str(e)}
        
        # GC stats
        try:
            stats["gc_stats"] = {
                "enabled": gc.isenabled(),
                "thresholds": gc.get_threshold(),
                "counts": {
                    "gen0": gc.get_count()[0],
                    "gen1": gc.get_count()[1],
                    "gen2": gc.get_count()[2]
                },
                "collections": [s.get("collections", 0) for s in gc.get_stats()]
            }
        except Exception as e:
            stats["gc_stats"] = {"error": str(e)}
        
        # Embedding classifier memory (if using in-memory embeddings)
        try:
            from config.config import settings
            if settings.use_embedding_classifier and not settings.use_mongodb_for_embeddings:
                from classifiers.embedded_classifier import _embedded_classifier_instance
                if _embedded_classifier_instance is not None and hasattr(_embedded_classifier_instance, 'intent_embeddings'):
                    embeddings = _embedded_classifier_instance.intent_embeddings
                    if embeddings:
                        import sys
                        import numpy as np
                        total_size = sum(
                            arr.nbytes if isinstance(arr, np.ndarray) else sys.getsizeof(arr)
                            for arr in embeddings.values()
                        )
                        stats["embedding_classifier"] = {
                            "embeddings_loaded": len(embeddings),
                            "estimated_memory_mb": round(total_size / 1024 / 1024, 1),
                            "note": "Embeddings loaded in memory (required for classification)"
                        }
        except Exception as e:
            stats["embedding_classifier"] = {"error": str(e)}
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Memory stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Memory stats failed: {str(e)}")


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
        from config.config import settings

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
        from config.config import settings

        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        cleared = await memory_store.clear_session(session_id)

        return {
            "session_id": session_id,
            "cleared": cleared
        }
    except Exception as e:
        logger.error(f"Failed to clear session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FEEDBACK ENDPOINT (PRODUCTION)
# ============================================================================

@router.post("/feedback")
async def submit_feedback(request: dict):
    """
    Submit user feedback (thumbs up/down) for a response.
    
    This endpoint is called by the UI when a user clicks thumbs up/down
    and optionally enters a comment.
    
    Flow:
    1. User receives response from chatbot
    2. User clicks 👍 or 👎 button in UI
    3. UI prompts user to enter optional comment (max 40 chars)
    4. Frontend calls this endpoint with feedback data
    5. Backend stores in response_feedback table
    
    Request body:
    {
        "response_id": "uuid-of-response",
        "feedback_type": "THUMBSUP" or "THUMBSDOWN",
        "session_id": "session-id",
        "query_text": "original user query",
        "response_text": "agent response",
        "user_comment": "optional comment (max 40 chars)",
        "user_id": "optional user id"
    }
    """
    try:
        from core.node_models import ResponseFeedbackSchema, FeedbackType
        from persistence import PersistenceStoreFactory
        
        # Validate required fields
        response_id = request.get("response_id")
        feedback_type_str = request.get("feedback_type", "").upper()
        session_id = request.get("session_id")
        query_text = request.get("query_text", "")
        response_text = request.get("response_text", "")
        user_comment = request.get("user_comment", "")
        user_id = request.get("user_id")
        
        if not response_id:
            raise HTTPException(status_code=400, detail="response_id is required")
        
        if feedback_type_str not in ["THUMBSUP", "THUMBSDOWN"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid feedback_type. Must be THUMBSUP or THUMBSDOWN"
            )
        
        # Map string to FeedbackType enum
        feedback_type = FeedbackType[feedback_type_str]
        
        # Create feedback schema
        feedback = ResponseFeedbackSchema(
            response_id=response_id,
            response_feedback=(feedback_type == FeedbackType.THUMBSUP),
            feedback_type=feedback_type,
            session_id=session_id,
            query_text=query_text,
            response_text=response_text,
            user_comment=user_comment,  # No length limit - user can write as much as they want
            user_id=user_id,
            response_createddatetime=datetime.now(timezone.utc)
        )
        
        # Store feedback
        store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        result = await store.save_response_feedback(feedback)
        
        logger.info(f"✅ Feedback submitted: {feedback_type_str} for response {response_id} | comment: '{user_comment[:20] if user_comment else 'none'}...'")
        
        return {
            "status": "success",
            "message": "Feedback stored successfully",
            "response_id": response_id,
            "feedback_type": feedback_type_str,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error storing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {str(e)}")


@router.get("/feedback")
async def get_feedback(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 100,
    skip: int = 0
):
    """
    Retrieve feedback records from Response_Feedback collection.
    
    Query Parameters:
    - user_id (optional): Filter by user ID
    - session_id (optional): Filter by session ID
    - limit (optional): Maximum number of records to return (default: 100)
    - skip (optional): Number of records to skip for pagination (default: 0)
    
    Returns:
    - List of feedback records matching the filter criteria
    
    Examples:
    - GET /api/v1/feedback?user_id=test_user_123
    - GET /api/v1/feedback?session_id=Test-1session-098
    - GET /api/v1/feedback?user_id=test_user_123&session_id=Test-1session-098
    - GET /api/v1/feedback?limit=10&skip=0
    """
    try:
        from persistence import PersistenceStoreFactory
        
        # Validate at least one filter is provided
        if not user_id and not session_id:
            raise HTTPException(
                status_code=400,
                detail="At least one filter parameter is required (user_id or session_id)"
            )
        
        # Build query filter
        query_filter = {}
        if user_id:
            query_filter["user_id"] = user_id
        if session_id:
            query_filter["session_id"] = session_id
        
        # Validate pagination parameters
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 1000"
            )
        if skip < 0:
            raise HTTPException(
                status_code=400,
                detail="Skip must be non-negative"
            )
        
        # Get feedback from persistence store
        store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        feedbacks = await store.get_all_feedback_with_filter(
            query_filter=query_filter,
            limit=limit,
            skip=skip
        )
        
        logger.info(f"📊 Retrieved {len(feedbacks)} feedback records | filters: {query_filter}")
        
        return {
            "status": "success",
            "count": len(feedbacks),
            "limit": limit,
            "skip": skip,
            "filters": query_filter,
            "data": feedbacks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feedback: {str(e)}")
