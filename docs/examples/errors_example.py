"""
Example Usage of Error Models

This file demonstrates how to integrate the AgentError models
into your conversational agent system.
"""

from typing import Optional
from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from core.error_models import (
    AgentError,
    ErrorResponse,
    ErrorCode,
    ErrorCategory,
    ErrorSeverity,
    create_validation_error,
    create_low_confidence_error,
    create_safety_error,
    create_api_error,
    create_llm_error,
    create_internal_error,
)
from core.logger import get_logger
import traceback

# Import generic serialization helpers
from utils.serialization import to_dict, to_json

logger = get_logger(__name__)


# ============================================================================
# Example 1: FastAPI Exception Handlers
# ============================================================================

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with structured error response
    
    Add this to your main.py:
        from core.errors_example import validation_exception_handler
        app.add_exception_handler(RequestValidationError, validation_exception_handler)
    """
    session_id = getattr(request.state, "session_id", None)
    
    # Extract first validation error for user message
    first_error = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(x) for x in first_error.get("loc", []))
    
    error = create_validation_error(
        message=f"Validation failed: {first_error.get('msg', 'Invalid input')}",
        field=field,
        session_id=session_id
    )
    
    # Log for internal tracking (using to_json helper)
    logger.warning(f"Validation error: {to_json(error)}")
    
    return JSONResponse(
        status_code=400,
        content=to_dict(ErrorResponse(error=error, session_id=session_id))  # Generic helper
    )


async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle all uncaught exceptions with structured error response
    
    Add this to your main.py:
        from core.errors_example import general_exception_handler
        app.add_exception_handler(Exception, general_exception_handler)
    """
    session_id = getattr(request.state, "session_id", None)
    tb = traceback.format_exc()
    
    error = create_internal_error(
        error_message=str(exc),
        stacktrace=tb,
        session_id=session_id
    )
    
    # Log for internal tracking (using to_json helper)
    logger.error(f"Unhandled exception: {to_json(error)}")
    
    return JSONResponse(
        status_code=500,
        content=to_dict(ErrorResponse(error=error, session_id=session_id))  # Generic helper
    )


# ============================================================================
# Example 2: Using in Intent Classification
# ============================================================================

async def handle_intent_classification(text: str, session_id: str):
    """Example of using errors in intent classification"""
    from agents.intent_classifier import classify_intent
    from core.config import settings
    
    try:
        result = await classify_intent(text)
        
        # Check confidence threshold
        if result["confidence"] < settings.confidence_threshold:
            error = create_low_confidence_error(
                confidence=result["confidence"],
                threshold=settings.confidence_threshold,
                intent=result.get("intent"),
                session_id=session_id
            )
            # Log and trigger clarification flow (using to_json helper)
            logger.info(f"Low confidence detected: {to_json(error)}")
            return {"error": error, "needs_clarification": True}
        
        return result
        
    except Exception as e:
        error = create_internal_error(
            error_message=f"Intent classification failed: {str(e)}",
            stacktrace=traceback.format_exc(),
            session_id=session_id,
            node_name="intent_agent"
        )
        logger.error(f"Intent classification error: {to_json(error)}")
        return {"error": error}


# ============================================================================
# Example 3: Using in Safety Checks
# ============================================================================

async def safety_precheck_example(text: str, session_id: str):
    """Example of using errors in safety checks"""
    from nodes.safety import safety_precheck
    
    result = await safety_precheck({"text": text, "session_id": session_id})
    
    if not result.get("safety_precheck_passed", False):
        reason = result.get("safety_block_reason", "Unknown safety violation")
        error = create_safety_error(
            reason=reason,
            is_precheck=True,
            session_id=session_id
        )
        logger.warning(f"Safety precheck failed: {to_json(error)}")
        return {"error": error, "blocked": True}
    
    return {"blocked": False}


# ============================================================================
# Example 4: Using in API/Tool Calls
# ============================================================================

async def call_claims_api_example(claim_id: str, session_id: str):
    """Example of using errors when calling external APIs"""
    from tools.claims_api import get_claim_status
    
    try:
        result = await get_claim_status(claim_id)
        return {"data": result}
        
    except TimeoutError as e:
        error = AgentError(
            error_code=ErrorCode.API_TIMEOUT,
            category=ErrorCategory.API_CALL,
            severity=ErrorSeverity.HIGH,
            message=f"Claims API timeout after 30s: {str(e)}",
            user_message="The request is taking longer than expected. Please try again.",
            session_id=session_id,
            node_name="claims_tool",
            metadata={"claim_id": claim_id, "timeout_seconds": 30},
            is_retryable=True,
            retry_after_seconds=30
        )
        logger.error(f"API timeout: {to_json(error)}")
        return {"error": error}
        
    except ConnectionError as e:
        error = create_api_error(
            api_name="Claims API",
            error_message=str(e),
            session_id=session_id,
            is_retryable=True
        )
        logger.error(f"API connection error: {to_json(error)}")
        return {"error": error}


# ============================================================================
# Example 5: Updated Chat Endpoint with Error Models
# ============================================================================

router = APIRouter()

@router.post("/chat_with_errors")
async def chat_with_structured_errors(request: dict):
    """
    Example chat endpoint with structured error handling
    
    This shows how to refactor your existing /chat endpoint
    to use the new error models.
    """
    from langgraph_agent import run_graph
    from core.telemetry import log_event
    from persistence import EventType
    
    session_id = request.get("session_id", "unknown")
    text = request.get("text", "")
    
    try:
        # Input validation
        if not text or len(text.strip()) == 0:
            error = create_validation_error(
                message="Text field is required and cannot be empty",
                field="text",
                value=text,
                session_id=session_id
            )
            await log_event(
                EventType.ERROR_OCCURRED,
                session_id,
                to_dict(error),  # Generic helper
                None
            )
            return to_dict(ErrorResponse(error=error, session_id=session_id))  # Generic helper
        
        if len(text) > 2000:
            error = AgentError(
                error_code=ErrorCode.INPUT_TOO_LONG,
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.LOW,
                message=f"Input text exceeds maximum length: {len(text)} > 2000",
                user_message="Your message is too long. Please keep it under 2000 characters.",
                session_id=session_id,
                is_retryable=False,
                metadata={"length": len(text), "max_length": 2000}
            )
            return to_dict(ErrorResponse(error=error, session_id=session_id))  # Generic helper
        
        # Run the graph
        final_state = await run_graph(
            text=text,
            session_id=session_id,
            user_info=request.get("user_info", {})
        )
        
        # Check if any node reported an error in state
        if final_state.get("error"):
            error_str = final_state["error"]
            error = create_internal_error(
                error_message=error_str,
                session_id=session_id,
                node_name=final_state.get("metadata", {}).get("last_node")
            )
            await log_event(
                EventType.ERROR_OCCURRED,
                session_id,
                to_dict(error),  # Generic helper
                None
            )
            return to_dict(ErrorResponse(error=error, session_id=session_id))  # Generic helper
        
        # Success response
        return {
            "success": True,
            "response": final_state.get("response", ""),
            "session_id": session_id,
            "intent": final_state.get("intent"),
            "confidence": final_state.get("confidence"),
            "metadata": final_state.get("metadata", {})
        }
        
    except Exception as e:
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=str(e),
            stacktrace=tb,
            session_id=session_id
        )
        
        # Log error for telemetry (using to_dict helper)
        await log_event(
            EventType.ERROR_OCCURRED,
            session_id,
            to_dict(error),  # Generic helper
            None
        )
        
        logger.error(f"Chat endpoint error: {to_json(error)}")
        return to_dict(ErrorResponse(error=error, session_id=session_id))  # Generic helper


# ============================================================================
# Example 6: Error Logging Helper
# ============================================================================

async def log_and_return_error(
    error: AgentError,
    user_id: Optional[str] = None
) -> ErrorResponse:
    """
    Helper function to log an error and return standardized response
    """
    from core.telemetry import log_event
    from persistence import EventType
    
    # Log to telemetry (using to_dict helper)
    await log_event(
        EventType.ERROR_OCCURRED,
        error.session_id or "unknown",
        to_dict(error),  # Generic helper
        user_id
    )
    
    # Log based on severity (using to_json helper)
    error_json = to_json(error)
    if error.severity == ErrorSeverity.CRITICAL:
        logger.critical(f"Critical error: {error_json}")
    elif error.severity == ErrorSeverity.HIGH:
        logger.error(f"Error: {error_json}")
    elif error.severity == ErrorSeverity.MEDIUM:
        logger.warning(f"Warning: {error_json}")
    else:
        logger.info(f"Info: {error_json}")
    
    return ErrorResponse(error=error, session_id=error.session_id)

