"""
Error Models for Conversational Agent System

Comprehensive Pydantic models for structured error handling across
the agent system, including intent classification, API calls, safety checks,
and general processing errors.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"           # Informational, minor issues
    MEDIUM = "medium"     # Handled errors, degraded functionality
    HIGH = "high"         # Significant errors, action required
    CRITICAL = "critical" # System failure, immediate attention needed


class ErrorCategory(str, Enum):
    """Error categories specific to conversational agent system"""
    VALIDATION = "validation"           # Input validation errors
    AUTHENTICATION = "authentication"   # Auth/authorization errors
    INTENT_CLASSIFICATION = "intent_classification"  # Intent detection failures
    CONFIDENCE = "confidence"           # Low confidence issues
    CLARIFICATION = "clarification"     # Clarification needed
    SAFETY = "safety"                   # Safety check failures
    API_CALL = "api_call"              # External API errors
    TOOL_EXECUTION = "tool_execution"  # Tool/function call errors
    MEMORY = "memory"                  # Memory store errors
    CACHE = "cache"                    # Cache operation errors
    PERSISTENCE = "persistence"        # Database/storage errors
    LLM = "llm"                        # LLM connection/response errors
    CONTEXT = "context"                # Context retrieval errors
    SESSION = "session"                # Session management errors
    TIMEOUT = "timeout"                # Request timeout errors
    RATE_LIMIT = "rate_limit"          # Rate limiting errors
    SYSTEM = "system"                  # General system errors
    UNKNOWN = "unknown"                # Unclassified errors


class ErrorCode(str, Enum):
    """Specific error codes for fine-grained error handling"""
    # Validation (1xxx)
    INVALID_INPUT = "E1001"
    MISSING_REQUIRED_FIELD = "E1002"
    INVALID_FORMAT = "E1003"
    INPUT_TOO_LONG = "E1004"
    
    # Intent & Confidence (2xxx)
    INTENT_DETECTION_FAILED = "E2001"
    LOW_CONFIDENCE = "E2002"
    AMBIGUOUS_INTENT = "E2003"
    UNSUPPORTED_INTENT = "E2004"
    ENTITY_EXTRACTION_FAILED = "E2005"
    
    # Safety (3xxx)
    SAFETY_PRECHECK_FAILED = "E3001"
    SAFETY_POSTCHECK_FAILED = "E3002"
    HARMFUL_CONTENT_DETECTED = "E3003"
    PII_DETECTED = "E3004"
    
    # API & Tools (4xxx)
    API_CONNECTION_ERROR = "E4001"
    API_TIMEOUT = "E4002"
    API_AUTHENTICATION_ERROR = "E4003"
    API_RATE_LIMIT = "E4004"
    API_INVALID_RESPONSE = "E4005"
    TOOL_EXECUTION_ERROR = "E4006"
    
    # LLM (5xxx)
    LLM_CONNECTION_ERROR = "E5001"
    LLM_TIMEOUT = "E5002"
    LLM_RATE_LIMIT = "E5003"
    LLM_INVALID_RESPONSE = "E5004"
    LLM_QUOTA_EXCEEDED = "E5005"
    
    # Memory & State (6xxx)
    MEMORY_STORE_ERROR = "E6001"
    SESSION_NOT_FOUND = "E6002"
    CACHE_ERROR = "E6003"
    PERSISTENCE_ERROR = "E6004"
    CONTEXT_RETRIEVAL_ERROR = "E6005"
    
    # System (9xxx)
    INTERNAL_ERROR = "E9001"
    CONFIGURATION_ERROR = "E9002"
    TIMEOUT = "E9003"
    UNKNOWN_ERROR = "E9999"


class ErrorDetail(BaseModel):
    """Detailed error information"""
    field: Optional[str] = Field(None, description="Field name that caused the error")
    value: Optional[Any] = Field(None, description="Value that caused the error")
    constraint: Optional[str] = Field(None, description="Constraint that was violated")
    suggestion: Optional[str] = Field(None, description="Suggestion to fix the error")


class AgentError(BaseModel):
    """
    Comprehensive error model for conversational agent system
    
    This model provides structured error information suitable for:
    - Logging and debugging
    - User-facing error messages
    - Telemetry and analytics
    - Error recovery and retry logic
    """
    
    # Core error information
    error_code: ErrorCode = Field(
        ..., 
        description="Specific error code for programmatic handling"
    )
    category: ErrorCategory = Field(
        ..., 
        description="High-level error category"
    )
    severity: ErrorSeverity = Field(
        ..., 
        description="Error severity level"
    )
    
    # Messages
    message: str = Field(
        ..., 
        description="Technical error message for debugging"
    )
    user_message: str = Field(
        ..., 
        description="User-friendly error message safe to display"
    )
    
    # Context
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the error occurred (ISO 8601 format)"
    )
    session_id: Optional[str] = Field(
        None, 
        description="Session ID where error occurred"
    )
    request_id: Optional[str] = Field(
        None, 
        description="Unique request/transaction ID"
    )
    user_id: Optional[str] = Field(
        None, 
        description="User ID if available"
    )
    
    # Technical details
    details: Optional[List[ErrorDetail]] = Field(
        None, 
        description="Detailed error information"
    )
    stacktrace: Optional[str] = Field(
        None, 
        description="Stack trace for debugging (not shown to users)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context-specific metadata"
    )
    
    # Recovery information
    is_retryable: bool = Field(
        False, 
        description="Whether the operation can be retried"
    )
    retry_after_seconds: Optional[int] = Field(
        None, 
        description="Suggested retry delay in seconds"
    )
    
    # Related information
    related_intent: Optional[str] = Field(
        None, 
        description="Intent being processed when error occurred"
    )
    node_name: Optional[str] = Field(
        None, 
        description="Agent node where error occurred"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "E2002",
                "category": "confidence",
                "severity": "medium",
                "message": "Intent confidence 0.45 below threshold 0.70",
                "user_message": "I'm not quite sure what you're asking. Could you rephrase that?",
                "timestamp": "2025-11-10T10:30:45.123456Z",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "request_id": "req_abc123",
                "is_retryable": False,
                "related_intent": "unknown",
                "node_name": "confidence_check",
                "metadata": {
                    "confidence_score": 0.45,
                    "threshold": 0.70
                }
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response format for API endpoints
    """
    success: bool = Field(False, description="Always False for error responses")
    error: AgentError = Field(..., description="Error details")
    session_id: Optional[str] = Field(None, description="Session ID if available")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Response timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": {
                    "error_code": "E4001",
                    "category": "api_call",
                    "severity": "high",
                    "message": "Failed to connect to claims API: Connection timeout",
                    "user_message": "I'm having trouble accessing your claims information right now. Please try again in a moment.",
                    "timestamp": "2025-11-10T10:30:45.123456Z",
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "is_retryable": True,
                    "retry_after_seconds": 30,
                    "node_name": "claims_tool"
                },
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2025-11-10T10:30:45.123456Z"
            }
        }


# ============================================================================
# Helper Functions for Creating Common Errors
# ============================================================================

def create_validation_error(
    message: str,
    field: Optional[str] = None,
    value: Optional[Any] = None,
    session_id: Optional[str] = None
) -> AgentError:
    """Create a validation error"""
    return AgentError(
        error_code=ErrorCode.INVALID_INPUT,
        category=ErrorCategory.VALIDATION,
        severity=ErrorSeverity.LOW,
        message=message,
        user_message="The information provided is not valid. Please check your input and try again.",
        session_id=session_id,
        details=[ErrorDetail(field=field, value=value)] if field else None,
        is_retryable=False
    )


def create_low_confidence_error(
    confidence: float,
    threshold: float,
    intent: Optional[str] = None,
    session_id: Optional[str] = None
) -> AgentError:
    """Create a low confidence error"""
    return AgentError(
        error_code=ErrorCode.LOW_CONFIDENCE,
        category=ErrorCategory.CONFIDENCE,
        severity=ErrorSeverity.MEDIUM,
        message=f"Intent confidence {confidence:.2f} below threshold {threshold:.2f}",
        user_message="I'm not quite sure what you're asking. Could you rephrase that or provide more details?",
        session_id=session_id,
        related_intent=intent,
        metadata={"confidence": confidence, "threshold": threshold},
        is_retryable=False
    )


def create_safety_error(
    reason: str,
    is_precheck: bool = True,
    session_id: Optional[str] = None
) -> AgentError:
    """Create a safety check error"""
    return AgentError(
        error_code=ErrorCode.SAFETY_PRECHECK_FAILED if is_precheck else ErrorCode.SAFETY_POSTCHECK_FAILED,
        category=ErrorCategory.SAFETY,
        severity=ErrorSeverity.HIGH,
        message=f"Safety check failed: {reason}",
        user_message="I cannot process this request as it may violate our safety guidelines.",
        session_id=session_id,
        node_name="safety_precheck" if is_precheck else "safety_postcheck",
        metadata={"reason": reason},
        is_retryable=False
    )


def create_api_error(
    api_name: str,
    error_message: str,
    status_code: Optional[int] = None,
    session_id: Optional[str] = None,
    is_retryable: bool = True
) -> AgentError:
    """Create an API call error"""
    return AgentError(
        error_code=ErrorCode.API_CONNECTION_ERROR,
        category=ErrorCategory.API_CALL,
        severity=ErrorSeverity.HIGH,
        message=f"{api_name} API error: {error_message}",
        user_message="I'm having trouble accessing the information you requested. Please try again in a moment.",
        session_id=session_id,
        metadata={"api": api_name, "status_code": status_code},
        is_retryable=is_retryable,
        retry_after_seconds=30 if is_retryable else None
    )


def create_llm_error(
    error_message: str,
    session_id: Optional[str] = None,
    is_retryable: bool = True
) -> AgentError:
    """Create an LLM connection/processing error"""
    return AgentError(
        error_code=ErrorCode.LLM_CONNECTION_ERROR,
        category=ErrorCategory.LLM,
        severity=ErrorSeverity.CRITICAL,
        message=f"LLM error: {error_message}",
        user_message="I'm experiencing technical difficulties. Please try again shortly.",
        session_id=session_id,
        is_retryable=is_retryable,
        retry_after_seconds=60 if is_retryable else None
    )


def create_session_error(
    session_id: str,
    error_message: str
) -> AgentError:
    """Create a session management error"""
    return AgentError(
        error_code=ErrorCode.SESSION_NOT_FOUND,
        category=ErrorCategory.SESSION,
        severity=ErrorSeverity.MEDIUM,
        message=f"Session error: {error_message}",
        user_message="I couldn't find your conversation session. Let's start fresh.",
        session_id=session_id,
        is_retryable=False
    )


def create_internal_error(
    error_message: str,
    stacktrace: Optional[str] = None,
    session_id: Optional[str] = None,
    node_name: Optional[str] = None
) -> AgentError:
    """Create a generic internal error"""
    return AgentError(
        error_code=ErrorCode.INTERNAL_ERROR,
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.CRITICAL,
        message=error_message,
        user_message="An unexpected error occurred. Our team has been notified.",
        session_id=session_id,
        node_name=node_name,
        stacktrace=stacktrace,
        is_retryable=False
    )

