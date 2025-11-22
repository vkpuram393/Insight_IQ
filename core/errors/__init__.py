"""
Error Handling Module

This module consolidates all error-related functionality:
- Error models and types
- Custom exceptions
- Error handlers
"""

from core.errors.models import (
    AgentError,
    ErrorResponse,
    ErrorCode,
    ErrorCategory,
    ErrorSeverity,
    ErrorDetail,
    create_validation_error,
    create_low_confidence_error,
    create_safety_error,
    create_api_error,
    create_llm_error,
    create_session_error,
    create_internal_error,
    create_orchestrator_empty_input_error,
    create_orchestrator_invalid_type_error,
    create_orchestrator_normalization_error,
)

from core.errors.exceptions import (
    APIBaseError,
    ExternalAPIError,
    ToolTimeoutError,
)

from core.errors.error_handler import to_agent_error

__all__ = [
    # Models
    "AgentError",
    "ErrorResponse",
    "ErrorCode",
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorDetail",
    # Factory functions
    "create_validation_error",
    "create_low_confidence_error",
    "create_safety_error",
    "create_api_error",
    "create_llm_error",
    "create_session_error",
    "create_internal_error",
    "create_orchestrator_empty_input_error",
    "create_orchestrator_invalid_type_error",
    "create_orchestrator_normalization_error",
    # Exceptions
    "APIBaseError",
    "ExternalAPIError",
    "ToolTimeoutError",
    # Handlers
    "to_agent_error",
]

