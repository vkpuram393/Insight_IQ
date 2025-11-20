"""
Map exceptions -> AgentError using your helper functions.
This file ensures every exception in the orchestrator is normalized to AgentError.
"""

import traceback
from typing import Optional
from core.error_models import (
    AgentError,
    create_api_error,
    create_internal_error,
    create_validation_error
)
from tools.exceptions import APIBaseError, ExternalAPIError, ToolTimeoutError

# ============================================================================
# TO AGENT ERROR 
# ============================================================================
def to_agent_error(exc: Exception, *, node: Optional[str] = None) -> AgentError:
    """
    Convert an exception to an AgentError using helper factories.
    """
    # Known timeout
    if isinstance(exc, ToolTimeoutError):
        # use create_api_error but mark as timeout-like
        ae = create_api_error(
            api_name=node or "unknown_api",
            error_message=str(exc),
            status_code=exc.details.get("status") if isinstance(exc, APIBaseError) else None,
            is_retryable=getattr(exc, "retriable", True)
        )
        # tweak message/category to reflect timeout if desired
        ae.message = f"Timeout: {ae.message}"
        return ae

    # Known external API error
    if isinstance(exc, ExternalAPIError):
        ae = create_api_error(
            api_name=node or "unknown_api",
            error_message=str(exc),
            status_code=exc.details.get("status") if isinstance(exc, APIBaseError) else None,
            is_retryable=getattr(exc, "retriable", False)
        )
        return ae

    # Generic tool/API base error
    if isinstance(exc, APIBaseError):
        return create_api_error(
            api_name=node or "unknown_api",
            error_message=str(exc),
            status_code=exc.details.get("status"),
            is_retryable=getattr(exc, "retriable", False)
        )

    # Fallback: unexpected/internal exception -> internal error
    trace = traceback.format_exc()
    return create_internal_error(
        error_message=str(exc),
        stacktrace=trace,
        node_name=node
    )
