"""
Custom exceptions for Claims API Orchestrator.
Lightweight metadata-bearing exceptions that will be converted to AgentError.
"""

from typing import Optional, Dict, Any

# ============================================================================
# BASE API ERROR
# ============================================================================
class APIBaseError(Exception):
    """Base class for API/tool related exceptions carrying metadata."""
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None, retriable: bool = False):
        super().__init__(message)
        self.details = details or {}
        self.retriable = retriable

# ============================================================================
# EXTERNAL API ERROR
# ============================================================================
class ExternalAPIError(APIBaseError):
    """Raised when external API returns HTTP error or invalid response."""

# ============================================================================
# TOOL TIMEOUT ERROR
# ============================================================================
class ToolTimeoutError(APIBaseError):
    """Raised when an external API request times out."""
