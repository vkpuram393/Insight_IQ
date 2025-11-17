"""
Logging Context Helper

Extracts standardized logging context from AgentState to ensure consistency
across all nodes when logging to SQLite.
"""

from typing import Dict, Optional
from state.schema import AgentState


def extract_logging_context(state: AgentState) -> Dict[str, Optional[str]]:
    """
    Extract standardized logging context from AgentState.
    
    This ensures all nodes use the same logic to extract:
    - session_id: Conversation session identifier
    - request_id: Request UUID from orchestrator (stored in state.uuid)
    - user_id: User identifier from user_info
    
    Args:
        state: The AgentState from LangGraph
        
    Returns:
        Dictionary with session_id, request_id, and user_id
        
    Example:
        >>> log_ctx = extract_logging_context(state)
        >>> await persistence_store.log_audit(
        ...     session_id=log_ctx["session_id"],
        ...     request_id=log_ctx["request_id"],
        ...     user_id=log_ctx["user_id"],
        ...     node_name="my_node",
        ...     event_type="my_event",
        ...     data={...}
        ... )
    """
    return {
        "session_id": state.get("session_id", "unknown"),
        "request_id": state.get("uuid"),  # UUID from orchestrator
        "user_id": state.get("user_info", {}).get("user_id")
    }

