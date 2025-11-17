"""
Logging Context Helper

Extracts standardized logging context from AgentState to ensure consistency
across all nodes when logging to SQLite.
"""

from typing import Dict, Optional, Any
from state.schema import AgentState
from persistence import PersistenceStoreFactory
from core.config import settings


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


async def log_state_snapshot(
    state: AgentState,
    node_name: str,
    node_result: Dict[str, Any]
) -> str:
    """
    Log full AgentState snapshot after a node completes.
    
    This creates a single log entry with the complete AgentState after
    the node's updates are merged. This provides a complete picture of
    the state at each step in the graph execution.
    
    Args:
        state: The AgentState before node execution
        node_name: Name of the node that just executed
        node_result: The partial update returned by the node
        
    Returns:
        Log ID of the created log entry
        
    Example:
        >>> result = await my_node(state)
        >>> log_id = await log_state_snapshot(state, "my_node", result)
    """
    log_ctx = extract_logging_context(state)
    
    # Simulate LangGraph merge: combine state with node result
    updated_state = {**state, **node_result}
    
    # Get persistence store
    persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
    
    # Log full AgentState snapshot
    log_id = await persistence_store.log_audit(
        session_id=log_ctx["session_id"],
        request_id=log_ctx["request_id"],
        user_id=log_ctx["user_id"],
        node_name=node_name,
        event_type="state_snapshot",
        data=updated_state  # Full AgentState after node execution
    )
    
    return log_id

