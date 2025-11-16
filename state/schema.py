"""
LangGraph State Schema

🎓 CONCEPT: State is like a form that travels through an assembly line.
Each worker (node/agent) fills in their section and passes it along.

In LangGraph, state is THE CORE CONCEPT. Everything reads from and writes to state.
"""

from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Complete state that flows through the graph

    🎯 UNDERSTAND THIS:
    - When a request arrives, we create initial state
    - State flows: Node1 → Node2 → Node3 → ... → Response
    - Each node can READ all fields and WRITE to fields
    - At the end, state contains everything that happened

    📊 STATE FLOW EXAMPLE:

    Initial:
        text="What's my claim status?"
        intent=None
        response=""

    After intent_agent:
        text="What's my claim status?"
        intent="claim_status"
        confidence=0.95
        response=""

    After response_agent:
        text="What's my claim status?"
        intent="claim_status"
        confidence=0.95
        response="Your claim #12345 is approved!"
    """

    # === INPUT (from user) ===
    text: str                                    # User's message
    session_id: str                              # Conversation ID
    user_info: Dict[str, Any]                    # User metadata
    uuid: Optional[str]                          # Request UUID from orchestrator
    domain: Optional[str]                        # Domain (e.g., "claims", "prescriptions")

    # === INTENT & ENTITIES (from intent_agent) ===
    intent: Optional[str]                        # What user wants
    confidence: Optional[float]                  # How sure we are (0-1)
    entities: Optional[Dict[str, Any]]           # Extracted info

    # === CLARIFICATION ===
    needs_clarification: bool                    # Ask user question?
    clarifying_question: Optional[str]           # The question

    # === CONTEXT ===
    conversation_history: List[Dict[str, str]]   # Recent messages
    relevant_facts: List[Dict[str, Any]]         # Important facts

    # === TOOL RESULTS ===
    tool_results: Optional[Dict[str, Any]]       # API call results

    # === SAFETY ===
    safety_precheck_passed: bool                 # Input safe?
    safety_postcheck_passed: bool                # Output safe?
    safety_block_reason: Optional[str]           # Why blocked

    # === OUTPUT (to user) ===
    response: str                                # Final answer

    # === METADATA ===
    metadata: Dict[str, Any]                     # Tracking info
    cache_hit: bool                              # From cache?
    error: Optional[str]                         # Any error

    # === LANGGRAPH MESSAGES (for checkpointing) ===
    messages: Annotated[List[BaseMessage], add_messages]  # Message history

def create_initial_state(
    text: str,
    session_id: str,
    user_info: Dict[str, Any] = None
) -> AgentState:
    """
    Create starting state for new request

    Think of this as filling out the top of a form before
    sending it down the assembly line.
    """
    return AgentState(
        text=text,
        session_id=session_id,
        user_info=user_info or {},
        uuid=None,
        domain=None,
        intent=None,
        confidence=None,
        entities=None,
        needs_clarification=False,
        clarifying_question=None,
        conversation_history=[],
        relevant_facts=[],
        tool_results=None,
        safety_precheck_passed=False,
        safety_postcheck_passed=False,
        safety_block_reason=None,
        response="",
        metadata={},
        cache_hit=False,
        error=None,
        messages=[]
    )
