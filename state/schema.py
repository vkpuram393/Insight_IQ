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

    # === INTENT & ENTITIES (from intent_agent) ===
    intent: Optional[str]                        # What user wants
    confidence: Optional[float]                  # How sure we are (0-1)
    entities: Optional[Dict[str, Any]]           # Extracted info
    
    # === API ROUTING (from config via intent_agent) ===
    api_endpoint: Optional[str]                  # Which CVS API to call (from config)
    required_entities_list: Optional[List[str]]  # Required entities for this intent
    requires_llm: bool                           # Intent needs LLM (no API)

    # === MASTER LLM AGENT (Stage 2 Routing) ===
    llm_action: Optional[str]                    # LLM decision: call_api, search_faq, ask_clarification, general_response
    llm_confidence: Optional[float]              # LLM confidence (0-1)
    llm_reasoning: Optional[str]                 # Why LLM made this decision
    llm_rerouted: bool                           # Did LLM reroute from Stage 1 decision?
    needs_api_reroute: bool                      # Signal to router: reroute to API
    needs_faq: bool                              # Signal to router: search FAQ

    # === CLARIFICATION ===
    needs_clarification: bool                    # Ask user question?
    clarifying_question: Optional[str]           # The question

    # === CONTEXT ===
    conversation_history: List[Dict[str, str]]   # Recent messages
    relevant_facts: List[Dict[str, Any]]         # Important facts

    # === TOOL RESULTS ===
    tool_results: Optional[Dict[str, Any]]       # API call results
    api_error: Optional[str]                     # API error message if call failed
    api_retry_count: int                         # Number of API retry attempts

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
        intent=None,
        confidence=None,
        entities=None,
        # API Routing (from config)
        api_endpoint=None,
        required_entities_list=None,
        requires_llm=False,
        # Master LLM Agent (Stage 2) fields
        llm_action=None,
        llm_confidence=None,
        llm_reasoning=None,
        llm_rerouted=False,
        needs_api_reroute=False,
        needs_faq=False,
        # Other fields
        needs_clarification=False,
        clarifying_question=None,
        conversation_history=[],
        relevant_facts=[],
        tool_results=None,
        api_error=None,
        api_retry_count=0,
        safety_precheck_passed=False,
        safety_postcheck_passed=False,
        safety_block_reason=None,
        response="",
        metadata={},
        cache_hit=False,
        error=None,
        messages=[]
    )
