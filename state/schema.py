"""
LangGraph State Schema

🎓 CONCEPT: State is like a form that travels through an assembly line.
Each worker (node/agent) fills in their section and passes it along.

In LangGraph, state is THE CORE CONCEPT. Everything reads from and writes to state.
"""

from typing import TypedDict, Optional, List, Dict, Any, Callable
from typing_extensions import Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

def merge_metadata(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reducer function to merge metadata dictionaries.
    
    Merges right into left, with right taking precedence for overlapping keys.
    For nested dictionaries, performs deep merge.
    """
    result = left.copy()
    for key, value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Deep merge nested dictionaries
            result[key] = merge_metadata(result[key], value)
        else:
            # Overwrite or add new key
            result[key] = value
    return result

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
    session_id: str                              # Conversation ID (per chatbot open — ephemeral)
    user_session: Optional[str]                  # UI-provided stable session ID for persistent history (per MyClaims login)
    user_info: Dict[str, Any]                    # User metadata
    uuid: Optional[str]                          # Request UUID from orchestrator
    domain: Optional[str]                        # Domain (e.g., "claims", "prescriptions")

    # === INTENT & ENTITIES (from intent_agent) ===
    intent: Optional[str]                        # What user wants
    confidence: Optional[float]                  # How sure we are (0-1); always the ensemble post-calibration score, never overwritten by llm_fallback_confidence
    llm_fallback_confidence: Optional[float]     # LLM fallback confidence (0-1); None when ensemble was used directly
    ensemble_intent: Optional[str]               # Ensemble classifier's pick (pre-LLM-fallback)
    ensemble_confidence: Optional[float]         # Ensemble post-calibration confidence; always set when multidomain runs, regardless of whether LLM fallback ran
    llm_fallback_intent: Optional[str]           # LLM fallback's pick; None when ensemble was used directly
    entities: Optional[Dict[str, Any]]           # Extracted info
    slots: Optional[Dict[str, Any]]              # API parameters (from intent classifier)
    required_slots: Optional[List[str]]          # Required slots for this intent (from intent classifier)
    missing_slots: Optional[List[str]]           # Required slots that are missing (from intent classifier)
    intent_reclassified: bool                    # Has LLM judge re-classified the intent? (prevents infinite loops)
    
    # === API ROUTING (from config via intent_agent) ===
    api_endpoint: Optional[str]                  # Which CVS API to call (from config)
    required_entities_list: Optional[List[str]]  # Required entities for this intent
    requires_llm: bool                           # Intent needs LLM (no API)
    is_complex: bool                             # Query needs complex analysis (aggregations, comparisons)
    embedding_failed: bool                       # Embedding classifier failed - route to LLM

    # === CLARIFICATION ===
    needs_clarification: bool                    # Ask user question? (Also used by response_agent to determine mode)
    clarifying_question: Optional[str]           # The question (after generation)
    clarification_context: Optional[Dict[str, Any]]  # Context for clarification generation
    # clarification_context structure:
    # {
    #     "reason": "low_confidence" | "missing_entity" | "ambiguous_intent",
    #     "confidence": float,
    #     "intent": str,
    #     "user_query": str,
    #     "missing_entities": List[str],
    #     "intent_candidates": List[Tuple[str, float]]
    # }

    # === CONTEXT ===
    conversation_history: List[Dict[str, str]]   # Recent messages
    relevant_facts: List[Dict[str, Any]]         # Important facts
    extracted_slots: Optional[Dict[str, Any]]     # Slots extracted from conversation history (from context builder)
    planner_context: Optional[Dict[str, Any]]    # Complete context object for planner/executor (from context builder)

    # === TOOL RESULTS ===
    tool_results: Optional[Dict[str, Any]]       # API call results
    api_error: Optional[str]                     # API error message if call failed
    api_retry_count: int                         # Number of API retry attempts

    # === SAFETY ===
    safety_precheck_passed: bool                 # Input safe?
    safety_postcheck_passed: bool                # Output safe?
    safety_block_reason: Optional[str]           # Why blocked
    
    # === PII/PHI TOKEN MAPPINGS (source-aware) ===
    # NEW ARCHITECTURE: Track token mappings by source for accurate unmasking
    # Priority for unmasking: tool_tokens > text_tokens > context_tokens
    tool_tokens: Optional[Dict[str, Any]]        # Tokens from tool/API responses
    text_tokens: Optional[Dict[str, Any]]        # Tokens from user query
    context_tokens: Optional[Dict[str, Any]]     # Tokens from conversation history/context

    # === OUTPUT (to user) ===
    response: str                                # Final answer
    response_id: Optional[str]                   # UUID for the response (for feedback tracking)
    recommendations: Optional[List[Dict[str, str]]]  # Recommendation chips for follow-up actions
    # recommendations structure:
    # [
    #     {"text": "View claim details", "action": "claim_details"},
    #     {"text": "Check my benefits", "action": "benefits_info"}
    # ]
    # NOTE: claim_list action is blocked — see BLOCKED_RECOMMENDATION_ACTIONS in response_agent.py

    # === RECOMMENDATION DEDUP TRACKING ===
    asked_questions_by_claim: Optional[Dict[str, List[str]]]  # Tracks questions asked per claim key
    # Structure: {"claimNumber_sequenceNumber": ["question1 text", "question2 text"]}
    # Used by response_agent to avoid recommending questions already asked for the SAME claim.
    # Different claim key = fresh tracking, so same question CAN be recommended for a new claim.

    # === METADATA ===
    metadata: Annotated[Dict[str, Any], merge_metadata]  # Tracking info (merged across nodes)
    cache_hit: bool                              # From cache?
    error: Optional[str]                         # Any error

    # === LANGGRAPH MESSAGES (for checkpointing) ===
    messages: Annotated[List[BaseMessage], add_messages]  # Message history

def create_initial_state(
    text: str,
    session_id: str,
    user_info: Dict[str, Any] = None,
    user_session: Optional[str] = None        # UI-provided stable session ID; None until UI provides it
) -> AgentState:
    """
    Create starting state for new request

    Think of this as filling out the top of a form before
    sending it down the assembly line.
    """
    return AgentState(
        text=text,
        session_id=session_id,
        user_session=user_session,            # Flows through graph untouched; used only by update_memory_node for MongoDB
        user_info=user_info or {},
        uuid=None,
        domain=None,
        intent=None,
        confidence=None,
        llm_fallback_confidence=None,
        ensemble_intent=None,
        ensemble_confidence=None,
        llm_fallback_intent=None,
        entities=None,
        slots=None,
        required_slots=None,
        missing_slots=None,
        # API Routing (from config)
        api_endpoint=None,
        required_entities_list=None,
        requires_llm=False,
        is_complex=False,
        embedding_failed=False,
        intent_reclassified=False,  # Initial classification, not yet reclassified by LLM judge
        # Other fields
        needs_clarification=False,
        clarifying_question=None,
        clarification_context=None,
        conversation_history=[],
        relevant_facts=[],
        extracted_slots=None,
        planner_context=None,
        tool_results=None,
        api_error=None,
        api_retry_count=0,
        safety_precheck_passed=False,
        safety_postcheck_passed=False,
        safety_block_reason=None,
        tool_tokens=None,
        text_tokens=None,
        context_tokens=None,
        response="",
        response_id=None,
        recommendations=[],              # Initialize empty recommendations list
        asked_questions_by_claim={},      # Initialize empty dedup tracking
        metadata={},
        cache_hit=False,
        error=None,
        messages=[]
    )
