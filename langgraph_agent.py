"""
LangGraph Agent - THE GRAPH DEFINITION

🎯 THIS IS THE HEART OF THE SYSTEM!
"""

from typing import Optional
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from state.schema import AgentState
from nodes import (
    orchestrator_node,
    safety_precheck_node,
    check_cache_node,
    build_context_node,
    clarification_node,
    confidence_check_router,
    confidence_checker_node,
    llm_judge_node,
    update_memory_node,
    cache_response_node,
    response_safety_pii_precheck_node,
    response_safety_pii_postcheck_node
)
from agents import intent_agent_node, response_agent_node
from tools import call_claims_tool_node
from config.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# Global objects
_graph_compiled = None
_async_saver_cm = None   # context manager
_async_saver = None       # actual saver instance
_init_lock = asyncio.Lock()

# Routers -------------------------------------------------------------------

def should_continue_after_precheck(state: AgentState) -> str:
    """
    After safety precheck, decide next step

    ROUTING:
        Blocked → END (return error)
        Passed → check_cache (continue)
    """
    if not state.get("safety_precheck_passed", False):
        return END
    return "check_cache"

def should_continue_after_cache(state: AgentState) -> str:
    """
    After cache check, decide next step

    ROUTING:
        Cache HIT or MISS → intent_agent (cache not fully implemented yet)
    """
    # Cache not fully implemented - always go to intent_agent
    return "intent_agent"


def route_after_response_postcheck(state: AgentState) -> str:
    """
    After response safety PII postcheck, decide next step
    
    ROUTING:
        If intent_reclassified == True AND no response yet (coming from llm_judge): route to confidence_checker for re-evaluation
        If response exists (coming from response_agent): route to update_memory (normal response flow)
    """
    intent_reclassified = state.get("intent_reclassified", False)
    response = state.get("response", "")
    # Check for non-empty response (empty string should be treated as no response)
    has_response = response is not None and len(str(response).strip()) > 0
    
    # If we have a non-empty response, always go to update_memory (normal flow)
    if has_response:
        logger.info("✅ Response generated - routing to update_memory")
        return "update_memory"
    
    # No response yet - check if coming from LLM judge
    if intent_reclassified:
        logger.info("🔄 Coming from LLM judge (no response yet) - routing to confidence_checker for re-evaluation")
        return "confidence_checker"
    else:
        logger.info("⚠️ No response and not from LLM judge - routing to update_memory (fallback)")
        return "update_memory"

def route_after_update_memory(state: AgentState) -> str:
    """
    After updating memory, decide next step
    
    ROUTING:
        If this is a clarification response → END (skip cache, not useful)
        If this is a normal response → cache_response (cache the final answer)
    """
    # Check if this is a clarification (has needs_clarification flag)
    if state.get("needs_clarification", False):
        logger.info("📝 Clarification response - skipping cache, going to END")
        return END
    
    # Normal response - cache it
    logger.info("💾 Normal response - caching before END")
    return "cache_response"

# Workflow builder -----------------------------------------------------------

def _build_workflow() -> StateGraph:
    """
    Build the complete workflow with unified safety check
    
    FLOW:
        User Query
            ↓
        orchestrator
            ↓
        [safety_precheck] ← Unified safety (pattern check + mask + Gemini + unmask)
            ↓
        check_cache
            ↓
        intent_agent (processes with PII/PHI intact)
            ↓
        confidence_checker
            ↓
        [router] → clarification OR build_context OR llm_judge
            ↓
        Option 1: clarification (low confidence/missing entities)
            ↓
        [response_safety_pii_precheck] ← Mask PII/PHI before LLM
            ↓
        response_agent (generates follow-up question)
            ↓
        [response_safety_pii_postcheck] ← Unmask & check leakage
            ↓
        update_memory → END (skip cache for clarifications)
            ↓
        Option 2: build_context → call_claims_tool (normal queries)
            ↓
        [response_safety_pii_precheck] ← Mask PII/PHI before LLM
            ↓
        response_agent (generates claim response)
            ↓
        [response_safety_pii_postcheck] ← Unmask & check leakage
            ↓
        update_memory → cache_response → END
    """
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("safety_precheck", safety_precheck_node)
    workflow.add_node("safety_precheck_for_llm", response_safety_pii_precheck_node)  # PII-only masking for LLM judge path
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("cache_response", cache_response_node)
    workflow.add_node("intent_agent", intent_agent_node)
    workflow.add_node("confidence_checker", confidence_checker_node)
    workflow.add_node("llm_judge", llm_judge_node)
    workflow.add_node("build_context", build_context_node)
    workflow.add_node("response_safety_pii_precheck", response_safety_pii_precheck_node)
    workflow.add_node("response_agent", response_agent_node)
    workflow.add_node("response_safety_pii_postcheck", response_safety_pii_postcheck_node)
    workflow.add_node("call_claims_tool", call_claims_tool_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("update_memory", update_memory_node)

    # Build the flow
    # Entry point
    workflow.set_entry_point("orchestrator")
    
    # Orchestrator → Safety Precheck
    workflow.add_edge("orchestrator", "safety_precheck")
    
    # Safety Precheck → Cache or END
    workflow.add_conditional_edges(
        "safety_precheck", should_continue_after_precheck, {"check_cache": "check_cache", END: END}
    )
    
    # Cache → Intent Agent (cache not fully implemented, always routes to intent_agent)
    workflow.add_conditional_edges(
        "check_cache", should_continue_after_cache, {"intent_agent": "intent_agent"}
    )
    
    # Intent Agent → Confidence Checker
    workflow.add_edge("intent_agent", "confidence_checker")
    
    # Confidence Checker → Routing (updated to include llm_judge and direct response)
    # - llm_judge: Low confidence/complex AND intent_reclassified == False
    # - clarification: Missing entities OR low confidence after LLM judge
    # - build_context: High confidence + has entities (needs API call)
    # - response_safety_pii_precheck: Greeting/help/out_of_scope (no API needed, direct to LLM)
    workflow.add_conditional_edges(
        "confidence_checker", 
        confidence_check_router, 
        {
            "llm_judge": "safety_precheck_for_llm",  # Route through PII masking before llm_judge
            "clarification": "clarification",
            "build_context": "build_context",
            "response_safety_pii_precheck": "response_safety_pii_precheck"  # Direct path (no API)
        }
    )
    
    # Safety Precheck (for LLM Judge) → LLM Judge (PII-only masking, always continues)
    workflow.add_edge("safety_precheck_for_llm", "llm_judge")
    
    # LLM Judge → Response Safety PII Postcheck → Confidence Checker (unmask PII before re-evaluation)
    workflow.add_edge("llm_judge", "response_safety_pii_postcheck")
    
    # Clarification → Response Safety PII Precheck → Response Agent (LLM generates follow-up question)
    workflow.add_edge("clarification", "response_safety_pii_precheck")
    
    # Build Context → Call Claims Tool
    workflow.add_edge("build_context", "call_claims_tool")
    
    # Tool Call → Response Safety PII Precheck → Response Agent → Response Safety PII Postcheck
    workflow.add_edge("call_claims_tool", "response_safety_pii_precheck")
    workflow.add_edge("response_safety_pii_precheck", "response_agent")
    workflow.add_edge("response_agent", "response_safety_pii_postcheck")
    
    # Response Safety PII Postcheck → Conditional routing
    # - If coming from llm_judge: route to confidence_checker (for re-evaluation)
    # - If coming from response_agent: route to update_memory (normal response flow)
    workflow.add_conditional_edges(
        "response_safety_pii_postcheck",
        route_after_response_postcheck,
        {
            "confidence_checker": "confidence_checker",
            "update_memory": "update_memory"
        }
    )
    
    # Update Memory → Conditional routing (clarification skips cache, normal responses cache)
    workflow.add_conditional_edges(
        "update_memory",
        route_after_update_memory,
        {
            "cache_response": "cache_response",
            END: END
        }
    )
    
    # Cache Response → END
    workflow.add_edge("cache_response", END)
    
    return workflow

# Lifecycle ------------------------------------------------------------------

async def init_graph():
    """Initialize graph + persistent async sqlite saver once."""
    global _graph_compiled, _async_saver_cm, _async_saver
    if _graph_compiled is not None:
        return _graph_compiled
    async with _init_lock:
        if _graph_compiled is not None:
            return _graph_compiled
        logger.info("📊 Initializing LangGraph AsyncSqliteSaver (persistent)...")
        _async_saver_cm = AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path)
        _async_saver = await _async_saver_cm.__aenter__()  # enter context once
        workflow = _build_workflow()
        _graph_compiled = workflow.compile(checkpointer=_async_saver)
        logger.info("✅ Graph initialized")
    return _graph_compiled

async def close_graph():
    """Cleanly close async saver on shutdown."""
    global _async_saver_cm, _async_saver, _graph_compiled
    if _async_saver_cm:
        logger.info("🧹 Closing LangGraph AsyncSqliteSaver...")
        await _async_saver_cm.__aexit__(None, None, None)
        _async_saver_cm = None
        _async_saver = None
        _graph_compiled = None
        logger.info("✅ Saver closed")

# Execution ------------------------------------------------------------------

async def run_graph(text: str, session_id: str, user_info: dict = None):
    from state.schema import create_initial_state
    await init_graph()
    initial_state = create_initial_state(text, session_id, user_info)
    config = {"configurable": {"thread_id": session_id}}
    final_state = await _graph_compiled.ainvoke(initial_state, config)  # type: ignore
    return final_state

async def run_graph_stream(
    text: str, 
    session_id: str, 
    user_info: dict = None
):
    """
    Execute graph and stream events in real-time.
    
    🎯 STREAMING STRATEGY:
    1. Execute graph normally (no changes to nodes/agents)
    2. Track node execution via astream
    3. Send status updates ONLY for user-facing nodes (reduces noise)
    4. Log ALL nodes internally for telemetry (full observability maintained)
    5. Wait for safety_postcheck to complete
    6. Stream validated response in chunks
    7. Send completion event with metadata
    
    🔒 SECURITY GUARANTEE:
    Response chunks are only streamed AFTER response_safety_pii_postcheck
    completes. This ensures:
    - PII/PHI leakage detection has run
    - Masked tokens are unmasked
    - Content is safe for user consumption
    
    📊 EVENT FLOW (User-Facing):
    ```
    node_start: "orchestrator" → "Processing your request..."
    node_start: "safety_precheck" → "Checking safety and privacy..."
    node_start: "intent_agent" → "Understanding your question..."
    node_start: "call_claims_tool" → "Retrieving your claims information..."
    node_start: "response_agent" → "Preparing your response..."
    response_chunk: "Your claim..." ← STREAMING STARTS HERE
    response_chunk: " for Lisinopril..."
    response_chunk: " was approved..."
    complete: {response: "...", intent: "...", confidence: 0.95}
    ```
    
    All nodes are still executed and logged internally. Only significant
    milestones are shown to users to reduce noise and improve UX.
    
    Args:
        text: User's query
        session_id: Conversation session ID
        user_info: User metadata (user_id, etc.)
        
    Yields:
        Dict[str, Any]: Event dictionaries with structure:
        {
            "type": "node_start" | "node_complete" | "response_chunk" | "complete" | "error",
            "data": <event-specific data>,
            "metadata": {"timestamp": "...", "node": "...", "user_facing": bool, ...}
        }
    """
    from state.schema import create_initial_state
    from datetime import datetime, timezone
    
    await init_graph()
    
    initial_state = create_initial_state(text, session_id, user_info)
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        logger.info(f"🌊 Starting streaming execution for session {session_id}")
        
        # Track state accumulation
        current_state = {}
        final_response = ""
        safety_postcheck_complete = False
        
        # Use astream to get state updates as graph executes
        # Note: Using astream (not astream_events) for simplicity and state tracking
        async for state_update in _graph_compiled.astream(initial_state, config, stream_mode="updates"):
            # state_update is a dict with node_name -> output
            for node_name, node_output in state_update.items():
                
                # ================================================================
                # INTERNAL LOGGING (ALL NODES - Full Observability)
                # ================================================================
                # Always log every node execution for debugging and telemetry
                # This maintains full observability regardless of user-facing config
                logger.info(f"📍 Node executing: {node_name}")
                logger.debug(f"   Node output keys: {list(node_output.keys())}")
                
                # ================================================================
                # USER-FACING STATUS UPDATES (SELECTIVE - Better UX)
                # ================================================================
                # Only send status updates for user-facing nodes (reduces noise)
                # Other nodes are logged above but not shown to end users
                should_show_to_user = node_name in settings.stream_user_facing_nodes
                
                if settings.stream_node_updates and should_show_to_user:
                    status_message = _get_status_message(node_name)
                    
                    # Log that we're sending user-facing update
                    logger.debug(f"   → Sending user-facing update: '{status_message}'")
                    
                    yield {
                        "type": "node_start",
                        "data": {
                            "node": node_name,
                            "message": status_message
                        },
                        "metadata": {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "node": node_name,
                            "user_facing": True  # Mark as user-facing for analytics
                        }
                    }
                else:
                    # Node executed but not shown to user (internal only)
                    logger.debug(f"   → Internal node (not shown to user)")
                
                # Update accumulated state
                current_state.update(node_output)
                
                # Check if this is the safety postcheck completing
                if node_name == "response_safety_pii_postcheck":
                    safety_postcheck_passed = node_output.get("safety_postcheck_passed", False)
                    final_response = current_state.get("response", "")
                    
                    logger.info(f"🔒 Safety postcheck complete: passed={safety_postcheck_passed}")
                    
                    if not safety_postcheck_passed:
                        # Safety violation - send error and stop
                        error_message = node_output.get("response", "Content safety violation detected")
                        logger.warning(f"🚫 Safety violation: {error_message}")
                        
                        yield {
                            "type": "error",
                            "data": {
                                "message": error_message,
                                "reason": "safety_violation"
                            },
                            "metadata": {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "node": node_name
                            }
                        }
                        return  # Stop streaming
                    
                    # Safety passed - NOW we can stream the response!
                    safety_postcheck_complete = True
                    
                    if final_response:
                        logger.info(f"✅ Safety validated - streaming {len(final_response)} chars to user")
                        
                        # Stream response in chunks for better UX
                        chunk_size = settings.streaming_chunk_size
                        
                        for i in range(0, len(final_response), chunk_size):
                            chunk = final_response[i:i + chunk_size]
                            yield {
                                "type": "response_chunk",
                                "data": {
                                    "text": chunk,
                                    "chunk_index": i // chunk_size,
                                    "total_length": len(final_response)
                                },
                                "metadata": {
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "node": "response_agent"
                                }
                            }
                            
                            # Optional: Small delay for smoother streaming effect (configurable)
                            if settings.streaming_delay_ms > 0:
                                await asyncio.sleep(settings.streaming_delay_ms / 1000.0)
                
                # ================================================================
                # NODE COMPLETE EVENTS (OPTIONAL - For Progress Tracking)
                # ================================================================
                # Send node_complete only for user-facing nodes
                # This can be used by frontend for progress bars or step indicators
                if settings.stream_node_updates and should_show_to_user:
                    yield {
                        "type": "node_complete",
                        "data": {
                            "node": node_name
                        },
                        "metadata": {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "node": node_name,
                            "user_facing": True
                        }
                    }
        
        # Graph execution complete - send final event with all metadata
        logger.info(f"✅ Graph execution complete for session {session_id}")
        
        yield {
            "type": "complete",
            "data": {
                "response": final_response,
                "intent": current_state.get("intent"),
                "confidence": current_state.get("confidence"),
                "needs_clarification": current_state.get("needs_clarification", False),
                "clarifying_question": current_state.get("clarifying_question"),
                "metadata": current_state.get("metadata", {})
            },
            "metadata": {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"🚨 Graph streaming error: {e}", exc_info=True)
        yield {
            "type": "error",
            "data": {
                "message": str(e),
                "error_type": type(e).__name__
            },
            "metadata": {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }


def _get_status_message(node_name: str) -> str:
    """
    Get user-friendly status message for each node.
    
    These messages are shown to users in real-time as the graph executes,
    providing transparency and reducing perceived latency.
    
    ✨ INDUSTRY BEST PRACTICES:
    - Keep messages concise (under 50 characters)
    - Use active present tense ("Processing..." not "Will process...")
    - Focus on user benefit, not technical details
    - Be consistent in tone and style
    - Match patterns from ChatGPT, Claude, Copilot, Gemini, etc.
    
    Args:
        node_name: Name of the node (e.g., "safety_precheck")
        
    Returns:
        str: User-friendly status message
    """
    # User-facing messages (shown to end users)
    # These are carefully crafted to be non-technical and reassuring
    status_messages = {
        # ----------------------------------------------------------------
        # USER-FACING NODES (shown to end users)
        # ----------------------------------------------------------------
        # Initial processing - Let user know we received their request
        "orchestrator": "Processing your request...",
        
        # Safety check - Show we care about privacy (builds trust)
        "safety_precheck": "Checking safety and privacy...",
        
        # Intent understanding - Show AI is comprehending the question
        "intent_agent": "Understanding your question...",
        
        # Data retrieval - Most important message (users care about this!)
        "call_claims_tool": "Retrieving your claims information...",
        
        # Response generation - Final step before streaming answer
        "response_agent": "Preparing your response...",
        
        # ----------------------------------------------------------------
        # INTERNAL NODES (not shown to users but listed here for reference)
        # These are still executed and logged, just not displayed
        # ----------------------------------------------------------------
        "check_cache": "Checking for cached response...",  # Internal optimization
        "confidence_checker": "Analyzing request confidence...",  # Internal logic
        "build_context": "Building conversation context...",  # Internal processing
        "response_safety_pii_precheck": "Preparing response data...",  # Internal security
        "response_safety_pii_postcheck": "Validating response safety...",  # Internal security
        "update_memory": "Updating conversation memory...",  # Internal storage
        "cache_response": "Caching response...",  # Internal optimization
        "clarification": "Preparing clarifying question...",  # User-facing but rare
    }
    
    # Return user-friendly message or generic fallback
    return status_messages.get(node_name, f"Processing {node_name}...")


