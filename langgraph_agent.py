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
    safety_precheck_node,
    check_cache_node,
    build_context_node,
    clarification_node,
    confidence_check_router,
    route_after_api_call,  # NEW: API error fallback routing
    safety_postcheck_node,
    update_memory_node,
    cache_response_node,
    master_llm_agent_node  # NEW: Master LLM Agent (Stage 2)
)
from agents import intent_agent_node, response_agent_node
from tools import call_claims_tool_node
from core.config import settings
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
        Cache HIT → END (return cached response)
        Cache MISS → build_context (continue processing)
    """
    if state.get("cache_hit", False):
        return END
    return "build_context"

# Workflow builder -----------------------------------------------------------

def _build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("safety_precheck", safety_precheck_node)
    workflow.add_node("safety_postcheck", safety_postcheck_node)
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("cache_response", cache_response_node)
    workflow.add_node("build_context", build_context_node)
    workflow.add_node("intent_agent", intent_agent_node)
    workflow.add_node("master_llm", master_llm_agent_node)  # NEW: Master LLM Agent (Stage 2)
    workflow.add_node("response_agent", response_agent_node)
    workflow.add_node("call_claims_tool", call_claims_tool_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("update_memory", update_memory_node)

    workflow.set_entry_point("safety_precheck")
    workflow.add_conditional_edges(
        "safety_precheck", should_continue_after_precheck, {"check_cache": "check_cache", END: END}
    )
    workflow.add_conditional_edges(
        "check_cache", should_continue_after_cache, {"build_context": "build_context", END: END}
    )
    workflow.add_edge("build_context", "intent_agent")
    
    # NEW: Three-way routing from intent_agent
    # - tool_call: Stage 1 caught API query → go to API
    # - master_llm: Stage 1 uncertain → go to Master LLM Agent (Stage 2)
    # - clarification: Stage 1 knows we need clarification
    workflow.add_conditional_edges(
        "intent_agent", 
        confidence_check_router, 
        {
            "clarification": "clarification", 
            "tool_call": "call_claims_tool",
            "master_llm": "master_llm"  # NEW: Route to Master LLM Agent
        }
    )
    
    # NEW: Master LLM Agent can reroute to API or respond directly
    def route_after_master_llm(state: AgentState) -> str:
        """Route after Master LLM Agent decision"""
        if state.get("needs_api_reroute"):
            logger.info("🔄 Master LLM rerouting to API path!")
            return "call_claims_tool"
        elif state.get("response"):
            # LLM provided a direct response (greeting, clarification, etc.)
            logger.info("💬 Master LLM provided direct response → skipping to postcheck")
            return "safety_postcheck"
        else:
            # Default: go to response agent
            return "response_agent"
    
    workflow.add_conditional_edges(
        "master_llm",
        route_after_master_llm,
        {
            "call_claims_tool": "call_claims_tool",
            "response_agent": "response_agent",
            "safety_postcheck": "safety_postcheck"
        }
    )
    
    workflow.add_edge("clarification", "update_memory")  # CHANGED: clarification no longer ends graph directly
    
    # NEW: API error handling with LLM fallback (CRITICAL for multiple APIs!)
    workflow.add_conditional_edges(
        "call_claims_tool",
        route_after_api_call,
        {
            "master_llm": "master_llm",          # LLM fallback on API failure
            "response_agent": "response_agent"   # Success
        }
    )
    
    workflow.add_edge("response_agent", "safety_postcheck")
    workflow.add_edge("safety_postcheck", "update_memory")
    workflow.add_edge("update_memory", "cache_response")
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
