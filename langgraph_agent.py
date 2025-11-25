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
    has_response = state.get("response") is not None
    
    # If we have a response, always go to update_memory (normal flow)
    if has_response:
        logger.info("✅ Response generated - routing to update_memory")
        return "update_memory"
    
    # No response yet - check if coming from LLM judge
    if intent_reclassified:
        logger.info("🔄 Coming from LLM judge - routing to confidence_checker for re-evaluation")
        return "confidence_checker"
    else:
        logger.info("✅ Normal response flow - routing to update_memory")
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
        [router] → clarification (END) OR build_context OR response_agent (complex)
            ↓
        build_context → call_claims_tool (simple queries with entities)
            ↓
        OR
            ↓
        response_agent (complex queries skip API)
            ↓
        [response_safety_pii_precheck] ← Mask PII/PHI again before LLM
            ↓
        response_agent (LLM - SAFE)
            ↓
        [response_safety_pii_postcheck] ← Unmask for user
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
    
    # Confidence Checker → Routing (updated to include llm_judge)
    # - llm_judge: Low confidence/complex AND intent_reclassified == False
    # - clarification: Missing entities OR low confidence after LLM judge
    # - build_context: High confidence + has entities
    workflow.add_conditional_edges(
        "confidence_checker", 
        confidence_check_router, 
        {
            "llm_judge": "safety_precheck_for_llm",  # Route through PII masking before llm_judge
            "clarification": "clarification",
            "build_context": "build_context"
        }
    )
    
    # Safety Precheck (for LLM Judge) → LLM Judge (PII-only masking, always continues)
    workflow.add_edge("safety_precheck_for_llm", "llm_judge")
    
    # LLM Judge → Response Safety PII Postcheck → Confidence Checker (unmask PII before re-evaluation)
    workflow.add_edge("llm_judge", "response_safety_pii_postcheck")
    
    # Clarification → Update Memory → END (template-based, no LLM call, skip cache)
    workflow.add_edge("clarification", "update_memory")
    
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


