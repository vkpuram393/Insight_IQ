"""
Context Building Node - Gather conversation background
"""

from typing import Dict, Any
from state.schema import AgentState
from core.logger import get_logger

logger = get_logger(__name__)

# Simple in-memory storage (production: database)
_short_term = {}
_long_term = {}

async def build_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Build context from conversation history

    🎓 CONCEPT:
    Before processing, gather:
    - Recent messages (short-term memory)
    - Important facts (long-term memory)

    This gives agents context about the conversation.
    """

    logger.info("🧠 Node: Build Context")

    session_id = state["session_id"]

    # Get conversation history
    history = _short_term.get(session_id, [])

    # Get relevant facts
    facts = _long_term.get(session_id, [])

    logger.debug(f"Context: {len(history)} messages, {len(facts)} facts")

    return {
        "conversation_history": history,
        "relevant_facts": facts
    }

async def update_memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Store conversation in memory and return updated context.

    After generating response, save it so we remember next time.
    """

    logger.info("💾 Node: Update Memory")

    session_id = state["session_id"]

    # Update short-term memory
    if session_id not in _short_term:
        _short_term[session_id] = []

    # Append latest user and assistant turns
    _short_term[session_id].append({
        "role": "user",
        "content": state["text"]
    })
    _short_term[session_id].append({
        "role": "assistant",
        "content": state.get("response", "")
    })

    # Keep only recent 10 messages
    _short_term[session_id] = _short_term[session_id][-10:]

    # Update long-term memory (extract important facts)
    if "claim" in state["text"].lower():
        if session_id not in _long_term:
            _long_term[session_id] = []
        _long_term[session_id].append({
            "type": "claim_mention",
            "text": state["text"]
        })

    updated_history = _short_term[session_id]
    updated_facts = _long_term.get(session_id, [])

    logger.info("✅ Memory updated")
    return {
        "conversation_history": updated_history,
        "relevant_facts": updated_facts,
        "metadata": {**state.get("metadata", {}), "memory_updated": True}
    }
