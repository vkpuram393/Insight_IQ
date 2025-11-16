"""
Context Building Node - Gather conversation background
"""

import traceback
from typing import Dict, Any
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger
from core.error_models import create_internal_error
from persistence import PersistenceStoreFactory
from memory import MemoryStoreFactory

logger = get_logger(__name__)

# Get memory store instance (facade pattern)
_memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)

async def build_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Build context from conversation history

    🎓 CONCEPT:
    Before processing, gather:
    - Recent messages (short-term memory)
    - Important facts (long-term memory)

    This gives agents context about the conversation.
    """
    node_name = "build_context"
    session_id = state.get("session_id", "unknown")
    request_id = state.get("uuid")
    user_id = state.get("user_info", {}).get("user_id")
    
    try:
        logger.info("🧠 Node: Build Context")

        session_id = state["session_id"]

        # Get conversation history from memory store
        history = await _memory_store.get_session_history(session_id)

        # Get relevant facts from memory store
        facts = await _memory_store.get_session_facts(session_id)

        logger.debug(f"Context: {len(history)} messages, {len(facts)} facts")

        return {
            "conversation_history": history,
            "relevant_facts": facts
        }
        
    except Exception as e:
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Context building failed: {str(e)}",
            stacktrace=tb,
            session_id=session_id,
            node_name=node_name
        )
        
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        await persistence_store.log_exception(
            error_code=error.error_code.value,
            category=error.category.value,
            severity=error.severity.value,
            message=error.message,
            user_message=error.user_message,
            session_id=session_id,
            request_id=request_id,
            node_name=node_name,
            stacktrace=error.stacktrace,
            metadata=error.metadata,
            user_id=user_id
        )
        
        logger.error(f"🚨 Exception in context building: {e}\n{tb}")
        
        return {
            "error": error.user_message,
            "conversation_history": [],
            "relevant_facts": [],
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }

async def update_memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Store conversation in memory and return updated context.

    After generating response, save it so we remember next time.
    """
    node_name = "update_memory"
    session_id = state.get("session_id", "unknown")
    request_id = state.get("uuid")
    user_id = state.get("user_info", {}).get("user_id")
    
    try:
        logger.info("💾 Node: Update Memory")

        session_id = state["session_id"]

        # Append user message to memory store
        await _memory_store.append_to_session(
            session_id=session_id,
            role="user",
            content=state["text"],
            max_messages=10
        )

        # Append assistant response to memory store
        await _memory_store.append_to_session(
            session_id=session_id,
            role="assistant",
            content=state.get("response", ""),
            max_messages=10
        )

        # Update long-term memory (extract important facts)
        if "claim" in state["text"].lower():
            await _memory_store.add_session_fact(
                session_id=session_id,
                fact_type="claim_mention",
                data={"text": state["text"]}
            )

        # Get updated context
        updated_history = await _memory_store.get_session_history(session_id)
        updated_facts = await _memory_store.get_session_facts(session_id)

        logger.info("✅ Memory updated")
        return {
            "conversation_history": updated_history,
            "relevant_facts": updated_facts,
            "metadata": {**state.get("metadata", {}), "memory_updated": True}
        }
        
    except Exception as e:
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Memory update failed: {str(e)}",
            stacktrace=tb,
            session_id=session_id,
            node_name=node_name
        )
        
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        await persistence_store.log_exception(
            error_code=error.error_code.value,
            category=error.category.value,
            severity=error.severity.value,
            message=error.message,
            user_message=error.user_message,
            session_id=session_id,
            request_id=request_id,
            node_name=node_name,
            stacktrace=error.stacktrace,
            metadata=error.metadata,
            user_id=user_id
        )
        
        logger.error(f"🚨 Exception in memory update: {e}\n{tb}")
        
        return {
            "error": error.user_message,
            "conversation_history": state.get("conversation_history", []),
            "relevant_facts": state.get("relevant_facts", []),
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value,
                "memory_updated": False
            }
        }
