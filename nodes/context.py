"""
Context Building Node - Gather conversation background and build planner context
"""

import re
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from state.schema import AgentState
from config.config import settings
from core.logger import get_logger
from core.errors.models import create_internal_error
from core.logging_context import extract_logging_context, log_state_snapshot
from persistence import PersistenceStoreFactory
from memory import MemoryStoreFactory
# ADDED: Import conversation context service for entity extraction from history
from services.conversation_context import extract_entities_from_history as service_extract_entities

logger = get_logger(__name__)

# Get memory store instance (facade pattern)
_memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)

# Load config cache
_config_cache = None

def _load_config() -> Dict[str, Any]:
    """Load domain config from JSON file"""
    global _config_cache
    config_path = Path(__file__).parent.parent / "config" / "domain_config.json"
    try:
        with open(config_path, 'r') as f:
            _config_cache = json.load(f)
        return _config_cache
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {
            "conversation_history_window": 50
        }

def _extract_slots_from_history(history: List[Dict[str, str]], current_slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract slots/entities from conversation history that might not be in current message.
    
    UPDATED: Delegates to ConversationContextService for pattern matching.
    This keeps pattern logic centralized in services/conversation_context.py
    
    Args:
        history: Conversation history messages
        current_slots: Entities from current message
        
    Returns:
        Merged entities (current takes precedence over history)
    """
    # ADDED: Delegate to conversation context service (pattern logic centralized)
    return service_extract_entities(history, current_slots)

async def build_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Build comprehensive context for planner/executor

    🎓 CONCEPT:
    Gathers all available information to build a complete context object:
    - Recent conversation history (last N messages, configurable)
    - Relevant facts from session
    - Extracted slots from conversation history
    - Intent, entities, slots from intent classifier
    - User profile information
    - Domain context
    
    This comprehensive context is passed to planner/executor to help them
    understand what API to call and what parameters to use.
    """
    node_name = "build_context"
    log_ctx = extract_logging_context(state)
    
    try:
        logger.info("🧠 Node: Build Context")

        # Load config
        config = _load_config()
        history_window = config.get("conversation_history_window", 50)
        
        session_id = state["session_id"]

        # Get conversation history from memory store
        full_history = await _memory_store.get_session_history(session_id)
        
        # Limit to last N messages (configurable)
        conversation_history = full_history[-history_window:] if len(full_history) > history_window else full_history

        # Get relevant facts from memory store
        relevant_facts = await _memory_store.get_session_facts(session_id)

        # Get current slots from state (from intent classifier)
        current_slots = state.get("slots") or {}
        
        # Extract additional slots from conversation history
        extracted_slots = _extract_slots_from_history(conversation_history, current_slots)
        
        # Get intent information from state
        intent = state.get("intent")
        confidence = state.get("confidence", 0.0)
        entities = state.get("entities") or {}
        required_slots = state.get("required_slots") or []
        missing_slots = state.get("missing_slots") or []
        
        # Build comprehensive planner context object
        planner_context = {
            "request_metadata": {
                "request_id": log_ctx["request_id"],
                "session_id": log_ctx["session_id"],
                "timestamp": datetime.now().isoformat(),
                "domain": state.get("domain")  # Domain from orchestrator, no default
            },
            "user": {
                "user_id": log_ctx["user_id"] or state.get("user_info", {}).get("user_id"),
                "profile": {}  # Placeholder for future user profile data
            },
            "intent": {
                "intent": intent,
                "confidence": confidence,
                "original_text": state.get("text", "")
            },
            "entities": entities,
            "slots": {
                "filled": extracted_slots,  # All slots (current + extracted from history)
                "missing": missing_slots,  # Required but not provided
                "required": required_slots,  # All required slots for this intent
                "extracted_from_history": {k: v for k, v in extracted_slots.items() if k not in current_slots}  # Only newly extracted ones
            },
            "conversation": {
                "history": conversation_history,  # Last N messages
                "relevant_facts": relevant_facts,  # All facts from session
                "history_length": len(conversation_history),
                "facts_count": len(relevant_facts)
            }
        }
        
        logger.info(f"📊 Context built: {len(conversation_history)} messages, {len(relevant_facts)} facts, {len(extracted_slots)} slots")
        logger.debug(f"   Extracted slots from history: {planner_context['slots']['extracted_from_history']}")

        # Return multiple fields (following existing pattern)
        result = {
            "conversation_history": conversation_history,
            "relevant_facts": relevant_facts,
            "extracted_slots": extracted_slots,
            "planner_context": planner_context
        }
        
        # Log full AgentState snapshot after this node
        await log_state_snapshot(state, node_name, result)
        
        return result
        
    except Exception as e:
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Context building failed: {str(e)}",
            stacktrace=tb,
            session_id=log_ctx["session_id"],
            node_name=node_name
        )
        
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        await persistence_store.log_exception(
            error_code=error.error_code.value,
            category=error.category.value,
            severity=error.severity.value,
            message=error.message,
            user_message=error.user_message,
            session_id=log_ctx["session_id"],
            request_id=log_ctx["request_id"],
            node_name=node_name,
            stacktrace=error.stacktrace,
            metadata=error.metadata,
            user_id=log_ctx["user_id"]
        )
        
        logger.error(f"🚨 Exception in context building: {e}\n{tb}")
        
        result = {
            "error": error.user_message,
            "conversation_history": [],
            "relevant_facts": [],
            "extracted_slots": {},
            "planner_context": None,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result

async def update_memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Store conversation in memory store and persistent storage.

    This node saves the conversation to:
    1. Memory store (Redis/InMemory/Memorystore) for real-time context
    2. Persistent database (conversation_history table) for long-term storage
    
    IMPORTANT: This runs AFTER response_safety_postcheck, so data is UNMASKED.
    
    What gets saved:
    - user_message: Original user query (UNMASKED)
    - agent_response: Final agent response (UNMASKED)
    - intent: Classified intent
    - tools_used: List of tools that were called
    - metadata: Additional context
    - duration_ms: Time taken for the request
    """
    node_name = "update_memory"
    log_ctx = extract_logging_context(state)
    
    try:
        print("\n" + "="*80)
        print("🔥 UPDATE_MEMORY_NODE CALLED! 🔥")
        print("="*80)
        
        # Get memory store type for accurate logging
        memory_store_type = settings.memory_store_type
        memory_store_name = type(_memory_store).__name__
        logger.info(f"💾 Node: Update Memory ({memory_store_type}/{memory_store_name} + persistent)")

        session_id = state["session_id"]
        user_id = log_ctx.get("user_id") or (state.get("user_info", {}) or {}).get("user_id")
        
        # Fix: Provide default user_id if none provided (database requires NOT NULL)
        if not user_id or user_id.strip() == "":
            user_id = "anonymous"  # Default user_id for requests without user info
        
        # Extract data from state
        # Use original_text from metadata if available (unmasked), otherwise use text
        metadata = state.get("metadata", {})
        user_message = metadata.get("original_text") or state.get("text", "")
        agent_response = state.get("response", "")
        intent = state.get("intent")
        tools_used = state.get("tools_used", [])
        
        # Calculate duration if available
        duration_ms = None
        if "start_time" in metadata and "end_time" in metadata:
            try:
                duration_ms = (metadata["end_time"] - metadata["start_time"]) * 1000
            except (TypeError, ValueError):
                duration_ms = None

        # 1. Update memory store (Redis/InMemory/Memorystore) for real-time context
        await _memory_store.append_to_session(
            session_id=session_id,
            role="user",
            content=user_message,
            max_messages=settings.conversation_history_limit
        )

        await _memory_store.append_to_session(
            session_id=session_id,
            role="assistant",
            content=agent_response,
            max_messages=settings.conversation_history_limit
        )

        # Update long-term memory facts (extract important facts)
        if "claim" in user_message.lower():
            await _memory_store.add_session_fact(
                session_id=session_id,
                fact_type="claim_mention",
                data={"text": user_message}
            )

        # 2. Save to persistent conversation_history table
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        
        print(f"\n📊 DATA TO SAVE:")
        print(f"   Session: {session_id}")
        print(f"   User: {user_id}")
        print(f"   Intent: {intent}")
        print(f"   Tools: {tools_used}")
        print(f"   User message length: {len(user_message)}")
        print(f"   Response length: {len(agent_response)}")
        print(f"   Duration: {duration_ms}ms" if duration_ms else "   Duration: N/A")
        
        logger.info(f"   💾 Saving to conversation_history table:")
        logger.info(f"      Session: {session_id}")
        logger.info(f"      User: {user_id}")
        logger.info(f"      Intent: {intent}")
        logger.info(f"      Tools: {tools_used}")
        logger.info(f"      Duration: {duration_ms}ms" if duration_ms else "      Duration: N/A")
        
        conversation_saved = False
        try:
            await persistence_store.save_conversation(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,      # UNMASKED - real data
                agent_response=agent_response,  # UNMASKED - real data
                intent=intent,
                tools_used=tools_used,
                metadata=metadata,
                duration_ms=duration_ms
            )
            logger.info("   ✅ Saved to conversation_history table")
            print("   ✅ SAVE SUCCESSFUL!")
            conversation_saved = True
        except Exception as save_error:
            logger.error(f"   ❌ Failed to save conversation: {save_error}")
            print(f"   ❌ SAVE FAILED: {save_error}")
            traceback.print_exc()
            # Continue execution even if save fails - don't break the workflow

        # Get updated context from memory store
        updated_history = await _memory_store.get_session_history(session_id)
        updated_facts = await _memory_store.get_session_facts(session_id)
        
        logger.info(f"✅ Memory updated ({memory_store_type}/{memory_store_name} + persistent={conversation_saved})")
        result = {
            "conversation_history": updated_history,
            "relevant_facts": updated_facts,
            "metadata": {
                **metadata, 
                "memory_updated": True,
                "conversation_saved": conversation_saved
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result
        
    except Exception as e:
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Memory update failed: {str(e)}",
            stacktrace=tb,
            session_id=log_ctx["session_id"],
            node_name=node_name
        )
        
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        await persistence_store.log_exception(
            error_code=error.error_code.value,
            category=error.category.value,
            severity=error.severity.value,
            message=error.message,
            user_message=error.user_message,
            session_id=log_ctx["session_id"],
            request_id=log_ctx["request_id"],
            node_name=node_name,
            stacktrace=error.stacktrace,
            metadata=error.metadata,
            user_id=log_ctx["user_id"]
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
