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
from core.config import settings
from core.logger import get_logger
from core.error_models import create_internal_error
from core.logging_context import extract_logging_context, log_state_snapshot
from persistence import PersistenceStoreFactory
from memory import MemoryStoreFactory

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
            "conversation_history_window": 5
        }

def _extract_slots_from_history(history: List[Dict[str, str]], current_slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract slots/entities from conversation history that might not be in current message.
    
    Uses simple regex patterns to find entities in previous messages.
    This supplements what the intent classifier found in the current message.
    
    NOTE: This is a generic extraction that looks for common patterns.
    For domain-specific extraction, this should be enhanced or moved to domain-specific modules.
    """
    extracted = {}
    
    # Combine all history messages into text
    history_text = " ".join([msg.get("content", "") for msg in history if isinstance(msg, dict)])
    
    # Generic patterns for common entity types (domain-agnostic)
    # These patterns can be extended or moved to domain-specific config
    
    # Pattern 1: Numeric IDs (4-10 digits) preceded by common keywords
    # This catches: claim number, prescription number, order number, etc.
    numeric_id_pattern = r"(?:(?:claim|prescription|order|ref|id|number|#)\s*(?:number|id|#)?\s*:?\s*)?(\d{4,10})\b"
    numeric_matches = re.findall(numeric_id_pattern, history_text, re.IGNORECASE)
    if numeric_matches:
        # Use the most recent numeric ID found
        # Note: This is generic - domain-specific logic should determine which slot name to use
        # For now, we'll try to infer from context
        if "claim" in history_text.lower():
            extracted["claim_number"] = numeric_matches[-1]
        elif "prescription" in history_text.lower() or "rx" in history_text.lower():
            extracted["prescription_number"] = numeric_matches[-1]
        else:
            # Generic numeric ID
            extracted["numeric_id"] = numeric_matches[-1]
    
    # Pattern 2: Alphanumeric IDs (6-20 chars) - member IDs, patient IDs, etc.
    alphanumeric_id_pattern = r"(?:(?:member|patient|user|account)\s*(?:id|number|#)?\s*:?\s*)?([A-Z0-9\-]{6,20})\b"
    alphanumeric_matches = re.findall(alphanumeric_id_pattern, history_text, re.IGNORECASE)
    if alphanumeric_matches:
        extracted["member_id"] = alphanumeric_matches[-1]
    
    # Pattern 3: Dates (various formats)
    date_pattern = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
    date_matches = re.findall(date_pattern, history_text)
    if date_matches:
        extracted["date"] = date_matches[-1]
    
    # Merge with current slots (current slots take precedence)
    merged_slots = {**extracted, **current_slots}
    
    return merged_slots

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
        history_window = config.get("conversation_history_window", 5)
        
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
        
        return {
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

async def update_memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Store conversation in memory and return updated context.

    After generating response, save it so we remember next time.
    """
    node_name = "update_memory"
    log_ctx = extract_logging_context(state)
    
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
