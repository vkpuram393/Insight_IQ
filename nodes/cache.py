"""
Cache Nodes - Speed up repeated questions
"""

import hashlib
import json
import traceback
from typing import Dict, Any
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger
from core.error_models import create_internal_error
from core.logging_context import extract_logging_context
from persistence import PersistenceStoreFactory
from memory import MemoryStoreFactory

logger = get_logger(__name__)

# Get memory store instance (facade pattern)
_memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)

def _hash(text: str) -> str:
    """Create hash of text for caching"""
    return hashlib.md5(text.lower().strip().encode()).hexdigest()

async def check_cache_node(state: AgentState) -> Dict[str, Any]:
    """
    Check if we've answered this before

    🎓 CONCEPT:
    If user asks same question, return cached answer instantly!
    No need to run through all nodes again.

    FLOW:
        Cache HIT → Return cached response, end graph
        Cache MISS → Continue to next node
    """
    node_name = "check_cache"
    log_ctx = extract_logging_context(state)
    
    try:
        logger.info("💾 Node: Check Cache")

        if not settings.enable_semantic_cache:
            return {"cache_hit": False}

        key = f"cache:{_hash(state['text'])}"

        # Use memory store facade
        cached_value = await _memory_store.get(key)

        if cached_value:
            cached = json.loads(cached_value) if isinstance(cached_value, str) else cached_value
            logger.info("🎯 Cache HIT!")
            return {
                "response": cached["response"],
                "intent": cached["intent"],
                "confidence": cached["confidence"],
                "cache_hit": True,
                "metadata": {**state.get("metadata", {}), "cache": "hit"}
            }

        logger.info("💨 Cache MISS")
        return {"cache_hit": False, "metadata": {**state.get("metadata", {}), "cache": "miss"}}
        
    except Exception as e:
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Cache check failed: {str(e)}",
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
        
        logger.error(f"🚨 Exception in cache check: {e}\n{tb}")
        
        return {
            "error": error.user_message,
            "cache_hit": False,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }

async def cache_response_node(state: AgentState) -> Dict[str, Any]:
    """Store response in cache for future"""
    node_name = "cache_response"
    log_ctx = extract_logging_context(state)
    
    try:
        logger.info("💾 Node: Cache Response")

        if settings.enable_semantic_cache:
            key = f"cache:{_hash(state['text'])}"
            cache_data = {
                "response": state.get("response", ""),
                "intent": state.get("intent"),
                "confidence": state.get("confidence")
            }

            # Store in memory store with 1 hour TTL
            await _memory_store.set(key, json.dumps(cache_data), ttl_seconds=3600)
            logger.info("✅ Response cached")

        # Always return metadata to register a change
        return {"metadata": {**state.get("metadata", {}), "cached": settings.enable_semantic_cache}}
        
    except Exception as e:
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Cache response failed: {str(e)}",
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
        
        logger.error(f"🚨 Exception in cache response: {e}\n{tb}")
        
        return {
            "error": error.user_message,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value,
                "cached": False
            }
        }
