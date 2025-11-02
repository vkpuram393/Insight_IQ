"""
Cache Nodes - Speed up repeated questions
"""

import hashlib
from typing import Dict, Any, Optional
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# Simple in-memory cache (in production: Redis)
_cache: Dict[str, Dict[str, Any]] = {}

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

    logger.info("💾 Node: Check Cache")

    if not settings.enable_semantic_cache:
        return {"cache_hit": False}

    key = _hash(state["text"])

    if key in _cache:
        cached = _cache[key]
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

async def cache_response_node(state: AgentState) -> Dict[str, Any]:
    """Store response in cache for future"""

    logger.info("💾 Node: Cache Response")

    if settings.enable_semantic_cache:
        key = _hash(state["text"])
        _cache[key] = {
            "response": state.get("response", ""),
            "intent": state.get("intent"),
            "confidence": state.get("confidence")
        }
        logger.info("✅ Response cached")

    # Always return metadata to register a change
    return {"metadata": {**state.get("metadata", {}), "cached": settings.enable_semantic_cache}}
