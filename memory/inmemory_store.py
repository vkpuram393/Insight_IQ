"""
In-Memory Store Implementation

Temporary implementation using Python dictionaries.
Perfect for development and testing.

⚠️ WARNING: Data is lost when application restarts!
This is replaced by Redis/Memorystore in production.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from memory import MemoryStore
from core.logger import get_logger

logger = get_logger(__name__)


class InMemoryStore(MemoryStore):
    """In-memory implementation using Python dicts"""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}  # key -> {value, expires_at}
        self._session_history: Dict[str, List[Dict[str, str]]] = {}  # session_id -> messages
        self._session_facts: Dict[str, List[Dict[str, Any]]] = {}  # session_id -> facts
        self._lock = asyncio.Lock()
        logger.info("🧠 InMemoryStore initialized (dev mode)")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.get("expires_at"):
                if datetime.now() > entry["expires_at"]:
                    del self._cache[key]
                    return None

            return entry["value"]

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache"""
        async with self._lock:
            expires_at = None
            if ttl_seconds:
                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

            self._cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": datetime.now()
            }
            return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        value = await self.get(key)
        return value is not None

    async def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        async with self._lock:
            return self._session_history.get(session_id, [])

    async def append_to_session(
        self,
        session_id: str,
        role: str,
        content: str,
        max_messages: int = 10
    ) -> bool:
        """Append message to session history"""
        async with self._lock:
            if session_id not in self._session_history:
                self._session_history[session_id] = []

            self._session_history[session_id].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })

            # Keep only recent messages
            self._session_history[session_id] = self._session_history[session_id][-max_messages:]

            return True

    async def get_session_facts(self, session_id: str) -> List[Dict[str, Any]]:
        """Get long-term facts for a session"""
        async with self._lock:
            return self._session_facts.get(session_id, [])

    async def add_session_fact(
        self,
        session_id: str,
        fact_type: str,
        data: Dict[str, Any]
    ) -> bool:
        """Add a fact to long-term memory"""
        async with self._lock:
            if session_id not in self._session_facts:
                self._session_facts[session_id] = []

            fact = {
                "type": fact_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            self._session_facts[session_id].append(fact)

            return True

    async def clear_session(self, session_id: str) -> bool:
        """Clear all data for a session"""
        async with self._lock:
            deleted = False
            if session_id in self._session_history:
                del self._session_history[session_id]
                deleted = True
            if session_id in self._session_facts:
                del self._session_facts[session_id]
                deleted = True
            return deleted

    async def close(self) -> None:
        """Cleanup resources"""
        logger.info("🧠 InMemoryStore closing...")
        async with self._lock:
            self._cache.clear()
            self._session_history.clear()
            self._session_facts.clear()
        logger.info("✅ InMemoryStore closed")

    # Additional utility methods for development

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        async with self._lock:
            return {
                "cache_keys": len(self._cache),
                "active_sessions": len(self._session_history),
                "total_facts": sum(len(facts) for facts in self._session_facts.values()),
                "total_messages": sum(len(msgs) for msgs in self._session_history.values())
            }

    async def clear_expired(self) -> int:
        """Manually clear expired cache entries"""
        async with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.get("expires_at") and now > entry["expires_at"]
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

