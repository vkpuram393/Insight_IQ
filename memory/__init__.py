"""
Memory Store Facade - Abstract interface for caching and session memory

This facade allows switching between implementations:
- Development: In-Memory
- Production: Redis/GCP Memorystore

All cache operations go through this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from config.config import settings


class MemoryStore(ABC):
    """Abstract base class for memory/cache implementations"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache by key"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL (time to live)"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    async def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        pass

    @abstractmethod
    async def append_to_session(
        self,
        session_id: str,
        role: str,
        content: str,
        max_messages: int = 10
    ) -> bool:
        """Append message to session history, keeping only recent N messages"""
        pass

    @abstractmethod
    async def get_session_facts(self, session_id: str) -> List[Dict[str, Any]]:
        """Get long-term facts for a session"""
        pass

    @abstractmethod
    async def add_session_fact(
        self,
        session_id: str,
        fact_type: str,
        data: Dict[str, Any]
    ) -> bool:
        """Add a fact to long-term memory"""
        pass

    @abstractmethod
    async def clear_session(self, session_id: str) -> bool:
        """Clear all data for a session"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup resources"""
        pass


class MemoryStoreFactory:
    """Factory to create appropriate memory store based on configuration"""

    _instance: Optional[MemoryStore] = None

    @classmethod
    def get_instance(cls, store_type: str = "inmemory") -> MemoryStore:
        """
        Get singleton instance of memory store

        Args:
            store_type: "inmemory", "redis", or "memorystore"
        """
        if cls._instance is None:
            if store_type == "inmemory":
                from memory.inmemory_store import InMemoryStore
                cls._instance = InMemoryStore()
            elif store_type == "redis":
                from memory.redis_store import RedisStore
                cls._instance = RedisStore(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    password=settings.redis_password,
                    username=settings.redis_username or "",
                    db=settings.redis_db,
                    ssl=settings.redis_ssl
                )
            elif store_type == "memorystore":
                # TODO: Implement when GCP Memorystore is available
                from memory.gcp_memorystore import GCPMemoryStore
                cls._instance = GCPMemoryStore()
            else:
                raise ValueError(f"Unknown memory store type: {store_type}")

        return cls._instance

    @classmethod
    async def close_instance(cls) -> None:
        """Close the current instance"""
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None

