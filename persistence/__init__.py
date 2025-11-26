"""
Persistence Store Facade - Abstract interface for telemetry and analytics

This facade allows switching between implementations:
- Development: SQLite
- Production: Firestore/BigQuery

All telemetry and persistence operations go through this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class EventType(Enum):
    """Types of events to track"""
    REQUEST_RECEIVED = "request_received"
    INTENT_CLASSIFIED = "intent_classified"
    RESPONSE_GENERATED = "response_generated"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    TOOL_CALLED = "tool_called"
    ERROR_OCCURRED = "error_occurred"
    SAFETY_BLOCKED = "safety_blocked"
    CLARIFICATION_NEEDED = "clarification_needed"
    STREAM_EVENT = "stream_event"  # For tracking streaming events


class PersistenceStore(ABC):
    """Abstract base class for persistence/telemetry implementations"""

    @abstractmethod
    async def log_event(
        self,
        event_type: EventType,
        session_id: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """
        Log an event for telemetry/analytics

        Returns:
            event_id: Unique identifier for the logged event
        """
        pass

    @abstractmethod
    async def log_request(
        self,
        session_id: str,
        user_text: str,
        intent: Optional[str],
        confidence: Optional[float],
        response: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Log a complete request-response cycle

        Returns:
            request_id: Unique identifier for the request
        """
        pass

    @abstractmethod
    async def get_session_events(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all events for a session"""
        pass

    @abstractmethod
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        pass

    @abstractmethod
    async def get_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get analytics data

        Returns summary statistics like:
        - Total requests
        - Intent distribution
        - Average confidence
        - Cache hit rate
        - Error rate
        """
        pass

    @abstractmethod
    async def search_logs(
        self,
        query: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search through logs (useful for debugging)"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup resources"""
        pass


class PersistenceStoreFactory:
    """Factory to create appropriate persistence store based on configuration"""

    _instance: Optional[PersistenceStore] = None

    @classmethod
    def get_instance(cls, store_type: str = "sqlite") -> PersistenceStore:
        """
        Get singleton instance of persistence store

        Args:
            store_type: "sqlite", "firestore", or "bigquery"
        """
        if cls._instance is None:
            if store_type == "sqlite":
                from persistence.sqlite_store import SQLitePersistenceStore
                cls._instance = SQLitePersistenceStore()
            elif store_type == "firestore":
                # TODO: Implement when Firestore is available
                from persistence.firestore_store import FirestorePersistenceStore
                cls._instance = FirestorePersistenceStore()
            elif store_type == "bigquery":
                # TODO: Implement when BigQuery is available
                from persistence.bigquery_store import BigQueryPersistenceStore
                cls._instance = BigQueryPersistenceStore()
            else:
                raise ValueError(f"Unknown persistence store type: {store_type}")

        return cls._instance

    @classmethod
    async def close_instance(cls) -> None:
        """Close the current instance"""
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None

