"""
Telemetry Helper - Easy logging of events and requests

This module provides convenience functions for logging telemetry data
using the persistence store facade.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
from config.config import settings
from core.logger import get_logger
from persistence import PersistenceStoreFactory, EventType

logger = get_logger(__name__)

# Get persistence store instance (facade pattern)
_persistence_store = None


def get_persistence_store():
    """Get or create persistence store instance"""
    global _persistence_store
    if _persistence_store is None:
        _persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
    return _persistence_store


async def log_event(
    event_type: EventType,
    session_id: str,
    data: Dict[str, Any],
    user_id: Optional[str] = None
) -> Optional[str]:
    """
    Log a telemetry event

    Args:
        event_type: Type of event (from EventType enum)
        session_id: Session identifier
        data: Event data/payload
        user_id: Optional user identifier

    Returns:
        event_id if successful, None if telemetry disabled
    """
    if not settings.enable_telemetry:
        return None

    try:
        store = get_persistence_store()
        event_id = await store.log_event(event_type, session_id, data, user_id)
        return event_id
    except Exception as e:
        logger.error(f"Failed to log event: {e}")
        return None


async def log_request_response(
    session_id: str,
    user_text: str,
    intent: Optional[str],
    confidence: Optional[float],
    response: str,
    metadata: Dict[str, Any]
) -> Optional[str]:
    """
    Log a complete request-response cycle

    Args:
        session_id: Session identifier
        user_text: User's input text
        intent: Classified intent
        confidence: Intent confidence score
        response: Generated response
        metadata: Additional metadata (timing, cache hits, etc.)

    Returns:
        request_id if successful, None if telemetry disabled
    """
    if not settings.enable_telemetry:
        return None

    try:
        store = get_persistence_store()
        request_id = await store.log_request(
            session_id=session_id,
            user_text=user_text,
            intent=intent,
            confidence=confidence,
            response=response,
            metadata=metadata
        )
        return request_id
    except Exception as e:
        logger.error(f"Failed to log request: {e}")
        return None


async def get_analytics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Get analytics data

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Dictionary with analytics data
    """
    try:
        store = get_persistence_store()
        return await store.get_analytics(start_date, end_date)
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        return {}


async def search_logs(query: str, limit: int = 100) -> list:
    """
    Search through logs

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        List of matching log entries
    """
    try:
        store = get_persistence_store()
        return await store.search_logs(query, limit)
    except Exception as e:
        logger.error(f"Failed to search logs: {e}")
        return []


class RequestTimer:
    """Context manager for timing request duration"""

    def __init__(self, session_id: str, event_type: EventType):
        self.session_id = session_id
        self.event_type = event_type
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration_ms = int((self.end_time - self.start_time) * 1000)

        # Log timing event asynchronously (fire and forget)
        import asyncio
        asyncio.create_task(
            log_event(
                self.event_type,
                self.session_id,
                {"duration_ms": duration_ms}
            )
        )

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration_ms = int((self.end_time - self.start_time) * 1000)

        await log_event(
            self.event_type,
            self.session_id,
            {"duration_ms": duration_ms}
        )

    @property
    def duration_ms(self) -> int:
        """Get duration in milliseconds"""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return 0


async def close_telemetry():
    """Close telemetry connections"""
    global _persistence_store
    if _persistence_store:
        await _persistence_store.close()
        _persistence_store = None

