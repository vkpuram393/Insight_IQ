"""
SQLite Persistence Store Implementation

Temporary implementation using SQLite database.
Perfect for development, testing, and lightweight production.

Database schema:
- events: Individual telemetry events
- requests: Complete request-response cycles
- analytics: Pre-computed analytics data

⚠️ Note: SQLite is single-file, portable, and fast for small-medium workloads.
For large-scale production, replace with Firestore or BigQuery.
"""

import aiosqlite
import json
import uuid
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
from persistence import PersistenceStore, EventType
from core.logger import get_logger

if TYPE_CHECKING:
    from core.node_models import ResponseFeedbackSchema

logger = get_logger(__name__)


class SQLitePersistenceStore(PersistenceStore):
    """SQLite implementation for telemetry and analytics"""

    def __init__(self, db_path: str = None):
        # Use settings.telemetry_db_path if not provided (allows test override)
        if db_path is None:
            from config.config import settings
            db_path = settings.telemetry_db_path
        self.db_path = db_path
        self.db = None
        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"📊 SQLitePersistenceStore initialized: {db_path}")

    async def _get_connection(self):
        """Get or create database connection"""
        if self.db is None:
            self.db = await aiosqlite.connect(self.db_path, timeout=10.0)  # 10 second timeout
            self.db.row_factory = aiosqlite.Row
            await self.db.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
            await self._init_schema()
            await self.db.commit()  # Ensure schema is committed
        return self.db

    async def _init_schema(self):
        """Initialize database schema"""
        # Use self.db directly to avoid circular dependency
        db = self.db
        if db is None:
            raise RuntimeError("Database connection not established before schema initialization")

        # Events table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_id TEXT,
                data TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Requests table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT,
                user_text TEXT NOT NULL,
                intent TEXT,
                confidence REAL,
                response TEXT NOT NULL,
                metadata TEXT,
                duration_ms INTEGER,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Logs table for audit logging
        await db.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                log_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                request_id TEXT,
                node_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Exceptions table for error logging
        await db.execute("""
            CREATE TABLE IF NOT EXISTS exceptions (
                exception_id TEXT PRIMARY KEY,
                error_code TEXT NOT NULL,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                user_message TEXT NOT NULL,
                session_id TEXT,
                request_id TEXT,
                node_name TEXT,
                stacktrace TEXT,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # ========================================================================
        # CONVERSATION HISTORY TABLE (stores unmasked data)
        # ========================================================================
        
        # Conversation History table: stores complete conversation history with UNMASKED data
        # Data is saved AFTER postcheck node has already unmasked PII/PHI
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                
                -- Unmasked content (actual user message and agent response)
                user_message TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                
                -- Metadata
                intent TEXT,
                tools_used TEXT,
                metadata_json TEXT,
                
                -- Metrics
                duration_ms REAL,
                
                created_at TEXT NOT NULL
            )
        """)

        # Create indexes for performance
        await db.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_requests_session ON requests(session_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_requests_user ON requests(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_logs_session ON logs(session_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_logs_request ON logs(request_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_logs_node ON logs(node_name)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_exceptions_session ON exceptions(session_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_exceptions_request ON exceptions(request_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_exceptions_node ON exceptions(node_name)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_exceptions_timestamp ON exceptions(timestamp)")
        
        # Indexes for conversation_history table
        await db.execute("CREATE INDEX IF NOT EXISTS idx_conversation_history_session ON conversation_history(session_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_conversation_history_user ON conversation_history(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_conversation_history_timestamp ON conversation_history(timestamp)")

        await db.commit()
        logger.info("✅ Database schema initialized (with conversation_history table)")

    async def log_event(
        self,
        event_type: EventType,
        session_id: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """Log an event"""
        db = await self._get_connection()

        event_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        await db.execute(
            """
            INSERT INTO events (event_id, event_type, session_id, user_id, data, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type.value,
                session_id,
                user_id,
                json.dumps(data),
                now,
                now
            )
        )
        await db.commit()

        logger.debug(f"📝 Event logged: {event_type.value} for session {session_id}")
        return event_id

    async def log_request(
        self,
        session_id: str,
        user_text: str,
        intent: Optional[str],
        confidence: Optional[float],
        response: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Log a complete request-response cycle"""
        db = await self._get_connection()

        request_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        user_id = metadata.get("user_id")
        duration_ms = metadata.get("duration_ms")

        await db.execute(
            """
            INSERT INTO requests 
            (request_id, session_id, user_id, user_text, intent, confidence, 
             response, metadata, duration_ms, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                session_id,
                user_id,
                user_text,
                intent,
                confidence,
                response,
                json.dumps(metadata),
                duration_ms,
                now,
                now
            )
        )
        await db.commit()

        logger.debug(f"📝 Request logged: {request_id}")
        return request_id

    async def get_session_events(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all events for a session"""
        db = await self._get_connection()

        async with db.execute(
            """
            SELECT * FROM events 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
            """,
            (session_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "session_id": row["session_id"],
                    "user_id": row["user_id"],
                    "data": json.loads(row["data"]),
                    "timestamp": row["timestamp"]
                }
                for row in rows
            ]

    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        db = await self._get_connection()

        async with db.execute(
            """
            SELECT DISTINCT session_id, MIN(timestamp) as first_request, 
                   MAX(timestamp) as last_request, COUNT(*) as request_count
            FROM requests 
            WHERE user_id = ? 
            GROUP BY session_id 
            ORDER BY last_request DESC 
            LIMIT ?
            """,
            (user_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "session_id": row["session_id"],
                    "first_request": row["first_request"],
                    "last_request": row["last_request"],
                    "request_count": row["request_count"]
                }
                for row in rows
            ]

    async def get_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get analytics data"""
        db = await self._get_connection()

        # Build date filter
        date_filter = ""
        params = []
        if start_date:
            date_filter += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            date_filter += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        # Total requests
        async with db.execute(
            f"SELECT COUNT(*) as total FROM requests WHERE 1=1 {date_filter}",
            params
        ) as cursor:
            row = await cursor.fetchone()
            total_requests = row["total"]

        # Intent distribution
        async with db.execute(
            f"SELECT intent, COUNT(*) as count FROM requests WHERE 1=1 {date_filter} GROUP BY intent",
            params
        ) as cursor:
            rows = await cursor.fetchall()
            intent_distribution = {row["intent"]: row["count"] for row in rows}

        # Average confidence
        async with db.execute(
            f"SELECT AVG(confidence) as avg_conf FROM requests WHERE confidence IS NOT NULL {date_filter}",
            params
        ) as cursor:
            row = await cursor.fetchone()
            avg_confidence = row["avg_conf"] or 0.0

        # Cache stats from events
        async with db.execute(
            f"""
            SELECT event_type, COUNT(*) as count 
            FROM events 
            WHERE event_type IN ('cache_hit', 'cache_miss') {date_filter}
            GROUP BY event_type
            """,
            params
        ) as cursor:
            rows = await cursor.fetchall()
            cache_stats = {row["event_type"]: row["count"] for row in rows}

        cache_hits = cache_stats.get("cache_hit", 0)
        cache_misses = cache_stats.get("cache_miss", 0)
        cache_total = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / cache_total * 100) if cache_total > 0 else 0.0

        # Error rate
        async with db.execute(
            f"SELECT COUNT(*) as errors FROM events WHERE event_type = 'error_occurred' {date_filter}",
            params
        ) as cursor:
            row = await cursor.fetchone()
            error_count = row["errors"]

        error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0.0

        # Average duration
        async with db.execute(
            f"SELECT AVG(duration_ms) as avg_duration FROM requests WHERE duration_ms IS NOT NULL {date_filter}",
            params
        ) as cursor:
            row = await cursor.fetchone()
            avg_duration_ms = row["avg_duration"] or 0.0

        return {
            "total_requests": total_requests,
            "intent_distribution": intent_distribution,
            "avg_confidence": round(avg_confidence, 3),
            "cache_hit_rate": round(cache_hit_rate, 2),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "error_count": error_count,
            "error_rate": round(error_rate, 2),
            "avg_duration_ms": round(avg_duration_ms, 2),
            "date_range": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            }
        }

    async def search_logs(
        self,
        query: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search through logs"""
        db = await self._get_connection()

        search_pattern = f"%{query}%"

        async with db.execute(
            """
            SELECT * FROM requests 
            WHERE user_text LIKE ? OR response LIKE ? OR intent LIKE ?
            ORDER BY timestamp DESC 
            LIMIT ?
            """,
            (search_pattern, search_pattern, search_pattern, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "request_id": row["request_id"],
                    "session_id": row["session_id"],
                    "user_text": row["user_text"],
                    "intent": row["intent"],
                    "confidence": row["confidence"],
                    "response": row["response"],
                    "timestamp": row["timestamp"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                }
                for row in rows
            ]

    async def log_audit(
        self,
        session_id: str,
        node_name: str,
        event_type: str,
        data: Dict[str, Any],
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """Log an audit event to the logs table"""
        db = await self._get_connection()

        log_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        await db.execute(
            """
            INSERT INTO logs (log_id, session_id, request_id, node_name, event_type, data, timestamp, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_id,
                session_id,
                request_id,
                node_name,
                event_type,
                json.dumps(data),
                now,
                user_id,
                now
            )
        )
        await db.commit()

        logger.debug(f"📝 Audit log: {event_type} from {node_name} for session {session_id}")
        return log_id

    async def log_exception(
        self,
        error_code: str,
        category: str,
        severity: str,
        message: str,
        user_message: str,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        node_name: Optional[str] = None,
        stacktrace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> str:
        """Log an exception to the exceptions table"""
        db = await self._get_connection()

        exception_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        await db.execute(
            """
            INSERT INTO exceptions 
            (exception_id, error_code, category, severity, message, user_message,
             session_id, request_id, node_name, stacktrace, metadata, timestamp, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exception_id,
                error_code,
                category,
                severity,
                message,
                user_message,
                session_id,
                request_id,
                node_name,
                stacktrace,
                json.dumps(metadata) if metadata else None,
                now,
                user_id,
                now
            )
        )
        await db.commit()

        logger.error(f"🚨 Exception logged: {error_code} in {node_name} for session {session_id}")
        return exception_id

    # ========================================================================
    # CONVERSATION HISTORY PERSISTENCE METHODS
    # ========================================================================

    async def save_conversation(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        agent_response: str,
        intent: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None
    ) -> int:
        """
        Save a conversation turn to database with UNMASKED data.
        
        This should be called AFTER postcheck node has already unmasked PII/PHI.
        We store the actual user message and agent response as-is.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            user_message: User message (unmasked, actual text)
            agent_response: Agent response (unmasked, actual text)
            intent: Detected intent
            tools_used: List of tools called
            metadata: Additional metadata
            duration_ms: Operation duration in milliseconds
            
        Returns:
            conversation_id: ID of inserted conversation record
        """
        db = await self._get_connection()
        now = datetime.now().isoformat()

        # Insert conversation with unmasked data
        cursor = await db.execute(
            """
            INSERT INTO conversation_history 
            (session_id, user_id, timestamp, user_message, 
             agent_response, intent, tools_used, 
             metadata_json, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_id,
                now,
                user_message,
                agent_response,
                intent,
                json.dumps(tools_used) if tools_used else None,
                json.dumps(metadata) if metadata else None,
                duration_ms,
                now
            )
        )
        conversation_id = cursor.lastrowid
        await db.commit()
        
        logger.debug(f"💾 Conversation saved (unmasked): session={session_id}, id={conversation_id}")
        return conversation_id

    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of conversation turns to retrieve
            
        Returns:
            List of conversation turns with unmasked messages
        """
        db = await self._get_connection()

        async with db.execute(
            """
            SELECT * FROM conversation_history 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
            """,
            (session_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            
            conversations = []
            for row in rows:
                conv = {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "user_id": row["user_id"],
                    "timestamp": row["timestamp"],
                    "user_message": row["user_message"],
                    "agent_response": row["agent_response"],
                    "intent": row["intent"],
                    "tools_used": json.loads(row["tools_used"]) if row["tools_used"] else [],
                    "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                    "duration_ms": row["duration_ms"]
                }
                conversations.append(conv)
            
            # Return in chronological order (oldest first)
            return list(reversed(conversations))

    async def get_session_stats(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session statistics
        """
        db = await self._get_connection()

        async with db.execute(
            """
            SELECT 
                COUNT(*) as total_messages,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message,
                AVG(duration_ms) as avg_duration_ms
            FROM conversation_history 
            WHERE session_id = ?
            """,
            (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return {
                "session_id": session_id,
                "total_messages": row["total_messages"],
                "first_message": row["first_message"],
                "last_message": row["last_message"],
                "avg_duration_ms": round(row["avg_duration_ms"], 2) if row["avg_duration_ms"] else 0.0
            }

    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all conversations for a user across all sessions
        
        Args:
            user_id: User identifier
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversations (unmasked)
        """
        db = await self._get_connection()

        async with db.execute(
            """
            SELECT * FROM conversation_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
            """,
            (user_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "user_id": row["user_id"],
                    "timestamp": row["timestamp"],
                    "user_message": row["user_message"],
                    "agent_response": row["agent_response"],
                    "intent": row["intent"],
                    "tools_used": json.loads(row["tools_used"]) if row["tools_used"] else []
                }
                for row in rows
            ]

    async def delete_session_conversations(
        self,
        session_id: str
    ) -> int:
        """
        Delete all conversations for a session (for privacy/GDPR)
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of conversations deleted
        """
        db = await self._get_connection()

        cursor = await db.execute(
            "DELETE FROM conversation_history WHERE session_id = ?",
            (session_id,)
        )
        await db.commit()
        
        deleted_count = cursor.rowcount
        logger.info(f"🗑️  Deleted {deleted_count} conversations for session {session_id}")
        return deleted_count

    # ============================================================================
    # RESPONSE FEEDBACK METHODS
    # ============================================================================

    async def save_response_feedback(self, feedback: "ResponseFeedbackSchema") -> Dict[str, Any]:
        """
        Save or update response feedback in SQLite
        
        Args:
            feedback: ResponseFeedbackSchema object containing feedback data
            
        Returns:
            Dict with operation status and details
        """
        try:
            db = await self._get_connection()
            
            # Check if feedback table exists, create if not
            await db.execute("""
                CREATE TABLE IF NOT EXISTS response_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    response_id TEXT UNIQUE NOT NULL,
                    response_feedback INTEGER NOT NULL,
                    feedback_type TEXT NOT NULL,
                    response_createddatetime TEXT NOT NULL,
                    session_id TEXT,
                    query_text TEXT,
                    response_text TEXT,
                    user_comment TEXT
                )
            """)
            
            # Check if feedback already exists
            cursor = await db.execute(
                "SELECT id FROM response_feedback WHERE response_id = ?",
                (feedback.response_id,)
            )
            existing = await cursor.fetchone()
            
            feedback_bool = 1 if feedback.response_feedback else 0
            
            if existing:
                # Update existing feedback
                user_comment = feedback.user_comment or ""
                
                await db.execute("""
                    UPDATE response_feedback 
                    SET response_feedback = ?,
                        feedback_type = ?,
                        response_createddatetime = ?,
                        session_id = ?,
                        query_text = ?,
                        response_text = ?,
                        user_comment = ?
                    WHERE response_id = ?
                """, (
                    feedback_bool,
                    feedback.feedback_type.value,
                    feedback.response_createddatetime.isoformat(),
                    feedback.session_id,
                    feedback.query_text,
                    feedback.response_text,
                    user_comment,
                    feedback.response_id
                ))
                await db.commit()
                logger.info(f"👍👎 Feedback updated for response_id: {feedback.response_id}")
                return {
                    "status": "updated",
                    "matched_count": 1,
                    "modified_count": 1,
                    "response_id": feedback.response_id
                }
            else:
                # Insert new feedback
                user_comment = feedback.user_comment or ""
                
                cursor = await db.execute("""
                    INSERT INTO response_feedback (
                        response_id, response_feedback, feedback_type,
                        response_createddatetime, session_id, query_text,
                        response_text, user_comment
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.response_id,
                    feedback_bool,
                    feedback.feedback_type.value,
                    feedback.response_createddatetime.isoformat(),
                    feedback.session_id,
                    feedback.query_text,
                    feedback.response_text,
                    user_comment
                ))
                await db.commit()
                logger.info(f"👍👎 Feedback created for response_id: {feedback.response_id}")
                return {
                    "status": "created",
                    "inserted_id": str(cursor.lastrowid),
                    "response_id": feedback.response_id
                }
                
        except Exception as e:
            logger.error(f"❌ Error saving response feedback: {str(e)}")
            raise
    
    async def get_all_feedback_with_filter(
        self, 
        query_filter: Dict[str, Any], 
        limit: int = 100, 
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all feedback with optional filtering and pagination.
        
        Args:
            query_filter: Dict with filters (session_id, user_id, feedback_type)
            limit: Maximum number of results
            skip: Number of results to skip
            
        Returns:
            List of feedback records
        """
        try:
            db = await self._get_connection()
            
            # Build WHERE clause from query_filter
            where_conditions = []
            params = []
            
            if "session_id" in query_filter:
                where_conditions.append("session_id = ?")
                params.append(query_filter["session_id"])
            
            if "user_id" in query_filter:
                # SQLite doesn't have user_id column by default
                logger.warning("⚠️ user_id filter not supported in SQLite (column doesn't exist)")
            
            if "feedback_type" in query_filter:
                where_conditions.append("feedback_type = ?")
                params.append(query_filter["feedback_type"])
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Add pagination
            params.extend([limit, skip])
            
            query = f"""
                SELECT * FROM response_feedback 
                WHERE {where_clause}
                ORDER BY response_createddatetime DESC
                LIMIT ? OFFSET ?
            """
            
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            
            feedbacks = []
            for row in rows:
                feedbacks.append({
                    "_id": str(row["id"]),
                    "response_id": row["response_id"],
                    "response_feedback": bool(row["response_feedback"]),
                    "feedback_type": row["feedback_type"],
                    "response_createddatetime": row["response_createddatetime"],
                    "session_id": row["session_id"],
                    "query_text": row["query_text"],
                    "response_text": row["response_text"],
                    "user_comment": row["user_comment"] if "user_comment" in row.keys() else ""
                })
            
            logger.info(f"📊 Retrieved {len(feedbacks)} feedback entries with filters: {query_filter}")
            return feedbacks
            
        except Exception as e:
            logger.error(f"❌ Error retrieving filtered feedback: {e}")
            return []

    async def close(self) -> None:
        """Close database connection"""
        if self.db:
            await self.db.close()
            self.db = None
            logger.info("📊 SQLitePersistenceStore closed")

