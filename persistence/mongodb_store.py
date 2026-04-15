"""
MongoDB Persistence Store Implementation

Production-ready implementation using MongoDB.
Replaces SQLite for scalable, distributed storage.

Collections:
- events: Individual telemetry events
- requests: Complete request-response cycles
- logs: Audit logs
- exceptions: Error/exception logs
- conversation_history: Conversation history with unmasked data
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime
import json
import uuid
from urllib.parse import quote_plus, urlparse, urlunparse
from persistence import PersistenceStore, EventType
from core.logger import get_logger

if TYPE_CHECKING:
    from core.node_models import ResponseFeedbackSchema

logger = get_logger(__name__)


class MongoDBPersistenceStore(PersistenceStore):
    """MongoDB implementation for telemetry and analytics"""

    def __init__(self, connection_string: str = None, database_name: str = None):
        """
        Initialize MongoDB persistence store.
        
        Connection string is constructed from environment variables:
        - MONGODB_USER, MONGODB_PASSWORD (from Vault)
        - MONGODB_HOST (from deployment config)
        """
        import os
        from config.config import settings

        # Build connection string from environment variables (Vault + deployment config)
        if connection_string is None:
            mongodb_user = os.getenv('MONGODB_USER', '')
            mongodb_password = os.getenv('MONGODB_PASSWORD', '')
            mongodb_host = os.getenv('MONGODB_HOST', '')
            
            if mongodb_user and mongodb_password and mongodb_host:
                # URL-encode password in case of special characters
                encoded_password = quote_plus(mongodb_password)
                connection_string = f"mongodb+srv://{mongodb_user}:{encoded_password}@{mongodb_host}/?retryWrites=true&w=majority"
                logger.info("🔐 MongoDB connection string constructed from Vault credentials")
            else:
                # Fallback to legacy config field (for local development)
                connection_string = getattr(settings, 'mongodb_connection_string', None)
                if not connection_string:
                    raise ValueError(
                        "❌ MongoDB connection not configured. Set either:\n"
                        "   1. MONGODB_USER + MONGODB_PASSWORD (Vault) + MONGODB_HOST (deployment)\n"
                        "   2. MONGODB_CONNECTION_STRING (legacy/local dev)"
                    )
        
        if database_name is None:
            database_name = os.getenv('MONGODB_DATABASE_NAME', '') or \
                          getattr(settings, 'mongodb_database_name', 'myclaims-DEV')

        # Validate and normalize connection string format
        if connection_string and not connection_string.startswith(('mongodb://', 'mongodb+srv://')):
            logger.warning(
                f"⚠️  Connection string does not start with 'mongodb://' or 'mongodb+srv://'. "
                f"Current format: {connection_string[:50]}..."
            )
            # Try to fix common issues
            if connection_string.startswith('mongodb+srv:'):
                # Missing // after mongodb+srv:
                connection_string = connection_string.replace('mongodb+srv:', 'mongodb+srv://', 1)
                logger.info(f"🔧 Fixed connection string format (added missing //)")
            elif connection_string.startswith('mongodb:'):
                # Missing // after mongodb:
                connection_string = connection_string.replace('mongodb:', 'mongodb://', 1)
                logger.info(f"🔧 Fixed connection string format (added missing //)")

        # Normalize connection string: ensure credentials are URL-encoded
        connection_string = self._normalize_connection_string(connection_string)

        self.connection_string = connection_string
        self.database_name = database_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None

        logger.info(f"📊 MongoDBPersistenceStore initialized: {database_name}")

    async def _get_connection(self):
        """Get or create MongoDB connection"""
        if self.client is None or self.db is None:
            try:
                # Uncomment below if SSL certificate issues on macOS:
                # import certifi
                # tls_ca_file = certifi.where()
                
                # Create client with connection timeout and server selection timeout
                self.client = AsyncIOMotorClient(
                    self.connection_string,
                    serverSelectionTimeoutMS=5000,  # 5 second timeout for server selection
                    connectTimeoutMS=10000,  # 10 second timeout for connection
                    retryWrites=True
                    # tlsCAFile=tls_ca_file  # Uncomment if using certifi above
                )
                # Test the connection
                await self.client.admin.command('ping')
                self.db = self.client[self.database_name]
                await self._init_indexes()
                logger.info(f"✅ MongoDB connection established to database: {self.database_name}")
            except Exception as e:
                error_str = str(e)
                logger.error(f"❌ Failed to connect to MongoDB: {error_str}")
                logger.error(f"   Connection string: {self._mask_connection_string()}")
                logger.error(f"   Database name: {self.database_name}")
                
                # Provide specific troubleshooting for authentication errors
                if "Authentication failed" in error_str or "authentication" in error_str.lower():
                    logger.error("   🔍 Authentication Troubleshooting:")
                    logger.error("      1. Verify username and password are correct")
                    logger.error("      2. Check if username is case-sensitive (e.g., 'myClaims_dev' vs 'myclaims_dev')")
                    logger.error("      3. Ensure the user exists in MongoDB Atlas")
                    logger.error("      4. Verify the user has access to the database: " + self.database_name)
                    logger.error("      5. Check if password contains special characters that need URL encoding")
                    logger.error("      6. For MongoDB Atlas, verify IP whitelist allows your connection")
                    logger.error("      7. Try adding authSource parameter: ?authSource=admin")
                
                # Log connection string format for debugging (check for missing //)
                conn_str_preview = self.connection_string[:50] + "..." if len(self.connection_string) > 50 else self.connection_string
                logger.error(f"   Connection string preview: {conn_str_preview}")
                
                # Check if connection string is malformed
                if not self.connection_string.startswith(('mongodb://', 'mongodb+srv://')):
                    logger.error(f"   ⚠️  WARNING: Connection string does not start with 'mongodb://' or 'mongodb+srv://'")
                    logger.error(f"   ⚠️  This may indicate a malformed connection string in environment variable or config")
                
                # Reset connection state on failure
                self.client = None
                self.db = None
                raise
        if self.db is None:
            raise RuntimeError("MongoDB database connection is None - connection failed")
        return self.db

    def _normalize_connection_string(self, conn_str: str) -> str:
        """
        Normalize MongoDB connection string by URL-encoding credentials.
        This ensures passwords with special characters are properly encoded.
        """
        try:
            # Parse the connection string
            parsed = urlparse(conn_str)
            
            # If there are credentials in the netloc, encode them
            if '@' in parsed.netloc:
                # Split netloc into credentials and host
                auth_part, host_part = parsed.netloc.rsplit('@', 1)
                
                # Split credentials into username and password
                if ':' in auth_part:
                    username, password = auth_part.split(':', 1)
                    # URL-encode username and password (in case of special characters)
                    encoded_username = quote_plus(username)
                    encoded_password = quote_plus(password)
                    # Reconstruct netloc with encoded credentials
                    new_netloc = f"{encoded_username}:{encoded_password}@{host_part}"
                    
                    # Reconstruct the full URL
                    new_parsed = parsed._replace(netloc=new_netloc)
                    normalized = urlunparse(new_parsed)
                    return normalized
            
            return conn_str
        except Exception as e:
            logger.warning(f"⚠️  Failed to normalize connection string: {str(e)}")
            return conn_str

    def _mask_connection_string(self) -> str:
        """Mask password in connection string for logging"""
        try:
            # Handle mongodb:// and mongodb+srv:// URLs
            if '@' in self.connection_string:
                # Split on @ to separate credentials from host
                parts = self.connection_string.split('@', 1)
                if len(parts) == 2:
                    user_pass_part = parts[0]
                    host_part = parts[1]
                    
                    # Find the last : before @ (this is the password separator)
                    # For mongodb+srv://user:pass@host, we need to find : after //
                    if '://' in user_pass_part:
                        # Extract protocol (mongodb+srv:// or mongodb://)
                        protocol_end = user_pass_part.find('://') + 3
                        protocol = user_pass_part[:protocol_end]
                        credentials = user_pass_part[protocol_end:]
                        
                        # Split credentials on : to get username and password
                        if ':' in credentials:
                            username, _ = credentials.split(':', 1)
                            return f"{protocol}{username}:****@{host_part}"
                        else:
                            # No password, just username
                            return f"{protocol}{credentials}@{host_part}"
                    else:
                        # No protocol, just user:pass@host format
                        if ':' in user_pass_part:
                            username, _ = user_pass_part.split(':', 1)
                            return f"{username}:****@{host_part}"
        except Exception:
            # If masking fails, return a safe version
            pass
        return self.connection_string

    async def _init_indexes(self):
        """Create indexes for performance (equivalent to SQLite indexes)"""
        db = await self._get_connection()

        try:
            # Events collection indexes
            await db.events.create_index("session_id")
            await db.events.create_index("event_type")
            await db.events.create_index("timestamp")

            # Requests collection indexes
            await db.requests.create_index("session_id")
            await db.requests.create_index("user_id")
            await db.requests.create_index("timestamp")

            # Logs collection indexes
            await db.logs.create_index("session_id")
            await db.logs.create_index("request_id")
            await db.logs.create_index("node_name")
            await db.logs.create_index("timestamp")

            # Exceptions collection indexes
            await db.exceptions.create_index("session_id")
            await db.exceptions.create_index("request_id")
            await db.exceptions.create_index("node_name")
            await db.exceptions.create_index("timestamp")

            # Conversation history indexes
            await db.conversation_history.create_index("session_id")
            await db.conversation_history.create_index("user_id")
            await db.conversation_history.create_index("timestamp")

            logger.info("✅ MongoDB indexes created")
        except Exception as e:
            # Index creation is idempotent, but log any errors
            logger.warning(f"⚠️  Some indexes may already exist: {str(e)}")

    async def log_event(
        self,
        event_type: EventType,
        session_id: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """Log an event"""
        try:
            db = await self._get_connection()
        except Exception as e:
            logger.warning(f"⚠️ Failed to get MongoDB connection for event log: {str(e)}")
            # Return a dummy ID to allow application to continue
            return str(uuid.uuid4())

        if db is None:
            logger.warning(f"⚠️ MongoDB database is None, skipping event log for {event_type.value}")
            return str(uuid.uuid4())

        event_id = str(uuid.uuid4())
        now = datetime.utcnow()

        document = {
            "_id": event_id,
            "event_type": event_type.value,
            "session_id": session_id,
            "user_id": user_id,
            "data": data,  # MongoDB stores dicts directly (no JSON serialization needed)
            "timestamp": now,
            "created_at": now
        }

        try:
            await db.events.insert_one(document)
            logger.debug(f"📝 Event logged: {event_type.value} for session {session_id}")
            return event_id
        except Exception as e:
            logger.warning(f"⚠️ Failed to insert event into MongoDB: {str(e)}")
            return event_id  # Return ID even if insert failed

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
        now = datetime.utcnow()

        user_id = metadata.get("user_id")
        duration_ms = metadata.get("duration_ms")

        document = {
            "_id": request_id,
            "session_id": session_id,
            "user_id": user_id,
            "user_text": user_text,
            "intent": intent,
            "confidence": confidence,
            "response": response,
            "metadata": metadata,  # Direct dict storage
            "duration_ms": duration_ms,
            "timestamp": now,
            "created_at": now
        }

        await db.requests.insert_one(document)
        logger.debug(f"📝 Request logged: {request_id}")
        return request_id

    async def get_session_events(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all events for a session"""
        db = await self._get_connection()

        cursor = db.events.find(
            {"session_id": session_id}
        ).sort("timestamp", -1).limit(limit)

        events = []
        async for doc in cursor:
            # Convert MongoDB document to dict (remove _id, convert ObjectId to str)
            event = {
                "event_id": str(doc["_id"]),
                "event_type": doc["event_type"],
                "session_id": doc["session_id"],
                "user_id": doc.get("user_id"),
                "data": doc["data"],  # Already a dict in MongoDB
                "timestamp": doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else doc["timestamp"],
                "created_at": doc["created_at"].isoformat() if isinstance(doc["created_at"], datetime) else doc["created_at"]
            }
            events.append(event)

        return events

    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        db = await self._get_connection()

        cursor = db.requests.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit)

        sessions = []
        async for doc in cursor:
            session = {
                "session_id": doc["session_id"],
                "request_id": str(doc["_id"]),
                "user_text": doc["user_text"],
                "intent": doc.get("intent"),
                "confidence": doc.get("confidence"),
                "timestamp": doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else doc["timestamp"]
            }
            sessions.append(session)

        return sessions

    async def get_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get analytics data using MongoDB aggregation"""
        db = await self._get_connection()

        # Build match filter
        match_filter = {}
        if start_date or end_date:
            match_filter["timestamp"] = {}
            if start_date:
                match_filter["timestamp"]["$gte"] = start_date
            if end_date:
                match_filter["timestamp"]["$lte"] = end_date

        # Total requests
        total_requests = await db.requests.count_documents(match_filter)

        # Intent distribution
        intent_pipeline = [
            {"$match": match_filter},
            {"$group": {"_id": "$intent", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        intent_distribution = {}
        async for doc in db.requests.aggregate(intent_pipeline):
            intent_distribution[doc["_id"]] = doc["count"]

        # Average confidence
        avg_confidence_pipeline = [
            {"$match": {**match_filter, "confidence": {"$ne": None}}},
            {"$group": {"_id": None, "avg_confidence": {"$avg": "$confidence"}}}
        ]
        avg_confidence = 0.0
        async for doc in db.requests.aggregate(avg_confidence_pipeline):
            avg_confidence = doc.get("avg_confidence", 0.0)

        # Cache stats from events
        cache_match = {**match_filter, "event_type": {"$in": ["cache_hit", "cache_miss"]}}
        cache_pipeline = [
            {"$match": cache_match},
            {"$group": {"_id": "$event_type", "count": {"$sum": 1}}}
        ]
        cache_stats = {}
        async for doc in db.events.aggregate(cache_pipeline):
            cache_stats[doc["_id"]] = doc["count"]

        cache_hits = cache_stats.get("cache_hit", 0)
        cache_misses = cache_stats.get("cache_miss", 0)
        cache_total = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / cache_total * 100) if cache_total > 0 else 0.0

        # Error rate
        error_match = {**match_filter, "event_type": "error_occurred"}
        total_errors = await db.events.count_documents(error_match)
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0

        # Average duration
        duration_pipeline = [
            {"$match": {**match_filter, "duration_ms": {"$ne": None}}},
            {"$group": {"_id": None, "avg_duration": {"$avg": "$duration_ms"}}}
        ]
        avg_duration_ms = 0.0
        async for doc in db.requests.aggregate(duration_pipeline):
            avg_duration_ms = doc.get("avg_duration", 0.0)

        return {
            "total_requests": total_requests,
            "intent_distribution": intent_distribution,
            "avg_confidence": round(avg_confidence, 3),
            "cache_hit_rate": round(cache_hit_rate, 2),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "error_count": total_errors,
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
        """Search through logs using MongoDB text search or regex"""
        db = await self._get_connection()

        # MongoDB text search (requires text index) or regex search
        search_filter = {
            "$or": [
                {"data": {"$regex": query, "$options": "i"}},
                {"message": {"$regex": query, "$options": "i"}},
                {"user_text": {"$regex": query, "$options": "i"}}
            ]
        }

        cursor = db.logs.find(search_filter).sort("timestamp", -1).limit(limit)

        results = []
        async for doc in cursor:
            result = {
                "log_id": str(doc["_id"]),
                "session_id": doc["session_id"],
                "node_name": doc["node_name"],
                "event_type": doc["event_type"],
                "data": doc["data"],
                "timestamp": doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else doc["timestamp"]
            }
            results.append(result)

        return results

    async def log_audit(
        self,
        session_id: str,
        node_name: str,
        event_type: str,
        data: Dict[str, Any],
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """Log an audit event"""
        try:
            db = await self._get_connection()
        except Exception as e:
            logger.warning(f"⚠️ Failed to get MongoDB connection for audit log: {str(e)}")
            # Return a dummy ID to allow application to continue
            return str(uuid.uuid4())

        if db is None:
            logger.warning(f"⚠️ MongoDB database is None, skipping audit log for {node_name}")
            return str(uuid.uuid4())

        log_id = str(uuid.uuid4())
        now = datetime.utcnow()

        document = {
            "_id": log_id,
            "session_id": session_id,
            "request_id": request_id,
            "node_name": node_name,
            "event_type": event_type,
            "data": data,  # Direct dict storage
            "timestamp": now,
            "user_id": user_id,
            "created_at": now
        }

        try:
            await db.logs.insert_one(document)
            logger.debug(f"📝 Audit log: {event_type} from {node_name} for session {session_id}")
            return log_id
        except Exception as e:
            if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
                error_msg = (
                    f"❌ MongoDB authorization error: User does not have write permission on database '{self.database_name}'. "
                    f"Please contact your MongoDB administrator to grant 'readWrite' role on this database."
                )
                logger.error(error_msg)
                raise PermissionError(error_msg) from e
            raise

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
        """Log an exception"""
        try:
            db = await self._get_connection()
        except Exception as e:
            logger.warning(f"⚠️ Failed to get MongoDB connection for exception log: {str(e)}")
            # Return a dummy ID to allow application to continue
            return str(uuid.uuid4())

        if db is None:
            logger.warning(f"⚠️ MongoDB database is None, skipping exception log for {node_name}")
            return str(uuid.uuid4())

        exception_id = str(uuid.uuid4())
        now = datetime.utcnow()

        document = {
            "_id": exception_id,
            "error_code": error_code,
            "category": category,
            "severity": severity,
            "message": message,
            "user_message": user_message,
            "session_id": session_id,
            "request_id": request_id,
            "node_name": node_name,
            "stacktrace": stacktrace,
            "metadata": metadata or {},
            "timestamp": now,
            "user_id": user_id,
            "created_at": now
        }

        try:
            await db.exceptions.insert_one(document)
            logger.error(f"🚨 Exception logged: {error_code} from {node_name}")
            return exception_id
        except Exception as e:
            if "not authorized" in str(e).lower() or "unauthorized" in str(e).lower():
                error_msg = (
                    f"❌ MongoDB authorization error: User does not have write permission on database '{self.database_name}'. "
                    f"Please contact your MongoDB administrator to grant 'readWrite' role on this database."
                )
                logger.error(error_msg)
                raise PermissionError(error_msg) from e
            raise

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
    ) -> str:
        """Save a conversation turn"""
        db = await self._get_connection()

        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow()

        document = {
            "_id": conversation_id,
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": now,
            "user_message": user_message,
            "agent_response": agent_response,
            "intent": intent,
            "tools_used": tools_used or [],
            "metadata_json": json.dumps(metadata) if metadata else None,
            "metadata": metadata,  # Also store as dict for easier querying
            "duration_ms": duration_ms,
            "created_at": now
        }

        await db.conversation_history.insert_one(document)
        logger.debug(f"💾 Conversation saved (unmasked): session={session_id}, id={conversation_id}")
        return conversation_id

    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session (default: 100 messages = 50 turns)"""
        db = await self._get_connection()

        cursor = db.conversation_history.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).limit(limit)  # Sort ascending for chronological order

        conversations = []
        async for doc in cursor:
            conv = {
                "id": str(doc["_id"]),
                "session_id": doc["session_id"],
                "user_id": doc["user_id"],
                "timestamp": doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else doc["timestamp"],
                "user_message": doc["user_message"],
                "agent_response": doc["agent_response"],
                "intent": doc.get("intent"),
                "tools_used": doc.get("tools_used", []),
                "metadata": doc.get("metadata", {}),
                "duration_ms": doc.get("duration_ms")
            }
            conversations.append(conv)

        return conversations

    async def get_session_stats(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Get statistics for a session using MongoDB aggregation"""
        db = await self._get_connection()

        match_filter = {"session_id": session_id}

        # Count requests
        total_requests = await db.requests.count_documents(match_filter)

        # Count events
        total_events = await db.events.count_documents(match_filter)

        # Count exceptions
        total_exceptions = await db.exceptions.count_documents(match_filter)

        # Get first and last request timestamps
        first_request = await db.requests.find_one(match_filter, sort=[("timestamp", 1)])
        last_request = await db.requests.find_one(match_filter, sort=[("timestamp", -1)])

        # Count conversation history messages
        total_messages = await db.conversation_history.count_documents(match_filter)

        # Average duration from conversation history
        duration_pipeline = [
            {"$match": match_filter},
            {"$group": {"_id": None, "avg_duration": {"$avg": "$duration_ms"}}}
        ]
        avg_duration_ms = 0.0
        async for doc in db.conversation_history.aggregate(duration_pipeline):
            avg_duration_ms = doc.get("avg_duration", 0.0)

        return {
            "session_id": session_id,
            "total_requests": total_requests,
            "total_events": total_events,
            "total_exceptions": total_exceptions,
            "total_messages": total_messages,
            "avg_duration_ms": round(avg_duration_ms, 2) if avg_duration_ms else 0.0,
            "first_request_at": first_request["timestamp"].isoformat() if first_request and first_request.get("timestamp") else None,
            "last_request_at": last_request["timestamp"].isoformat() if last_request and last_request.get("timestamp") else None
        }

    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all conversations for a user"""
        db = await self._get_connection()

        cursor = db.conversation_history.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit)

        conversations = []
        async for doc in cursor:
            conv = {
                "id": str(doc["_id"]),
                "session_id": doc["session_id"],
                "user_id": doc["user_id"],
                "timestamp": doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else doc["timestamp"],
                "user_message": doc["user_message"],
                "agent_response": doc["agent_response"],
                "intent": doc.get("intent"),
                "tools_used": doc.get("tools_used", [])
            }
            conversations.append(conv)

        return conversations

    async def delete_session_conversations(
        self,
        session_id: str
    ) -> bool:
        """Delete all conversations for a session"""
        db = await self._get_connection()

        result = await db.conversation_history.delete_many({"session_id": session_id})
        deleted_count = result.deleted_count
        logger.info(f"🗑️  Deleted {deleted_count} conversations for session {session_id}")
        return deleted_count > 0

    # ============================================================================
    # RESPONSE FEEDBACK METHODS
    # ============================================================================

    async def save_response_feedback(self, feedback: "ResponseFeedbackSchema") -> Dict[str, Any]:
        """
        Save or update response feedback in MongoDB
        
        Args:
            feedback: ResponseFeedbackSchema object containing feedback data
            
        Returns:
            Dict with operation status and details
        """
        try:
            db = await self._get_connection()
            collection = db["Response_Feedback"]
            
            # Check if feedback already exists for this response_id
            existing_feedback = await collection.find_one(
                {"response_id": feedback.response_id}
            )
            
            feedback_dict = feedback.dict()
            
            if existing_feedback:
                # Update existing feedback (user changed their mind)
                result = await collection.update_one(
                    {"response_id": feedback.response_id},
                    {"$set": feedback_dict}
                )
                logger.info(f"👍👎 Feedback updated for response_id: {feedback.response_id}")
                return {
                    "status": "updated",
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count,
                    "response_id": feedback.response_id
                }
            else:
                # Insert new feedback
                result = await collection.insert_one(feedback_dict)
                logger.info(f"👍👎 Feedback created for response_id: {feedback.response_id}")
                return {
                    "status": "created",
                    "inserted_id": str(result.inserted_id),
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
            query_filter: MongoDB query filter
            limit: Maximum number of results
            skip: Number of results to skip
            
        Returns:
            List of feedback records
        """
        try:
            db = await self._get_connection()
            collection = db["Response_Feedback"]
            
            cursor = collection.find(query_filter)\
                .sort("response_createddatetime", -1)\
                .skip(skip)\
                .limit(limit)
            
            feedbacks = await cursor.to_list(length=limit)
            
            # Convert ObjectId to string and format data
            for feedback in feedbacks:
                if "_id" in feedback:
                    feedback["_id"] = str(feedback["_id"])
                # Convert datetime to ISO format
                if "response_createddatetime" in feedback:
                    if hasattr(feedback["response_createddatetime"], 'isoformat'):
                        feedback["response_createddatetime"] = feedback["response_createddatetime"].isoformat()
                # Convert enum to string
                if "feedback_type" in feedback:
                    feedback["feedback_type"] = str(feedback["feedback_type"]).replace("FeedbackType.", "")
            
            logger.info(f"📊 Retrieved {len(feedbacks)} feedback entries with filters: {query_filter}")
            return feedbacks
            
        except Exception as e:
            logger.error(f"❌ Error retrieving filtered feedback: {e}")
            return []

    # ============================================================================
    # LLM THINKING PROCESS LOGGING (Issue 2 - for debugging inconsistent responses)
    # ============================================================================

    async def log_thinking_process(
        self,
        session_id: str,
        request_id: str,
        user_query: str,
        intent: str,
        thinking_content: str,
        final_response: str,
        model: str,
        execution_time_ms: Optional[float] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log LLM thinking process for analysis.
        Fire-and-forget, non-blocking.
        
        This stores Gemini's chain-of-thought in the llm_thoughts collection
        for later analysis of why responses vary for the same question.
        
        Args:
            session_id: Session identifier
            request_id: Request identifier  
            user_query: Original user question
            intent: Detected intent
            thinking_content: Gemini's chain of thought
            final_response: Final response text
            model: LLM model used
            execution_time_ms: Optional execution time
            user_id: Optional user identifier
            metadata: Optional additional metadata
            
        Returns:
            str: Thought log ID
        """
        try:
            db = await self._get_connection()
        except Exception as e:
            logger.warning(f"⚠️ Failed to get MongoDB connection for thought log: {str(e)}")
            # Return a dummy ID to allow application to continue (fire-and-forget pattern)
            return str(uuid.uuid4())

        if db is None:
            logger.warning("⚠️ MongoDB database is None, skipping thought log")
            return str(uuid.uuid4())

        thought_id = str(uuid.uuid4())
        now = datetime.utcnow()

        document = {
            "_id": thought_id,
            "session_id": session_id,
            "request_id": request_id,
            "user_id": user_id,
            "user_query": user_query,
            "intent": intent,
            "thinking_content": thinking_content,
            "final_response": final_response,
            "model": model,
            "execution_time_ms": execution_time_ms,
            "metadata": metadata or {},
            "timestamp": now,
            "created_at": now
        }

        try:
            await db.llm_thoughts.insert_one(document)
            logger.debug(f"🧠 Thought logged: {thought_id}")
            return thought_id
        except Exception as e:
            logger.warning(f"⚠️ Failed to insert thought (non-fatal): {str(e)}")
            return thought_id  # Return ID even if insert failed

    async def close(self) -> None:
        """Close MongoDB connection"""
        if self.client:
            try:
                self.client.close()
                logger.info("📊 MongoDBPersistenceStore closed")
            except Exception as e:
                logger.warning(f"⚠️  Error closing MongoDB connection: {str(e)}")
            finally:
                self.client = None
                self.db = None