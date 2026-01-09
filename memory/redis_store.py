"""
Redis Store Implementation

Connects to Redis using credentials from environment variables (Vault sidecar).
Never hardcodes credentials - all values loaded from config (which reads env vars).

Supports:
- Local Redis (development)
- Cloud Redis (staging)
- GCP Memorystore (production)
"""

import json
import asyncio
import time
from typing import Dict, Any, Optional, List
from memory import MemoryStore
from core.logger import get_logger

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
    # Check redis-py version for compatibility
    try:
        REDIS_VERSION = redis.__version__
    except AttributeError:
        try:
            import redis as redis_sync
            REDIS_VERSION = redis_sync.__version__
        except (AttributeError, ImportError):
            REDIS_VERSION = "unknown"
except ImportError:
    REDIS_AVAILABLE = False
    REDIS_VERSION = None

logger = get_logger(__name__)


class RedisStore(MemoryStore):
    """
    Redis implementation with environment-based configuration.
    
    Credentials are loaded from environment variables via config.Settings:
    - REDIS_HOST
    - REDIS_PORT  
    - REDIS_PASSWORD
    - REDIS_USERNAME (optional)
    - REDIS_DB
    - REDIS_SSL
    """

    def __init__(
        self,
        host: str,
        port: int,
        password: str,
        username: str = "",
        db: int = 0,
        ssl: bool = False,
        ssl_cert_reqs: str = "required"
    ):
        """
        Initialize Redis connection using environment variables.
        
        Args:
            host: Redis host (from REDIS_HOST env var)
            port: Redis port (from REDIS_PORT env var)
            password: Redis password (from REDIS_PASSWORD env var - injected by Vault)
            username: Redis username (from REDIS_USERNAME env var - optional)
            db: Redis database number (from REDIS_DB env var)
            ssl: Use SSL/TLS connection (from REDIS_SSL env var)
            ssl_cert_reqs: SSL certificate requirements
        """
        if not REDIS_AVAILABLE:
            logger.error("Redis package not installed. Run: pip install redis")
            raise ImportError("redis package required. Install with: pip install redis")
        
        if not host or not password:
            logger.error("Redis credentials missing. Required: REDIS_HOST, REDIS_PASSWORD")
            raise ValueError("Redis host and password must be provided via environment variables")
        
        # Validate host and port
        if not isinstance(host, str) or not host.strip():
            raise ValueError(f"Invalid Redis host: {repr(host)} (expected non-empty string)")
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValueError(f"Invalid Redis port: {repr(port)} (expected integer 1-65535)")
        
        self.host = host.strip()
        self.port = port
        self.password = password
        self.username = username
        self.db = db
        self.ssl = ssl
        self.ssl_cert_reqs = ssl_cert_reqs
        self.client: Optional[redis.Redis] = None
        
        # Connection state tracking for circuit breaker pattern
        self._connection_failed: bool = False
        self._last_connection_attempt: float = 0.0
        self._connection_retry_delay: float = 5.0  # Start with 5 seconds
        self._max_retry_delay: float = 60.0  # Max 60 seconds between retries
        self._circuit_breaker_cooldown: float = 30.0  # Don't retry for 30 seconds after failure
        self._last_error_log_time: float = 0.0
        self._error_log_interval: float = 60.0  # Only log connection errors once per minute
        
        logger.info(f"Initializing RedisStore: {host}:{port} (DB: {db}, SSL: {ssl})")
        logger.info("Using credentials from environment variables (Vault sidecar)")

    async def _ensure_connected(self) -> None:
        """
        Ensure Redis client is connected (lazy initialization).
        
        Implements circuit breaker pattern to avoid repeated connection attempts
        when Redis is unavailable. Only attempts reconnection after cooldown period.
        """
        current_time = time.time()
        
        # If we have a client, verify it's still connected
        if self.client is not None:
            try:
                await self.client.ping()
                # Connection is good, reset failure state
                if self._connection_failed:
                    self._connection_failed = False
                    self._connection_retry_delay = 5.0  # Reset retry delay
                    logger.info(f"Redis connection restored at {self.host}:{self.port}")
                return
            except Exception:
                # Connection lost, reset client
                self.client = None
                self._connection_failed = True
        
        # Circuit breaker: Don't attempt connection if we recently failed
        if self._connection_failed:
            time_since_last_attempt = current_time - self._last_connection_attempt
            if time_since_last_attempt < self._circuit_breaker_cooldown:
                # Only log error once per interval to reduce log noise
                if current_time - self._last_error_log_time >= self._error_log_interval:
                    logger.warning(
                        f"Redis unavailable (circuit breaker active). "
                        f"Last attempt: {time_since_last_attempt:.1f}s ago. "
                        f"Retrying in {self._circuit_breaker_cooldown - time_since_last_attempt:.1f}s. "
                        f"Operations will continue without Redis caching."
                    )
                    self._last_error_log_time = current_time
                raise ConnectionError(
                    f"Redis connection unavailable. Circuit breaker active. "
                    f"Retry in {self._circuit_breaker_cooldown - time_since_last_attempt:.1f}s"
                )
        
        # Attempt connection
        self._last_connection_attempt = current_time
        
        try:
            # Build connection kwargs
            # For GCP Memorystore, use longer timeouts (VPC connections may need more time)
            connection_kwargs = {
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "password": self.password,
                "decode_responses": True,  # Automatically decode bytes to strings
                "socket_connect_timeout": 5,  # 5 second connection timeout (increased for GCP Memorystore)
                "socket_timeout": 5,  # 5 second socket timeout
                "retry_on_timeout": True,  # Retry on timeout
                "health_check_interval": 30,  # Health check every 30 seconds
            }
            
            # Add username if using Redis ACL
            if self.username:
                connection_kwargs["username"] = self.username
            
            # Add SSL if enabled (required for GCP Memorystore with TLS enabled)
            if self.ssl:
                # Use exact SSL configuration that works with redis-py 5.0.4 and GCP Memorystore
                connection_kwargs["ssl"] = True
                connection_kwargs["ssl_check_hostname"] = False
                connection_kwargs["ssl_cert_reqs"] = None  # Explicitly set to None (not omitted)
            
            self.client = redis.Redis(**connection_kwargs)
            
            # Test connection with timeout (increased for GCP Memorystore)
            await asyncio.wait_for(self.client.ping(), timeout=5.0)
            
            # Connection successful
            if self._connection_failed:
                logger.info(f"Redis connection restored at {self.host}:{self.port}")
            else:
                logger.info(f"Successfully connected to Redis at {self.host}:{self.port}")
            
            self._connection_failed = False
            self._connection_retry_delay = 5.0  # Reset retry delay on success
            
        except asyncio.TimeoutError:
            self.client = None
            self._connection_failed = True
            # Exponential backoff for retry delay
            self._connection_retry_delay = min(
                self._connection_retry_delay * 1.5,
                self._max_retry_delay
            )
            if current_time - self._last_error_log_time >= self._error_log_interval:
                logger.error(
                    f"Redis connection timeout at {self.host}:{self.port}. "
                    f"Circuit breaker activated. Operations will continue without Redis caching."
                )
                self._last_error_log_time = current_time
            raise ConnectionError(
                f"Redis connection timeout. Circuit breaker active. "
                f"Retry in {self._circuit_breaker_cooldown}s"
            )
        except Exception as e:
            self.client = None
            self._connection_failed = True
            # Exponential backoff for retry delay
            self._connection_retry_delay = min(
                self._connection_retry_delay * 1.5,
                self._max_retry_delay
            )
            
            error_message = str(e)
            
            if current_time - self._last_error_log_time >= self._error_log_interval:
                logger.error(
                    f"Failed to connect to Redis at {self.host}:{self.port}: {error_message}. "
                    f"Circuit breaker activated. Operations will continue without Redis caching."
                )
                self._last_error_log_time = current_time
            
            raise ConnectionError(
                f"Redis connection failed: {error_message}. Circuit breaker active. "
                f"Retry in {self._circuit_breaker_cooldown}s"
            )

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis by key."""
        try:
            await self._ensure_connected()
            value = await self.client.get(key)
            
            if value is None:
                return None
            
            # Try to parse as JSON, fall back to string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Redis GET failed for key '{key}': {e}")
            return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in Redis with optional TTL."""
        try:
            await self._ensure_connected()
            
            # Serialize value as JSON if it's not a string
            if not isinstance(value, str):
                value = json.dumps(value)
            
            await self.client.set(key, value, ex=ttl_seconds)
            return True
            
        except Exception as e:
            logger.error(f"Redis SET failed for key '{key}': {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            await self._ensure_connected()
            result = await self.client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis DELETE failed for key '{key}': {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            await self._ensure_connected()
            result = await self.client.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis EXISTS failed for key '{key}': {e}")
            return False

    async def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.
        Uses Redis LIST to store messages in order.
        """
        try:
            await self._ensure_connected()
            key = f"session:{session_id}:history"
            
            # Get all messages from list (LRANGE 0 -1)
            messages = await self.client.lrange(key, 0, -1)
            
            # Parse each message as JSON
            history = []
            for msg in messages:
                try:
                    history.append(json.loads(msg))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse message in session {session_id}")
            
            return history
            
        except (ConnectionError, OSError) as e:
            # Connection errors are already logged by _ensure_connected with rate limiting
            # Log a warning about empty result due to connection issue
            logger.warning(f"Returning empty history for session '{session_id}' due to Redis connection issue: {type(e).__name__}")
            return []
        except Exception as e:
            logger.error(f"Failed to get session history for '{session_id}': {e}")
            return []

    async def append_to_session(
        self,
        session_id: str,
        role: str,
        content: str,
        max_messages: int = 10
    ) -> bool:
        """
        Append message to session history.
        Uses RPUSH to add to end of list, then LTRIM to keep only last N messages.
        """
        try:
            await self._ensure_connected()
            key = f"session:{session_id}:history"
            
            # Create message object
            message = json.dumps({"role": role, "content": content})
            
            # Add to end of list
            await self.client.rpush(key, message)
            
            # Trim to keep only last max_messages
            await self.client.ltrim(key, -max_messages, -1)
            
            # Set expiration (24 hours)
            await self.client.expire(key, 86400)
            
            return True
            
        except (ConnectionError, OSError) as e:
            # Connection errors are already logged by _ensure_connected with rate limiting
            # Don't log again to avoid duplicate error messages
            return False
        except Exception as e:
            logger.error(f"Failed to append to session '{session_id}': {e}")
            return False

    async def get_session_facts(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get extracted facts for a session (e.g., claim numbers, member info).
        Uses Redis LIST to store facts.
        """
        try:
            await self._ensure_connected()
            key = f"session:{session_id}:facts"
            
            # Get all facts from list
            facts = await self.client.lrange(key, 0, -1)
            
            # Parse each fact as JSON
            parsed_facts = []
            for fact in facts:
                try:
                    parsed_facts.append(json.loads(fact))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse fact in session {session_id}")
            
            return parsed_facts
            
        except (ConnectionError, OSError) as e:
            # Connection errors are already logged by _ensure_connected with rate limiting
            # Log a warning about empty result due to connection issue
            logger.warning(f"Returning empty facts for session '{session_id}' due to Redis connection issue: {type(e).__name__}")
            return []
        except Exception as e:
            logger.error(f"Failed to get session facts for '{session_id}': {e}")
            return []

    async def add_session_fact(
        self,
        session_id: str,
        fact_type: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Add an extracted fact to session.
        Uses Redis LIST to append facts.
        """
        try:
            await self._ensure_connected()
            key = f"session:{session_id}:facts"
            
            # Create fact object
            fact = json.dumps({
                "type": fact_type,
                "data": data
            })
            
            # Add to list
            await self.client.rpush(key, fact)
            
            # Set expiration (24 hours)
            await self.client.expire(key, 86400)
            
            return True
            
        except (ConnectionError, OSError) as e:
            # Connection errors are already logged by _ensure_connected with rate limiting
            # Don't log again to avoid duplicate error messages
            return False
        except Exception as e:
            logger.error(f"Failed to add fact to session '{session_id}': {e}")
            return False

    async def clear_session(self, session_id: str) -> bool:
        """
        Clear all session data.
        Deletes all keys matching session:{session_id}:*
        """
        try:
            await self._ensure_connected()
            
            # Find all keys for this session
            pattern = f"session:{session_id}:*"
            keys = []
            
            # Scan for keys (better than KEYS for production)
            cursor = 0
            while True:
                cursor, partial_keys = await self.client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                keys.extend(partial_keys)
                if cursor == 0:
                    break
            
            # Delete all found keys
            if keys:
                await self.client.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys for session '{session_id}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear session '{session_id}': {e}")
            return False

    async def list_all_sessions(self) -> List[str]:
        """
        List all session IDs that have data in Redis.
        Scans for keys matching session:*:history pattern.
        
        Returns:
            List of unique session IDs
        """
        try:
            await self._ensure_connected()
            
            # Scan for all session history keys
            pattern = "session:*:history"
            session_ids = set()
            
            cursor = 0
            while True:
                cursor, keys = await self.client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                
                # Extract session IDs from keys (format: session:{session_id}:history)
                for key in keys:
                    if isinstance(key, bytes):
                        key = key.decode('utf-8')
                    # Extract session_id from "session:{session_id}:history"
                    parts = key.split(':')
                    if len(parts) >= 2:
                        session_ids.add(parts[1])
                
                if cursor == 0:
                    break
            
            return sorted(list(session_ids))
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    async def get_all_session_keys(self, session_id: str) -> List[str]:
        """
        Get all Redis keys for a specific session.
        
        Args:
            session_id: The session ID to query
            
        Returns:
            List of all keys for this session
        """
        try:
            await self._ensure_connected()
            pattern = f"session:{session_id}:*"
            keys = []
            
            cursor = 0
            while True:
                cursor, partial_keys = await self.client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                # Decode bytes to strings if needed
                decoded_keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in partial_keys]
                keys.extend(decoded_keys)
                if cursor == 0:
                    break
            
            return sorted(keys)
            
        except (ConnectionError, OSError) as e:
            logger.warning(f"Returning empty keys for session '{session_id}' due to Redis connection issue: {type(e).__name__}")
            return []
        except Exception as e:
            logger.error(f"Failed to get session keys for '{session_id}': {e}")
            return []

    async def is_connected(self) -> bool:
        """
        Check if Redis is currently connected.
        
        Returns:
            True if connected and responsive, False otherwise
        """
        try:
            if self.client is None:
                return False
            await asyncio.wait_for(self.client.ping(), timeout=2.0)
            return True
        except Exception:
            return False
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """
        Get detailed connection status information.
        
        Returns:
            Dictionary with connection status, host, port, and error info if disconnected
        """
        status = {
            "connected": False,
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "ssl": self.ssl,
            "has_client": self.client is not None,
            "connection_failed": self._connection_failed
        }
        
        try:
            if await self.is_connected():
                status["connected"] = True
                # Try to get Redis info
                try:
                    info = await self.client.info("server")
                    status["redis_version"] = info.get("redis_version", "unknown")
                except Exception:
                    pass
            else:
                status["error"] = "Not connected to Redis"
                if self._connection_failed:
                    status["error"] = "Connection failed (circuit breaker active)"
        except Exception as e:
            status["error"] = str(e)
        
        return status

    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            try:
                await self.client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
        self.client = None
