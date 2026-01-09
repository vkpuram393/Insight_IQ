"""
Redis Checkpointer for LangGraph

Stores AgentState checkpoints in Redis instead of SQLite to reduce in-memory state accumulation.
This helps prevent pod memory spikes by keeping state in Redis rather than accumulating in memory.

Key Benefits:
- State stored in Redis (external memory) instead of pod memory
- Reduces memory footprint during graph execution
- Enables state sharing across multiple pod instances
- Automatic TTL for cleanup of old checkpoints
"""

import json
import time
from typing import Optional, List, Dict, Any, AsyncIterator, Tuple, Sequence
from datetime import datetime, timezone

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.base import CheckpointTuple
from langchain_core.runnables import RunnableConfig
from core.logger import get_logger

logger = get_logger(__name__)


class RedisCheckpointSaver(BaseCheckpointSaver):
    """
    Redis-based checkpointer for LangGraph.
    
    Stores checkpoints in Redis with the following key structure:
    - checkpoint:{thread_id}:{checkpoint_id} -> checkpoint data (JSON)
    - checkpoint:{thread_id}:metadata:{checkpoint_id} -> metadata (JSON)
    - checkpoint:{thread_id}:versions -> latest version number (string)
    - checkpoint:{thread_id}:list -> sorted set of checkpoint IDs (for listing)
    
    Args:
        redis_client: Async Redis client instance
        ttl_seconds: Time-to-live for checkpoints (default: 24 hours)
        key_prefix: Prefix for all Redis keys (default: "checkpoint")
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_seconds: int = 86400,  # 24 hours
        key_prefix: str = "checkpoint"
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package required. Install with: pip install redis")
        
        self.client = redis_client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        
        logger.info(f"Initialized RedisCheckpointSaver (TTL: {ttl_seconds}s, prefix: {key_prefix})")
    
    def _checkpoint_key(self, thread_id: str, checkpoint_id: str) -> str:
        """Generate Redis key for checkpoint data"""
        return f"{self.key_prefix}:{thread_id}:{checkpoint_id}"
    
    def _metadata_key(self, thread_id: str, checkpoint_id: str) -> str:
        """Generate Redis key for checkpoint metadata"""
        return f"{self.key_prefix}:{thread_id}:metadata:{checkpoint_id}"
    
    def _versions_key(self, thread_id: str) -> str:
        """Generate Redis key for version tracking"""
        return f"{self.key_prefix}:{thread_id}:versions"
    
    def _list_key(self, thread_id: str) -> str:
        """Generate Redis key for checkpoint list (sorted set)"""
        return f"{self.key_prefix}:{thread_id}:list"
    
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
        parent_checkpoint_id: Optional[str] = None
    ) -> RunnableConfig:
        """
        Save a checkpoint to Redis.
        
        Args:
            config: Runnable config containing thread_id
            checkpoint: Checkpoint data to save
            metadata: Checkpoint metadata
            new_versions: Channel versions
            parent_checkpoint_id: Optional parent checkpoint ID
            
        Returns:
            Updated RunnableConfig with checkpoint_id
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError("thread_id required in config.configurable")
        
        checkpoint_id = checkpoint.get("id") or f"checkpoint_{int(time.time() * 1000000)}"
        
        try:
            # Serialize checkpoint and metadata
            checkpoint_json = json.dumps(checkpoint, default=self._json_serializer)
            metadata_json = json.dumps(metadata, default=self._json_serializer)
            
            # Store checkpoint data (encode to bytes if decode_responses=False)
            checkpoint_key = self._checkpoint_key(thread_id, checkpoint_id)
            checkpoint_bytes = checkpoint_json.encode('utf-8') if isinstance(checkpoint_json, str) else checkpoint_json
            await self.client.setex(
                checkpoint_key,
                self.ttl_seconds,
                checkpoint_bytes
            )
            
            # Store metadata
            metadata_key = self._metadata_key(thread_id, checkpoint_id)
            metadata_bytes = metadata_json.encode('utf-8') if isinstance(metadata_json, str) else metadata_json
            await self.client.setex(
                metadata_key,
                self.ttl_seconds,
                metadata_bytes
            )
            
            # Update version tracking
            versions_key = self._versions_key(thread_id)
            versions_json = json.dumps(new_versions, default=self._json_serializer)
            versions_bytes = versions_json.encode('utf-8') if isinstance(versions_json, str) else versions_json
            await self.client.setex(
                versions_key,
                self.ttl_seconds,
                versions_bytes
            )
            
            # Add to sorted set for listing (use timestamp as score)
            list_key = self._list_key(thread_id)
            timestamp_raw = checkpoint.get("ts", time.time())
            # Ensure timestamp is a float (zadd requires float score)
            if timestamp_raw is None:
                timestamp = time.time()
            elif isinstance(timestamp_raw, str):
                try:
                    timestamp = float(timestamp_raw)
                except (ValueError, TypeError):
                    timestamp = time.time()
            elif isinstance(timestamp_raw, (int, float)):
                timestamp = float(timestamp_raw)
            else:
                # For datetime or other types, use current time
                timestamp = time.time()
            await self.client.zadd(list_key, {checkpoint_id: timestamp})
            await self.client.expire(list_key, self.ttl_seconds)
            
            logger.debug(f"Saved checkpoint {checkpoint_id} for thread {thread_id}")
            
            # Update config with checkpoint_id
            config = config.copy() if config else {}
            if "configurable" not in config:
                config["configurable"] = {}
            config["configurable"]["checkpoint_id"] = checkpoint_id
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint to Redis: {e}", exc_info=True)
            raise
    
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = ""
    ) -> None:
        """
        Store intermediate writes linked to a checkpoint.
        
        This method is called during graph execution to store intermediate state writes
        that are linked to a checkpoint. These writes represent channel updates that
        happen before the final checkpoint is saved.
        
        Args:
            config: Configuration of the related checkpoint
            writes: List of writes to store (tuples of channel_name and value)
            task_id: Identifier for the task creating the writes
            task_path: Path of the task creating the writes
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            # If no thread_id, this is likely a no-op scenario
            return
        
        if not writes:
            return
        
        try:
            # Store intermediate writes with a key linked to the task
            # Format: writes:{thread_id}:{task_id} -> JSON array of writes
            writes_key = f"{self.key_prefix}:writes:{thread_id}:{task_id}"
            
            # Serialize writes
            writes_data = {
                "writes": list(writes),
                "task_id": task_id,
                "task_path": task_path,
                "timestamp": time.time()
            }
            writes_json = json.dumps(writes_data, default=self._json_serializer)
            writes_bytes = writes_json.encode('utf-8') if isinstance(writes_json, str) else writes_json
            
            # Store with TTL (shorter than checkpoints since these are intermediate)
            await self.client.setex(
                writes_key,
                self.ttl_seconds,
                writes_bytes
            )
            
            logger.debug(
                f"Stored {len(writes)} intermediate write(s) for thread {thread_id} "
                f"(task: {task_id})"
            )
            
        except Exception as e:
            # Log but don't raise - intermediate writes are not critical
            # The main checkpoint will still be saved via aput()
            logger.warning(f"Failed to store intermediate writes to Redis: {e}")
    
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        Retrieve a checkpoint tuple (checkpoint + metadata) from Redis.
        
        Args:
            config: Runnable config containing thread_id and optionally checkpoint_id
            
        Returns:
            CheckpointTuple if found, None otherwise
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        
        if not thread_id:
            return None
        
        # If no checkpoint_id specified, get the latest
        if not checkpoint_id:
            checkpoint_id = await self._get_latest_checkpoint_id(thread_id)
            if not checkpoint_id:
                return None
        
        try:
            # Retrieve checkpoint data
            checkpoint_key = self._checkpoint_key(thread_id, checkpoint_id)
            checkpoint_data = await self.client.get(checkpoint_key)
            
            if not checkpoint_data:
                return None
            
            # Retrieve metadata
            metadata_key = self._metadata_key(thread_id, checkpoint_id)
            metadata_data = await self.client.get(metadata_key)
            
            # Decode bytes to string if needed
            if isinstance(checkpoint_data, bytes):
                checkpoint_json = checkpoint_data.decode('utf-8')
            else:
                checkpoint_json = checkpoint_data
            
            if isinstance(metadata_data, bytes):
                metadata_json = metadata_data.decode('utf-8')
            elif metadata_data:
                metadata_json = metadata_data
            else:
                metadata_json = "{}"
            
            # Deserialize
            checkpoint = json.loads(checkpoint_json)
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_checkpoint_id=None  # Could be stored in metadata if needed
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve checkpoint from Redis: {e}", exc_info=True)
            return None
    
    async def alist(
        self,
        config: RunnableConfig,
        *,
        before: Optional[str] = None,
        limit: Optional[int] = None
    ) -> AsyncIterator[CheckpointTuple]:
        """
        List checkpoints for a thread.
        
        Args:
            config: Runnable config containing thread_id
            before: Optional checkpoint_id to list before
            limit: Maximum number of checkpoints to return
            
        Yields:
            CheckpointTuple instances
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return
        
        try:
            list_key = self._list_key(thread_id)
            
            # Get checkpoint IDs from sorted set (descending order by timestamp)
            if before:
                # Get IDs before the specified checkpoint
                before_timestamp = await self._get_checkpoint_timestamp(thread_id, before)
                if before_timestamp:
                    checkpoint_ids = await self.client.zrevrangebyscore(
                        list_key,
                        max=before_timestamp - 0.001,  # Slightly before
                        limit=(0, limit or 100)
                    )
                else:
                    checkpoint_ids = []
            else:
                checkpoint_ids = await self.client.zrevrange(
                    list_key,
                    0,
                    (limit or 100) - 1
                )
            
            # Retrieve each checkpoint
            for checkpoint_id in checkpoint_ids:
                if isinstance(checkpoint_id, bytes):
                    checkpoint_id = checkpoint_id.decode('utf-8')
                
                checkpoint_config = config.copy()
                checkpoint_config.setdefault("configurable", {})["checkpoint_id"] = checkpoint_id
                
                tuple_result = await self.aget_tuple(checkpoint_config)
                if tuple_result:
                    yield tuple_result
                    
        except Exception as e:
            logger.error(f"Failed to list checkpoints from Redis: {e}", exc_info=True)
    
    async def adelete_thread(self, config: RunnableConfig) -> None:
        """
        Delete all checkpoints for a thread.
        
        Args:
            config: Runnable config containing thread_id
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return
        
        try:
            # Get all checkpoint IDs
            list_key = self._list_key(thread_id)
            checkpoint_ids = await self.client.zrange(list_key, 0, -1)
            
            # Delete all checkpoint data and metadata
            keys_to_delete = []
            for checkpoint_id in checkpoint_ids:
                if isinstance(checkpoint_id, bytes):
                    checkpoint_id = checkpoint_id.decode('utf-8')
                
                keys_to_delete.append(self._checkpoint_key(thread_id, checkpoint_id))
                keys_to_delete.append(self._metadata_key(thread_id, checkpoint_id))
            
            # Delete all keys
            if keys_to_delete:
                await self.client.delete(*keys_to_delete)
            
            # Delete tracking keys
            await self.client.delete(list_key)
            await self.client.delete(self._versions_key(thread_id))
            
            logger.info(f"Deleted {len(checkpoint_ids)} checkpoints for thread {thread_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete thread checkpoints from Redis: {e}", exc_info=True)
    
    async def _get_latest_checkpoint_id(self, thread_id: str) -> Optional[str]:
        """Get the latest checkpoint ID for a thread"""
        try:
            list_key = self._list_key(thread_id)
            result = await self.client.zrevrange(list_key, 0, 0)  # Get most recent
            if result:
                checkpoint_id = result[0]
                if isinstance(checkpoint_id, bytes):
                    checkpoint_id = checkpoint_id.decode('utf-8')
                return checkpoint_id
            return None
        except Exception as e:
            logger.error(f"Failed to get latest checkpoint ID: {e}", exc_info=True)
            return None
    
    async def _get_checkpoint_timestamp(self, thread_id: str, checkpoint_id: str) -> Optional[float]:
        """Get timestamp for a checkpoint"""
        try:
            list_key = self._list_key(thread_id)
            score = await self.client.zscore(list_key, checkpoint_id)
            return float(score) if score is not None else None
        except Exception as e:
            logger.error(f"Failed to get checkpoint timestamp: {e}", exc_info=True)
            return None
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for datetime and other types"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    async def close(self) -> None:
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            logger.info("Redis checkpointer connection closed")

