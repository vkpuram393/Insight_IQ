"""
Memory Cleanup Utilities

Provides functions to clean up memory after tests and optimize memory usage.
"""

import gc
import asyncio
from typing import Dict, Any
from core.logger import get_logger
from config.config import settings

logger = get_logger(__name__)


async def cleanup_after_tests() -> Dict[str, Any]:
    """
    Comprehensive cleanup function to be called after test execution.
    
    This function:
    1. Cleans old sessions from memory store
    2. Cleans old checkpoints from LangGraph
    3. Clears LRU caches
    4. Triggers garbage collection
    5. Clears expired cache entries
    
    Returns:
        Dict with cleanup statistics
    """
    results = {
        "sessions_cleaned": 0,
        "checkpoints_cleaned": 0,
        "cache_keys_cleaned": 0,
        "objects_collected": 0,
        "lru_caches_cleared": 0
    }
    
    try:
        # 1. Clean old sessions from memory store
        try:
            from memory import MemoryStoreFactory
            memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
            if hasattr(memory_store, '_cleanup_old_sessions'):
                sessions_cleaned = await memory_store._cleanup_old_sessions()
                results["sessions_cleaned"] = sessions_cleaned
                logger.info(f"🧹 Cleaned {sessions_cleaned} old sessions")
            
            # Clear expired cache entries
            if hasattr(memory_store, 'clear_expired'):
                cache_cleaned = await memory_store.clear_expired()
                results["cache_keys_cleaned"] = cache_cleaned
                logger.info(f"🧹 Cleaned {cache_cleaned} expired cache entries")
        except Exception as e:
            logger.warning(f"⚠️ Memory store cleanup failed: {e}")
        
        # 2. Clean old checkpoints
        try:
            from langgraph_agent import cleanup_old_checkpoints
            checkpoints_cleaned = await cleanup_old_checkpoints(days=1)  # More aggressive for tests
            results["checkpoints_cleaned"] = checkpoints_cleaned
            logger.info(f"🧹 Cleaned {checkpoints_cleaned} old checkpoints")
        except Exception as e:
            logger.warning(f"⚠️ Checkpoint cleanup failed: {e}")
        
        # 3. Clear LRU caches
        try:
            from tools.api_repository import get_api_repository
            # Clear the LRU cache - lru_cache decorator adds cache_clear method
            if hasattr(get_api_repository, 'cache_clear'):
                get_api_repository.cache_clear()
                results["lru_caches_cleared"] = 1
                logger.info("🧹 Cleared API repository LRU cache")
            else:
                # Try to access the underlying cache_info to verify it exists
                cache_info = getattr(get_api_repository, 'cache_info', None)
                if cache_info:
                    logger.debug(f"LRU cache info: {cache_info()}")
        except Exception as e:
            logger.warning(f"⚠️ LRU cache clear failed: {e}")
        
        # 4. Trigger garbage collection
        try:
            collected = gc.collect()
            results["objects_collected"] = collected
            logger.info(f"🧹 Garbage collection: {collected} objects collected")
        except Exception as e:
            logger.warning(f"⚠️ GC failed: {e}")
        
        # 5. Clear embedding classifier cache if needed (only if using in-memory embeddings)
        try:
            from config.config import settings
            if settings.use_embedding_classifier and not settings.use_mongodb_for_embeddings:
                # In-memory embeddings mode - check if we can optimize
                from classifiers.embedded_classifier import _embedded_classifier_instance
                if _embedded_classifier_instance is not None:
                    # Embeddings are loaded in memory - this is expected and necessary
                    # We can't clear them as they're needed for classification
                    # But we can log the size for monitoring
                    import sys
                    if hasattr(_embedded_classifier_instance, 'intent_embeddings'):
                        embeddings = _embedded_classifier_instance.intent_embeddings
                        if embeddings:
                            # Estimate memory size (rough calculation)
                            total_size = sum(
                                arr.nbytes if hasattr(arr, 'nbytes') else sys.getsizeof(arr)
                                for arr in embeddings.values()
                            )
                            logger.debug(f"📊 Embedding classifier memory: ~{total_size / 1024 / 1024:.1f} MB")
        except Exception as e:
            logger.debug(f"Could not check embedding classifier memory: {e}")
        
        logger.info(f"✅ Memory cleanup completed: {results}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Memory cleanup error: {e}")
        return results


async def force_memory_cleanup() -> Dict[str, Any]:
    """
    Force aggressive memory cleanup - use after heavy test runs.
    
    This is more aggressive than regular cleanup:
    - Cleans sessions older than 1 hour (instead of 24 hours)
    - Cleans checkpoints older than 1 day (instead of 7 days)
    - Runs GC multiple times
    """
    results = {
        "sessions_cleaned": 0,
        "checkpoints_cleaned": 0,
        "cache_keys_cleaned": 0,
        "objects_collected": 0,
        "lru_caches_cleared": 0
    }
    
    try:
        # Aggressive session cleanup (1 hour TTL)
        try:
            from memory import MemoryStoreFactory
            memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
            if hasattr(memory_store, '_session_ttl_hours'):
                original_ttl = memory_store._session_ttl_hours
                memory_store._session_ttl_hours = 1  # 1 hour for aggressive cleanup
                sessions_cleaned = await memory_store._cleanup_old_sessions()
                memory_store._session_ttl_hours = original_ttl  # Restore
                results["sessions_cleaned"] = sessions_cleaned
                logger.info(f"🧹 Aggressively cleaned {sessions_cleaned} old sessions")
            
            # Clear all expired cache
            if hasattr(memory_store, 'clear_expired'):
                cache_cleaned = await memory_store.clear_expired()
                results["cache_keys_cleaned"] = cache_cleaned
        except Exception as e:
            logger.warning(f"⚠️ Aggressive memory store cleanup failed: {e}")
        
        # Aggressive checkpoint cleanup (1 day)
        try:
            from langgraph_agent import cleanup_old_checkpoints
            checkpoints_cleaned = await cleanup_old_checkpoints(days=1)
            results["checkpoints_cleaned"] = checkpoints_cleaned
        except Exception as e:
            logger.warning(f"⚠️ Aggressive checkpoint cleanup failed: {e}")
        
        # Clear LRU caches
        try:
            from tools.api_repository import get_api_repository
            if hasattr(get_api_repository, 'cache_clear'):
                get_api_repository.cache_clear()
                results["lru_caches_cleared"] = 1
                logger.info("🧹 Cleared API repository LRU cache (aggressive)")
        except Exception as e:
            logger.warning(f"⚠️ LRU cache clear failed: {e}")
        
        # Multiple GC passes
        total_collected = 0
        for _ in range(3):
            collected = gc.collect()
            total_collected += collected
        results["objects_collected"] = total_collected
        logger.info(f"🧹 Aggressive GC: {total_collected} objects collected")
        
        logger.info(f"✅ Aggressive memory cleanup completed: {results}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Aggressive memory cleanup error: {e}")
        return results

