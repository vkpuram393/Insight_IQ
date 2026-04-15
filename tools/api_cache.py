"""
Claims API Response Cache Module

This module provides caching functionality for Claims API responses.
It wraps external API calls to avoid redundant network requests for
follow-up questions about the same claim.

Cache Strategy:
- Key format: session:{sessionId}:api_cache:{userId}_{claimNumber}_{sequenceNumber}
- Value format: {"data": <API response>, "cached_at": <ISO timestamp>}
- TTL: Configurable via settings (default 1 hour)
- Graceful degradation: If Redis fails, API calls proceed normally

Author: AI Agent Team
Created: January 2026
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports for JSON handling and timestamps
import json  # Used to serialize/deserialize cache values
from datetime import datetime, timezone  # Used to generate ISO timestamps for debugging

# Type hints for better code clarity and IDE support
from typing import Dict, Any, Optional, Tuple

# Project imports - following the same pattern as other tools/*.py files
from config.config import settings  # Application settings including cache feature flag and TTL
from core.logger import get_logger  # Centralized logging following codebase pattern
from memory import MemoryStoreFactory  # Factory to get Redis/InMemory store instance


# =============================================================================
# LOGGER SETUP
# =============================================================================

# Initialize logger for this module using the standard codebase pattern
# This creates a logger named "tools.api_cache" for easy filtering in logs
logger = get_logger(__name__)


# =============================================================================
# MEMORY STORE INITIALIZATION
# =============================================================================

# Get the memory store instance (Redis in production, InMemory for testing)
# This uses the factory pattern already established in the codebase
# The store type is determined by settings.memory_store_type ("redis", "inmemory", etc.)
_memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)


# =============================================================================
# CACHE KEY GENERATION
# =============================================================================

def generate_cache_key(
    user_id: str,
    session_id: str,
    claim_number: str,
    sequence_number: str
) -> str:
    """
    Generate a unique cache key for a Claims API response.
    
    Key Format: session:{sessionId}:api_cache:{userId}_{claimNumber}_{sequenceNumber}
    
    Why this format?
    - session:{sessionId}: Follows existing codebase pattern for session data
    - Enables automatic cleanup via clear_session() when session ends
    - userId: Additional security layer + audit trail
    - claimNumber + sequenceNumber: Identifies specific claim data
    
    Args:
        user_id: The authenticated user's ID (from auth token/user_info)
        session_id: The current conversation session ID
        claim_number: The 15-digit claim number being queried
        sequence_number: The 3-digit sequence number (e.g., "001")
        
    Returns:
        A formatted cache key string like "session:sess456:api_cache:user123_123456789012345_001"
        
    Example:
        >>> generate_cache_key("user123", "sess456", "123456789012345", "001")
        "session:sess456:api_cache:user123_123456789012345_001"
    """
    # Sanitize user_id - use "anonymous" if not provided (defensive coding)
    # strip() removes any accidental whitespace that could cause key mismatches
    user_id = str(user_id).strip() if user_id else "anonymous"
    
    # Sanitize session_id - use "unknown" if not provided
    # This ensures we always have a valid key even if session tracking fails
    session_id = str(session_id).strip() if session_id else "unknown"
    
    # Sanitize claim_number - must be provided, convert to string just in case
    # API may pass as int, so we convert and strip whitespace
    claim_number = str(claim_number).strip()
    
    # Sanitize sequence_number - default to "000" if not provided
    # This handles cases where sequence might be None or empty
    sequence_number = str(sequence_number).strip() if sequence_number else "000"
    
    # Build cache key following session:{id}:* pattern for auto-cleanup
    # Format: session:{sessionId}:api_cache:{userId}_{claimNumber}_{sequenceNumber}
    # This allows clear_session() to automatically clean up cache when session ends
    cache_key = f"session:{session_id}:api_cache:{user_id}_{claim_number}_{sequence_number}"
    
    # Debug log the generated key for troubleshooting cache issues
    # Uses debug level to avoid flooding logs in production
    logger.debug(f"🔑 Generated cache key: {cache_key}")
    
    # Return the formatted cache key
    return cache_key


# =============================================================================
# CACHE READ OPERATION
# =============================================================================

async def get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to retrieve a cached API response from Redis.
    
    This function implements graceful degradation - if Redis is unavailable
    or any error occurs, it returns None (cache miss) and the system
    continues to call the actual API. This ensures cache failures never
    break the main application flow.
    
    Args:
        cache_key: The cache key generated by generate_cache_key()
        
    Returns:
        - Dict containing {"data": <API response>, "cached_at": <timestamp>} on cache HIT
        - None on cache MISS or any error (graceful degradation)
        
    Example:
        >>> cached = await get_cached_response("user123_sess456_123456789012345_001")
        >>> if cached:
        ...     return cached["data"]  # Use cached response
    """
    # Check if caching is enabled via feature flag
    # This allows instant disable without code changes - just set env var
    if not settings.enable_claims_api_cache:
        # Caching disabled - log at debug level and return None (cache miss)
        logger.debug("⏭️ Cache disabled via feature flag, skipping cache check")
        return None
    
    try:
        # Attempt to read from Redis/memory store
        # _memory_store.get() returns None if key doesn't exist
        cached_value = await _memory_store.get(cache_key)
        
        # Check if we got a cache hit (value exists)
        if cached_value is not None:
            # Parse JSON if the value is a string (Redis returns strings)
            # The memory store may return either dict or JSON string depending on implementation
            if isinstance(cached_value, str):
                cached_value = json.loads(cached_value)
            
            # Log cache hit with emoji for easy visual scanning in logs
            # This helps quickly identify cache effectiveness in log analysis
            logger.info(f"🎯 CACHE HIT: {cache_key}")
            
            # Log additional debug info about when data was cached
            logger.debug(f"   📅 Cached at: {cached_value.get('cached_at', 'unknown')}")
            
            # Return the cached value (contains "data" and "cached_at" fields)
            return cached_value
        
        # Cache miss - key doesn't exist in Redis
        # Log at debug level to avoid excessive logging
        logger.debug(f"💨 CACHE MISS: {cache_key}")
        return None
        
    except json.JSONDecodeError as e:
        # Handle malformed JSON in cache (shouldn't happen, but defensive coding)
        # Log warning and return None to trigger fresh API call
        logger.warning(f"⚠️ Cache JSON parse error for {cache_key}: {e}")
        return None
        
    except Exception as e:
        # Catch-all for any Redis errors (connection issues, timeouts, etc.)
        # We log the error but return None to allow the API call to proceed
        # This is the "graceful degradation" pattern - cache failures don't break the app
        logger.warning(f"⚠️ Cache read error for {cache_key}: {e}")
        logger.debug(f"   ℹ️ Proceeding with API call (graceful degradation)")
        return None


# =============================================================================
# CACHE WRITE OPERATION
# =============================================================================

async def set_cached_response(
    cache_key: str,
    response_data: Dict[str, Any],
    ttl_seconds: Optional[int] = None
) -> bool:
    """
    Store an API response in the cache (Redis).
    
    The cache value is kept simple: just the data and a timestamp.
    All other metadata (user_id, claim_number, etc.) is already encoded
    in the cache key, so we don't duplicate it in the value.
    
    Cache Value Structure:
    {
        "data": <the actual API response>,
        "cached_at": <ISO timestamp for debugging>
    }
    
    This function is non-blocking - if cache write fails, the API response
    is still returned to the user (just not cached for next time).
    
    Args:
        cache_key: The cache key generated by generate_cache_key()
        response_data: The API response to cache (dict)
        ttl_seconds: Optional TTL override (defaults to settings.claims_api_cache_ttl_seconds)
        
    Returns:
        True if cached successfully, False otherwise (non-blocking, failures are logged)
        
    Example:
        >>> success = await set_cached_response(
        ...     cache_key="user123_sess456_123456789012345_001",
        ...     response_data={"claimDetails": {...}},
        ...     ttl_seconds=3600  # 1 hour
        ... )
    """
    # Check if caching is enabled via feature flag
    # If disabled, skip cache write silently
    if not settings.enable_claims_api_cache:
        logger.debug("⏭️ Cache disabled via feature flag, skipping cache write")
        return False
    
    try:
        # Use provided TTL or fall back to configured default (1 hour by default)
        # This allows per-call TTL override if needed for special cases
        ttl = ttl_seconds or settings.claims_api_cache_ttl_seconds
        
        # Build the cache value - keeping it simple as per user's request
        # Only store: data + timestamp (for debugging when data was cached)
        # We don't need to store claim_number, user_id, etc. - those are in the key
        cache_value = {
            "data": response_data,  # The actual API response (claim details, pricing, etc.)
            "cached_at": datetime.now(timezone.utc).isoformat()  # ISO format timestamp for debugging
        }
        
        # Serialize to JSON string for Redis storage
        # Redis stores strings, so we need to serialize the dict
        serialized_value = json.dumps(cache_value)
        
        # Write to Redis with TTL (auto-expires after ttl_seconds)
        # The memory store's set() method handles the Redis SET command with EX option
        success = await _memory_store.set(
            key=cache_key,
            value=serialized_value,
            ttl_seconds=ttl
        )
        
        # Log the result for observability
        if success:
            # Cache write succeeded - log with emoji for easy visual scanning
            logger.info(f"💾 CACHED: {cache_key} (TTL: {ttl}s)")
        else:
            # Cache write returned False (unusual, but possible)
            logger.warning(f"⚠️ Cache write returned False for {cache_key}")
        
        # Return success status (True/False)
        return success
        
    except Exception as e:
        # Cache write failures are non-critical - log and continue
        # The API response is still returned to the user, just not cached
        # This ensures cache issues never break the main flow
        logger.warning(f"⚠️ Cache write error for {cache_key}: {e}")
        logger.debug(f"   ℹ️ Response will not be cached, but user gets their data")
        return False


# =============================================================================
# CACHE-AWARE WRAPPER FUNCTION
# =============================================================================

async def get_cached_or_fetch_enriched_details(
    claim_number: str,
    claim_sequence: str,
    user_id: str,
    session_id: str,
    auth_token: str,
    fetch_function  # The actual API function (combine_claim_details_and_list)
) -> Tuple[Dict[str, Any], bool]:
    """
    Cache-aware wrapper for the enriched claim details API flow.
    
    This is the main function called from claims_api.py. It implements
    the cache-aside pattern:
    1. Generate a cache key from the parameters
    2. Check if response is already cached (cache hit)
    3. If not cached, call the actual API function
    4. Cache the successful response for future requests
    5. Return the data along with a flag indicating if it came from cache
    
    Flow Diagram:
    ┌─────────────────────────────────────────────────────────────────┐
    │  User asks about claim → Generate cache key                     │
    │            │                                                    │
    │            ▼                                                    │
    │  ┌─────────────────┐                                           │
    │  │ Check Redis     │                                           │
    │  └────────┬────────┘                                           │
    │           │                                                    │
    │     ┌─────┴─────┐                                              │
    │     │           │                                              │
    │   HIT?        MISS?                                            │
    │     │           │                                              │
    │     ▼           ▼                                              │
    │  Return      Call API → Cache → Return                         │
    │  cached      response           fresh                          │
    └─────────────────────────────────────────────────────────────────┘
    
    Args:
        claim_number: The 15-digit claim number (e.g., "123456789012345")
        claim_sequence: The 3-digit sequence number (e.g., "001")
        user_id: The authenticated user's ID (for cache key security)
        session_id: The current conversation session ID (for cache key isolation)
        auth_token: The auth token for the API call (passed to fetch_function)
        fetch_function: The actual API function to call on cache miss
                       (combine_claim_details_and_list from claims_api.py)
        
    Returns:
        Tuple of (response_data, from_cache):
        - response_data: The API response (from cache or fresh API call)
        - from_cache: True if served from cache, False if fresh API call
        
    Example:
        >>> result, from_cache = await get_cached_or_fetch_enriched_details(
        ...     claim_number="123456789012345",
        ...     claim_sequence="001",
        ...     user_id="user123",
        ...     session_id="sess456",
        ...     auth_token="Bearer xyz...",
        ...     fetch_function=combine_claim_details_and_list
        ... )
        >>> if from_cache:
        ...     print("Data came from cache!")
    """
    # Step 1: Generate the cache key from all parameters
    # Key format: session:{sessionId}:api_cache:{userId}_{claimNumber}_{sequenceNumber}
    cache_key = generate_cache_key(user_id, session_id, claim_number, claim_sequence)
    
    # Log that we're checking the cache for this claim
    logger.info(f"🔍 Checking cache for claim {claim_number} seq {claim_sequence}")
    logger.debug(f"   🔑 Cache key: {cache_key}")
    
    # Step 2: Try to get cached response from Redis
    cached = await get_cached_response(cache_key)
    
    # Step 3: If cache hit, return cached data immediately
    # This saves the 2-5 second API call!
    if cached is not None:
        # Extract the "data" field from our cache value structure
        # Cache value format: {"data": <response>, "cached_at": <timestamp>}
        logger.info(f"✅ Returning cached response (saved external API call)")
        
        # Return the cached data and True to indicate cache hit
        return cached["data"], True
    
    # Step 4: Cache miss - need to call the actual API
    logger.info(f"📡 Cache miss - calling external API")
    
    # Call the actual API function (combine_claim_details_and_list)
    # This is passed in as a parameter to maintain separation of concerns
    # The function signature is: combine_claim_details_and_list(claimNumber, claimSequence, auth_token)
    result = await fetch_function(claim_number, claim_sequence, auth_token)
    
    # Step 5: Cache the successful response for future requests
    # Note: We only reach here if fetch_function succeeded (no exception)
    # If the API call failed with an exception, it would propagate up
    await set_cached_response(cache_key, result)
    
    # Step 6: Return the fresh data with False to indicate it came from API (not cache)
    logger.info(f"✅ Returning fresh API response (now cached for follow-ups)")
    
    # Return the result and False to indicate this was a fresh API call
    return result, False

