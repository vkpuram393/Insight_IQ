"""
Overrides_api.cache_helpers — Redis cache helper for the override domain.

Builds the override-specific cache key and delegates to the existing
tools.api_cache.get_cached_response / set_cached_response. The shared cache
infrastructure is domain-generic (MemoryStoreFactory + Pydantic settings); only
the key format is overridden here.

Key format (per spec):
    session:{session_id}:api_cache:overrides:{user_id}_{claim_id}

The `overrides:` namespace prefix prevents collisions with the CHS / CAP key
format (session:{sid}:api_cache:{uid}_{cn}_{sn}).
"""

import logging
from typing import Any, Dict, Optional

from config.config import settings
from tools.api_cache import get_cached_response, set_cached_response

logger = logging.getLogger(__name__)


def build_overrides_cache_key(
    *,
    session_id: str,
    user_id: str,
    claim_id: str,
) -> Optional[str]:
    """
    Build the override-domain cache key.

    Returns None when any required component is missing — callers MUST treat
    None as "do not use cache" and proceed with a live API call.
    """
    sid = (session_id or "").strip()
    uid = (user_id or "").strip()
    cid = (claim_id or "").strip()
    if not sid or not uid or not cid:
        logger.debug("[OverridesCache] Cache key skipped — missing component "
                     "(sid=%r uid=%r cid=%r)", bool(sid), bool(uid), bool(cid))
        return None
    return f"session:{sid}:api_cache:overrides:{uid}_{cid}"


def coerce_user_id(state: Dict[str, Any]) -> str:
    """
    Extract a stable user_id from state.user_info, falling back to session_id
    when no user_id is set. Never returns "anonymous" (which would cause
    cross-session cache collisions).
    """
    user_info = state.get("user_info") or {}
    candidates = (
        user_info.get("user_id"),
        user_info.get("userId"),
        user_info.get("uid"),
        state.get("user_session"),
        state.get("session_id"),
    )
    for c in candidates:
        if c:
            return str(c).strip()
    return ""


def coerce_claim_id(state: Dict[str, Any]) -> str:
    """
    Resolve claim_id from state with the same priority order as
    Claims_search_api.claims_search_node_v2._coerce_claim_id.

    Centralizes this logic so the cache write-key and read-key are derived
    from the same source order — preventing write-here / read-elsewhere bugs.
    """
    entities = state.get("entities") or {}
    extracted_slots = state.get("extracted_slots") or {}
    user_info = state.get("user_info") or {}

    candidates = (
        entities.get("claim_ids"),
        entities.get("claim_id"),
        entities.get("claimNumber"),
        entities.get("claim_number"),
        extracted_slots.get("claim_ids"),
        extracted_slots.get("claim_id"),
        extracted_slots.get("claimNumber"),
        extracted_slots.get("claim_number"),
        user_info.get("claim_id"),
    )
    for c in candidates:
        if not c:
            continue
        if isinstance(c, list):
            if c:
                return str(c[0]).strip()
        else:
            return str(c).strip()
    return ""


async def get_cached_overrides(
    *, session_id: str, user_id: str, claim_id: str,
) -> Optional[Dict[str, Any]]:
    """Read-through helper. Returns None on miss / disabled / error."""
    key = build_overrides_cache_key(session_id=session_id, user_id=user_id, claim_id=claim_id)
    if not key:
        return None
    cached = await get_cached_response(key)
    if cached is not None:
        logger.info("[OverridesCache] HIT  key=%s", key)
        return cached
    logger.debug("[OverridesCache] MISS key=%s", key)
    return None


async def set_cached_overrides(
    *,
    session_id: str,
    user_id: str,
    claim_id: str,
    response_data: Dict[str, Any],
) -> bool:
    """
    Write-through helper. Skips empty payloads (no PA records → don't cache).

    TTL is read from settings.overrides_api_cache_ttl_seconds (default 900s).
    """
    key = build_overrides_cache_key(session_id=session_id, user_id=user_id, claim_id=claim_id)
    if not key:
        return False

    pa_records = (response_data or {}).get("priorAuthorizations") or []
    if not pa_records:
        logger.info("[OverridesCache] Skipping write — empty priorAuthorizations.")
        return False

    ttl = getattr(settings, "overrides_api_cache_ttl_seconds", 900)
    ok = await set_cached_response(key, response_data, ttl_seconds=ttl)
    if ok:
        logger.info("[OverridesCache] WROTE key=%s ttl=%ss records=%d",
                    key, ttl, len(pa_records))
    return ok
