"""
Overrides_api.overrides_node — LangGraph node `call_overrides_tool`.

Plugs into the same routing slot as call_claims_search_node_v2:

    build_context  →  _route_after_build_context
                       ├── call_claims_search   (CHS / member history)
                       ├── call_overrides_tool  (NEW — PA lookup)
                       └── call_claims_tool     (CAP / single claim)

Responsibilities (mirrors Claims_search_api/claims_search_node_v2.py):
  1. Resolve claim_id from state (entities → extracted_slots → user_info).
  2. Read auth headers from state['user_info'].
  3. Cache lookup via Overrides_api.cache_helpers.
  4. Two-step API call (Step 1: CAGM resolve, Step 2: PA records).
  5. Trim & slim the records for downstream LLM consumption.
  6. Cache write (TTL from settings.overrides_api_cache_ttl_seconds).
  7. Write structured tool_results envelope to state — NEVER raise.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional

from .api_utils import (
    get_member_cagm_from_claim,
    get_overrides_by_member,
)
from .cache_helpers import (
    coerce_claim_id,
    coerce_user_id,
    get_cached_overrides,
    set_cached_overrides,
)
from .llm_query_responder import prepare_overrides_data
from .llm_formatter import format_pa_records_for_llm
from .response_trimmer import extract_member_summary_from_cagm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _failure_payload(
    error: str,
    *,
    needs_clarification: bool = False,
    missing_entities: Optional[List[str]] = None,
    domain_mapping: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Canonical failure return shape — mirrors claims_search_node_v2._failure_payload
    so downstream consumers (response_agent) see a consistent envelope.
    """
    payload: Dict[str, Any] = {
        "tool_results": {
            "tool_name": "overrides_v1",
            "status":    "failure",
            "success":   False,
            "error":     error,
            "data": {
                "is_override_search": True,
                "priorAuthorizations": [],
                "_slim_pa_records":   [],
                "_member_summary":    {},
                "totalCount":         0,
                "filteredCount":      0,
            },
        },
    }
    if needs_clarification:
        payload["needs_clarification"] = True
        payload["clarification_context"] = {
            "missing_entities": missing_entities or ["claim_number"],
            "intent_domain":    "override_domain",
        }
    if domain_mapping:
        payload["domain_mapping"] = domain_mapping
    return payload


def _domain_mapping_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build / preserve the domain_mapping dict for state.
    If extended_intent_agent_node already set one, reuse the intent.
    """
    existing = state.get("domain_mapping") or {}
    intent = (existing.get("intent")
              or state.get("intent")
              or "")
    return {
        "intent":           intent,
        "domain":           "override_domain",
        "route_target":     "call_overrides_tool",
        "render_config":    "override_domain",
        "tool_result_flag": "is_override_search",
    }


def _read_auth(state: Dict[str, Any]) -> Dict[str, str]:
    user_info = state.get("user_info") or {}
    return {
        "bearer_token":  str(user_info.get("bearer_token") or user_info.get("auth_token") or ""),
        "x_api_key":     str(user_info.get("x_api_key") or user_info.get("xApiKey") or ""),
        "x_clientrefid": str(user_info.get("x_clientrefid") or user_info.get("clientRefId") or ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

async def call_overrides_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async LangGraph node — orchestrates Step 1 + Step 2 of the Overrides API
    plus cache and PII-safe trimming.

    Always writes a `tool_results` envelope. Never raises.
    """
    domain_mapping = _domain_mapping_from_state(state)

    # ── 1. Resolve claim_id (mandatory entity) ────────────────────────────
    claim_id = coerce_claim_id(state)
    if not claim_id:
        logger.info("[OverridesNode] Missing claim_id → clarification.")
        return _failure_payload(
            "Missing claim_number — required for Prior Authorization lookup.",
            needs_clarification=True,
            missing_entities=["claim_number"],
            domain_mapping=domain_mapping,
        )

    session_id = (state.get("session_id") or "").strip()
    user_id    = coerce_user_id(state)

    # ── 2. Auth ───────────────────────────────────────────────────────────
    auth = _read_auth(state)
    if not auth["bearer_token"]:
        logger.warning("[OverridesNode] Missing bearer_token in user_info.")
        return _failure_payload(
            "Missing authorization token — cannot call Overrides API.",
            domain_mapping=domain_mapping,
        )

    # ── 3. Cache lookup ───────────────────────────────────────────────────
    try:
        cached = await get_cached_overrides(
            session_id=session_id, user_id=user_id, claim_id=claim_id,
        )
    except Exception as exc:
        logger.warning("[OverridesNode] Cache read failed: %s", exc)
        cached = None

    if cached and isinstance(cached, dict):
        # cache stores {"data": <api_response>, "cached_at": ...}
        api_response = cached.get("data") or cached
        cagm_from_cache = (cached.get("data") or {}).get("_cagm_used") or {}
        prep = prepare_overrides_data(api_response, cagm=cagm_from_cache)
        logger.info("[OverridesNode] Cache HIT — skipping API calls.")
        return _success_payload(
            api_response=api_response,
            cagm=cagm_from_cache,
            claim_id=claim_id,
            prep=prep,
            cache_hit=True,
            domain_mapping=domain_mapping,
        )

    # ── 4. Step 1 — resolve CAGM ──────────────────────────────────────────
    try:
        cagm = await get_member_cagm_from_claim(
            claim_number=claim_id,
            bearer_token=auth["bearer_token"],
            x_api_key=auth["x_api_key"],
            x_clientrefid=auth["x_clientrefid"],
        )
    except Exception as exc:
        logger.error("[OverridesNode] Step 1 raised: %s\n%s", exc, traceback.format_exc())
        return _failure_payload(
            f"Step 1 (claim_list) error: {type(exc).__name__}: {exc}",
            domain_mapping=domain_mapping,
        )

    if not cagm:
        return _failure_payload(
            "Could not resolve member from the provided claim_number. "
            "Please double-check the claim number.",
            domain_mapping=domain_mapping,
        )

    # ── 5. Step 2 — fetch PA records ──────────────────────────────────────
    try:
        api_response = await get_overrides_by_member(
            cagm=cagm,
            bearer_token=auth["bearer_token"],
            x_api_key=auth["x_api_key"],
            x_clientrefid=auth["x_clientrefid"],
        )
    except Exception as exc:
        logger.error("[OverridesNode] Step 2 raised: %s\n%s", exc, traceback.format_exc())
        return _failure_payload(
            f"Step 2 (overrides_api) error: {type(exc).__name__}: {exc}",
            domain_mapping=domain_mapping,
        )

    # ── 6. Trim + slim ────────────────────────────────────────────────────
    try:
        prep = prepare_overrides_data(api_response, cagm=cagm)
    except Exception as exc:
        logger.error("[OverridesNode] Data prep raised: %s\n%s", exc, traceback.format_exc())
        return _failure_payload(
            f"Failed to prepare PA data: {type(exc).__name__}: {exc}",
            domain_mapping=domain_mapping,
        )

    # ── 7. Cache write (only on success with non-empty payload) ───────────
    is_fallback = bool((api_response or {}).get("_fallback_note"))
    if not is_fallback:
        try:
            # stamp the CAGM into the payload before caching so a cache hit
            # can reconstruct the member summary without re-running Step 1
            payload_to_cache = dict(api_response)
            payload_to_cache["_cagm_used"] = cagm
            await set_cached_overrides(
                session_id=session_id, user_id=user_id, claim_id=claim_id,
                response_data=payload_to_cache,
            )
        except Exception as exc:
            logger.warning("[OverridesNode] Cache write failed: %s", exc)

    return _success_payload(
        api_response=api_response,
        cagm=cagm,
        claim_id=claim_id,
        prep=prep,
        cache_hit=False,
        domain_mapping=domain_mapping,
    )


def _success_payload(
    *,
    api_response: Dict[str, Any],
    cagm: Dict[str, Any],
    claim_id: str,
    prep: Dict[str, Any],
    cache_hit: bool,
    domain_mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Canonical success return — mirrors claims_search_node_v2 envelope."""
    member_summary = prep.get("member_summary") or extract_member_summary_from_cagm(cagm)
    slim_records = prep.get("slim_pa_records") or []

    # Build compact text summary for _masked_response (mirrors CHS _masked_response).
    try:
        masked_response = format_pa_records_for_llm(slim_records, member_summary)
    except Exception as _fmt_exc:
        logger.warning("[OverridesNode] format_pa_records_for_llm failed (%s)", _fmt_exc)
        masked_response = (
            f"PA records: {len(slim_records)} record(s) available for LLM."
        )

    return {
        "tool_results": {
            "tool_name": "overrides_v1",
            "status":    "success",
            "success":   True,
            "error":     "",
            "data": {
                "is_override_search":   True,
                "priorAuthorizations":  (api_response or {}).get("priorAuthorizations") or [],
                "_slim_pa_records":     slim_records,
                "_member_summary":      member_summary,
                "_masked_response":     masked_response,
                "_fallback_note":       prep.get("_fallback_note"),
                "totalCount":           prep.get("total_count", 0),
                "filteredCount":        prep.get("filtered_count", 0),
                "claim_id":             claim_id,
            },
        },
        "cache_hit":      bool(cache_hit),
        "domain_mapping": domain_mapping,
    }
