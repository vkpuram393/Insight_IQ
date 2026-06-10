"""
Dynamic Claims API Orchestrator (final version).
Single entry node: call_claims_tool_node(state)
Uses repository-based API registry and returns ToolResult (with AgentError)
"""

import time
import uuid
import asyncio
import httpx
import traceback
from typing import Dict, Any, Optional

from tools.api_repository import get_api_repository  # your repository builder
from core.errors.error_handler import to_agent_error
from utils.retry import async_retry
from core.errors.exceptions import ExternalAPIError, ToolTimeoutError
from tools.api_fallbacks import get_fallback_details, get_fallback_list  # dynamic fallback data
from tools.sequence_transformer import transform_sequences_to_user_format  # deep sequence 999→001 converter

from core.node_models import ToolResult, ToolExecutionStatus
from core.errors.models import (
    AgentError,
    create_validation_error,
    create_api_error,
    create_internal_error
)

from core.node_models import IntentResult
from state.schema import AgentState
from core.logger import get_logger
from core.logging_context import extract_logging_context, log_state_snapshot
from config.config import settings
from persistence import PersistenceStoreFactory
# Note: PII masking is handled by safety layer, not in tools layer

# =============================================================================
# CLAIMS API RESPONSE CACHE
# =============================================================================
# Import cache wrapper for Claims API responses
# This enables caching to avoid redundant API calls for follow-up questions
# Cache key format: session:{sessionId}:api_cache:{userId}_{claimNumber}_{sequenceNumber}
from tools.api_cache import get_cached_or_fetch_enriched_details

# ============================================================================
# LOGGER
# ============================================================================
logger = get_logger(__name__)

# NOTE: Auth token is now passed as parameter to functions, not stored globally
# This ensures thread-safety for millions of concurrent users

# ============================================================================
# SHARED HTTP CLIENT (Memory Optimization)
# ============================================================================
# Reuse a single httpx.AsyncClient instance to prevent memory leaks
# Creating new clients for each request causes connection pool buildup
_shared_http_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()

async def _get_shared_http_client() -> httpx.AsyncClient:
    """Get or create shared httpx.AsyncClient instance for connection pooling"""
    global _shared_http_client
    if _shared_http_client is None:
        async with _client_lock:
            if _shared_http_client is None:
                _shared_http_client = httpx.AsyncClient(
                    timeout=30.0,
                    verify=True,
                    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
                )
                logger.info("🌐 Created shared HTTP client for connection pooling")
    return _shared_http_client

async def _close_shared_http_client():
    """Close shared HTTP client (called on shutdown)"""
    global _shared_http_client
    if _shared_http_client is not None:
        async with _client_lock:
            if _shared_http_client is not None:
                await _shared_http_client.aclose()
                _shared_http_client = None
                logger.info("🔌 Closed shared HTTP client")

# ============================================================================
# MAIN CLAIMS API TOOL NODE
# ============================================================================
# Main Claims API Tool Node
async def call_claims_tool_node(state) -> Dict[str, Any]:
    """
    Primary orchestrator node called by agent.
    Returns ToolResult as dict.
    
    Accepts either:
    INPUT (from state):
        - intent: What data to fetch
        - entities: Parameters (e.g., claim_number)

    OUTPUT (to state):
        - tool_results: API response data
    """
    logger.info("🔧 Node: Call Claims Tool (dynamic router)")
    
    # Extract logging context if called from LangGraph
    log_ctx = extract_logging_context(state) if isinstance(state, dict) else {}
    
    # Extract auth token from user_info - passed as parameter (thread-safe)
    user_info = state.get("user_info", {}) if isinstance(state, dict) else {}
    auth_token = user_info.get("auth_token", "") if isinstance(user_info, dict) else ""
    
    # Debug log to verify token is captured
    if auth_token:
        logger.info(f"🔑 Auth token captured: {auth_token[:30]}...")
    else:
        logger.warning("⚠️ No auth token provided in request headers")
    
    persistence_store = None
    
    try:
        # Log node entry for audit trail
        if log_ctx and settings.enable_telemetry:
            persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
            await persistence_store.log_audit(
                session_id=log_ctx.get("session_id"),
                request_id=log_ctx.get("request_id"),
                user_id=log_ctx.get("user_id"),
                node_name="call_claims_tool",
                event_type="node_entry",
                data={"node": "call_claims_tool"}
            )
        
        # Handle both AgentState (dict) and IntentResult (Pydantic model)
        if isinstance(state, dict):
            # Called from LangGraph with AgentState
            intent = state.get("intent")
            
            # CRITICAL: Merge extracted_slots (from conversation history) with current entities
            # This enables follow-up questions without re-asking for claim_id
            extracted_slots = state.get("extracted_slots", {})
            current_entities = state.get("entities", {})
            
            # Normalize BOTH separately first (so keys like claim_number and claimId both become claimNumber)
            normalized_extracted = normalize_entities(extracted_slots) if extracted_slots else {}
            normalized_current = normalize_entities(current_entities) if current_entities else {}
            
            # Current entities EXPLICITLY take precedence (same keys will overwrite)
            # This fixes the bug where old claim numbers from history were used instead of new ones
            entities = {**normalized_extracted, **normalized_current}
            
            if extracted_slots:
                logger.info(f"🔗 Context-aware: merged {len(extracted_slots)} slots from history + {len(current_entities)} current entities")
                logger.debug(f"   From history: {extracted_slots}")
                logger.debug(f"   From current: {current_entities}")
                logger.debug(f"   Final merged: {entities}")
            else:
                logger.debug(f"Input type: AgentState (dict), entities: {entities}")
        else:
            # Called directly with IntentResult model (for testing)
            intent = state.intent
            entities = state.entities if state.entities else {}
            logger.debug(f"Input type: IntentResult (Pydantic model)")
    
        # Validate intent exists
        if not intent:
            logger.warning("⚠️ Validation failed: No intent provided")
            ae = create_validation_error(
                message="No intent provided in state",
                field="intent",
                value=None
            )
            tool_result = ToolResult(
                tool_name="claims_api",
                status=ToolExecutionStatus.FAILURE,
                data={},
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                api_endpoint=None,
                http_status_code=None,
                is_retryable=False
            )
            # Log to exceptions collection for debugging (additive - existing log_audit above preserved)
            try:
                if log_ctx and settings.enable_telemetry:
                    if persistence_store is None:
                        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
                    await persistence_store.log_exception(
                        error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                        category=ae.category.value if hasattr(ae.category, "value") else str(ae.category),
                        severity=ae.severity.value if hasattr(ae.severity, "value") else str(ae.severity),
                        message=ae.message,
                        user_message=ae.user_message,
                        session_id=log_ctx.get("session_id"),
                        request_id=log_ctx.get("request_id"),
                        node_name="call_claims_tool",
                        metadata={"validation_type": "no_intent"},
                        user_id=log_ctx.get("user_id")
                    )
            except Exception:
                pass  # Never break tool flow if exception logging fails
            # Log state snapshot for debugging
            result_dict = {"tool_results": tool_result.dict()}
            if isinstance(state, dict):
                await log_state_snapshot(state, "call_claims_tool", result_dict)
            return result_dict
            
        # Validate entities exist
        if not entities:
            logger.warning("⚠️ Validation failed: No entities provided")
            ae = create_validation_error(
                message="No entities provided in state",
                field="entities",
                value=None
            )
            tool_result = ToolResult(
                tool_name="claims_api",
                status=ToolExecutionStatus.FAILURE,
                data={},
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                api_endpoint=None,
                http_status_code=None,
                is_retryable=False
            )
            # Log to exceptions collection for debugging (additive)
            try:
                if log_ctx and settings.enable_telemetry:
                    if persistence_store is None:
                        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
                    await persistence_store.log_exception(
                        error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                        category=ae.category.value if hasattr(ae.category, "value") else str(ae.category),
                        severity=ae.severity.value if hasattr(ae.severity, "value") else str(ae.severity),
                        message=ae.message,
                        user_message=ae.user_message,
                        session_id=log_ctx.get("session_id"),
                        request_id=log_ctx.get("request_id"),
                        node_name="call_claims_tool",
                        metadata={"validation_type": "no_entities"},
                        user_id=log_ctx.get("user_id")
                    )
            except Exception:
                pass  # Never break tool flow if exception logging fails
            # Log state snapshot for debugging
            result_dict = {"tool_results": tool_result.dict()}
            if isinstance(state, dict):
                await log_state_snapshot(state, "call_claims_tool", result_dict)
            return result_dict
        
        entities = normalize_entities(entities) 

        logger.info(f"📋 Intent: {intent}")
        logger.info(f"📋 Entities: {entities}")

        # ========================================================================
        # CLAIM NUMBER + SEQUENCE VALIDATION
        # Sequence number is MANDATORY when claim number is provided.
        # The list API (byclaimnumber) is NOT called individually anymore.
        # ========================================================================
        has_claim_number = "claimNumber" in entities or "claimId" in entities
        has_sequence = "claimSequence" in entities
        
        # If claim number is provided but sequence is missing, return error
        if has_claim_number and not has_sequence:
            claim_num = entities.get("claimNumber") or entities.get("claimId")
            
            # ========================================================================
            # 🚨 CLARIFICATION DECISION LOG - Detailed diagnostic (GCP + MongoDB)
            # ========================================================================
            history_preview = "(no history)"
            if isinstance(state, dict):
                conv_history = state.get("conversation_history", [])
                user_messages = [m.get("content", "") for m in conv_history if m.get("role") == "user"]
                if user_messages:
                    history_preview = " | ".join(user_messages)
                    if len(history_preview) > 200:
                        history_preview = "..." + history_preview[-200:]
            
            # Log to GCP (stdout only - no MongoDB to avoid latency)
            logger.warning(
                f"\n{'='*70}\n"
                f"🚨 CLARIFICATION_DECISION: SEQUENCE_MISSING\n"
                f"{'='*70}\n"
                f"SESSION: {log_ctx.get('session_id', 'unknown')[:20] if log_ctx else 'unknown'}\n"
                f"INTENT: {intent}\n"
                f"\n"
                f"WHAT WE HAVE:\n"
                f"  ✅ Claim Number: {claim_num}\n"
                f"  ❌ Sequence: NOT FOUND\n"
                f"\n"
                f"WHERE WE LOOKED:\n"
                f"  • Current message entities: {list(current_entities.keys()) if current_entities else '(empty)'}\n"
                f"  • History extracted_slots: {list(extracted_slots.keys()) if extracted_slots else '(empty)'}\n"
                f"\n"
                f"HISTORY PREVIEW: {history_preview}\n"
                f"{'='*70}"
            )
            
            error_msg = f"Sequence number is required for claim number '{claim_num}'. Please provide the sequence number along with the claim number."
            
            ae = create_validation_error(
                message=error_msg,
                field="claimSequence",
                value=None
            )
            
            tool_result = ToolResult(
                tool_name="claims_api",
                status=ToolExecutionStatus.FAILURE,
                data={
                    "error_type": "sequence_required",
                    "claim_number": claim_num,
                    "message": error_msg
                },
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                api_endpoint=None,
                http_status_code=None,
                is_retryable=False
            )
            # Log to exceptions collection for debugging (additive - existing GCP log above preserved)
            try:
                if log_ctx and settings.enable_telemetry:
                    if persistence_store is None:
                        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
                    await persistence_store.log_exception(
                        error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                        category=ae.category.value if hasattr(ae.category, "value") else str(ae.category),
                        severity=ae.severity.value if hasattr(ae.severity, "value") else str(ae.severity),
                        message=ae.message,
                        user_message=ae.user_message,
                        session_id=log_ctx.get("session_id"),
                        request_id=log_ctx.get("request_id"),
                        node_name="call_claims_tool",
                        metadata={"validation_type": "sequence_required", "claim_number": claim_num},
                        user_id=log_ctx.get("user_id")
                    )
            except Exception:
                pass  # Never break tool flow if exception logging fails
            # Log state snapshot for debugging
            result_dict = {"tool_results": tool_result.dict()}
            if isinstance(state, dict):
                await log_state_snapshot(state, "call_claims_tool", result_dict)
            return result_dict
        
        # ========================================================================
        # ENRICHED CLAIM DETAILS FLOW (claim number + sequence)
        # Both claimNumber/claimId AND claimSequence are required
        # ========================================================================
        if has_claim_number and has_sequence:
            # Get the claim number (might be under claimNumber or claimId key)
            claim_num = entities.get("claimNumber") or entities.get("claimId")
            claim_seq = entities.get("claimSequence")
            
            logger.info("🔄 HYBRID ROUTING: Both claimNumber/claimId + claimSequence present → forcing enriched details flow")
            logger.info(f"   📍 ClaimNumber: {claim_num} (type: {type(claim_num).__name__}, repr: {repr(claim_num)})")
            logger.info(f"   📍 ClaimSequence: {claim_seq} (type: {type(claim_seq).__name__}, repr: {repr(claim_seq)})")
            logger.info(f"   📍 Intent: {intent} (not used - entities are decisive)")
            
            # Both entities present, proceed with enriched flow
            # (No validation needed - already confirmed above)
            
            # Log enriched flow attempt
            if log_ctx and persistence_store:
                await persistence_store.log_audit(
                    session_id=log_ctx.get("session_id"),
                    request_id=log_ctx.get("request_id"),
                    user_id=log_ctx.get("user_id"),
                    node_name="call_claims_tool",
                    event_type="enriched_flow_attempt",
                    data={
                        "intent": intent,
                        "flow_type": "claim_details_enriched",
                        "entities": entities
                    }
                )
            
            # Execute enriched claim details flow (WITH CACHING)
            # Cache key format: session:{sessionId}:api_cache:{userId}_{claimNumber}_{sequenceNumber}
            start = time.time()
            try:
                logger.info(f"🌐 Fetching enriched claim details (cache-aware)")
                
                # Extract user_id from user_info for cache key
                # This ensures each user's cached data is isolated (security)
                user_id_for_cache = user_info.get("user_id", "anonymous") if isinstance(user_info, dict) else "anonymous"
                
                # Extract session_id from state for cache key
                # This ensures cache is scoped to the current conversation session
                session_id_for_cache = state.get("session_id", "unknown") if isinstance(state, dict) else "unknown"
                
                # Use cache-aware wrapper instead of direct API call
                # This checks cache first, calls API on miss, and caches the result
                # Returns tuple: (result, from_cache) where from_cache is True if cache hit
                result, from_cache = await get_cached_or_fetch_enriched_details(
                    claim_number=claim_num,        # The 15-digit claim number user asked about
                    claim_sequence=claim_seq,      # The 3-digit sequence number user specified
                    user_id=user_id_for_cache,     # For cache key (security isolation)
                    session_id=session_id_for_cache,  # For cache key (session isolation)
                    auth_token=auth_token,         # For API authentication (if cache miss)
                    fetch_function=combine_claim_details_and_list  # The actual API function to call on miss
                )
                elapsed_ms = (time.time() - start) * 1000.0
                
                # Log whether we got a cache hit or made an API call
                # This helps with monitoring cache effectiveness
                if from_cache:
                    logger.info(f"🎯 CACHE HIT! Retrieved in {elapsed_ms:.2f}ms (saved external API call)")
                else:
                    logger.info(f"✅ API call succeeded in {elapsed_ms:.2f}ms (response now cached for follow-ups)")
                
                # Log success (include cache status in audit log)
                if log_ctx and persistence_store:
                    await persistence_store.log_audit(
                        session_id=log_ctx.get("session_id"),
                        request_id=log_ctx.get("request_id"),
                        user_id=log_ctx.get("user_id"),
                        node_name="call_claims_tool",
                        event_type="enriched_flow_success",
                        data={
                            "flow_type": "claim_details_enriched",
                            "execution_time_ms": elapsed_ms,
                            "status": "success",
                            "from_cache": from_cache  # Track cache hits for analytics
                        }
                    )
                
                # Return raw unmasked data - PII masking will be handled by safety layer
                # This follows separation of concerns: tools return data, safety layer handles security
                logger.info(f"✅ Returning raw enriched claim details (PII masking will be done by safety layer)")
                
                tool_result = ToolResult(
                    tool_name="claim_details_enriched",
                    status=ToolExecutionStatus.SUCCESS,
                    data=result,  # Raw unmasked data
                    error_message=None,
                    error_code=None,
                    agent_error=None,
                    execution_time_ms=elapsed_ms,
                    api_endpoint="/myclaims/claims/v1/details",
                    http_status_code=200,
                    is_retryable=False,
                    metadata={}
                )
                # Log state snapshot for debugging (success path - no exception logging needed)
                result_dict = {"tool_results": tool_result.dict()}
                if isinstance(state, dict):
                    await log_state_snapshot(state, "call_claims_tool", result_dict)
                return result_dict
                
            except Exception as exc:
                elapsed_ms = (time.time() - start) * 1000.0
                logger.error(f"❌ Enriched claim details flow failed after {elapsed_ms:.2f}ms: {str(exc)}")
                
                # Map exception to AgentError
                ae = to_agent_error(exc, node="claim_details_enriched")
                
                # Log failure (existing log_audit preserved, log_exception added alongside)
                if log_ctx and persistence_store:
                    await persistence_store.log_audit(
                        session_id=log_ctx.get("session_id"),
                        request_id=log_ctx.get("request_id"),
                        user_id=log_ctx.get("user_id"),
                        node_name="call_claims_tool",
                        event_type="enriched_flow_failure",
                        data={
                            "flow_type": "claim_details_enriched",
                            "execution_time_ms": elapsed_ms,
                            "error": str(exc),
                            "status": "failure"
                        }
                    )
                    # Also log to exceptions collection for debugging
                    try:
                        await persistence_store.log_exception(
                            error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                            category=ae.category.value if hasattr(ae.category, "value") else str(ae.category),
                            severity=ae.severity.value if hasattr(ae.severity, "value") else str(ae.severity),
                            message=ae.message,
                            user_message=ae.user_message,
                            session_id=log_ctx.get("session_id"),
                            request_id=log_ctx.get("request_id"),
                            node_name="call_claims_tool",
                            stacktrace=traceback.format_exc(),
                            metadata={"flow": "enriched_details_failure", "execution_time_ms": elapsed_ms},
                            user_id=log_ctx.get("user_id")
                        )
                    except Exception:
                        pass  # Never break tool flow if exception logging fails
                
                # Extract full API error response details from exception if available
                # This ensures response agent gets complete error info for all error types
                api_error_details = {}
                if hasattr(exc, 'details') and isinstance(exc.details, dict):
                    exc_details = exc.details
                    error_type = exc_details.get("error_type", "api_error")
                    
                    # Always include common fields
                    api_error_details = {
                        "error_type": error_type,
                        "status_code": exc_details.get("status"),
                        "claim_number": exc_details.get("claimNumber"),
                        "claim_sequence": exc_details.get("claimSequence"),
                        "message": exc_details.get("message") or str(exc),
                        # Include full API response for debugging/logging
                        "response_body": exc_details.get("response_body"),
                        "response_json": exc_details.get("response_json"),
                        "api_url": exc_details.get("url"),
                        "api_name": exc_details.get("api")
                    }
                    
                    # Add error-type specific fields
                    if error_type == "sequence_not_found":
                        api_error_details["invalid_sequence"] = exc_details.get("invalidSequence")
                        # Note: availableSequences intentionally not passed to response agent
                    
                    logger.info(f"📋 Error details for response agent: error_type={error_type}, status={exc_details.get('status')}")
                
                tool_result = ToolResult(
                    tool_name="claim_details_enriched",
                    status=ToolExecutionStatus.FAILURE,
                    data=api_error_details,  # Put API error details in data field
                    error_message=ae.message,
                    error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                    api_endpoint="/myclaims/claims/v1/details",
                    http_status_code=ae.metadata.get("status_code") if isinstance(ae.metadata, dict) else None,
                    is_retryable=ae.is_retryable
                )
                # Log state snapshot for debugging
                result_dict = {"tool_results": tool_result.dict()}
                if isinstance(state, dict):
                    await log_state_snapshot(state, "call_claims_tool", result_dict)
                return result_dict
        
        # ========================================================================
        # STANDARD FLOW: All other intents use normal matching logic
        # ========================================================================
        # 1) match API (Option A: treat no match as validation error)
        logger.debug(f"🔍 Matching API for intent: {intent}")
        api = match_api(intent, entities)

        if not api:
            logger.warning(f"❌ No matching API found for intent '{intent}' with entities: {list(entities.keys())}")
            ae = create_validation_error(
                message="No matching API found for given intent/entities",
                field="entities",
                value=entities
            )

            tool_result = ToolResult(
                tool_name="claims_api",
                status=ToolExecutionStatus.FAILURE,
                data={},
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                api_endpoint=None,
                http_status_code=None,
                is_retryable=False
            )
            # Log to exceptions collection for debugging (additive)
            try:
                if log_ctx and settings.enable_telemetry:
                    if persistence_store is None:
                        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
                    await persistence_store.log_exception(
                        error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                        category=ae.category.value if hasattr(ae.category, "value") else str(ae.category),
                        severity=ae.severity.value if hasattr(ae.severity, "value") else str(ae.severity),
                        message=ae.message,
                        user_message=ae.user_message,
                        session_id=log_ctx.get("session_id"),
                        request_id=log_ctx.get("request_id"),
                        node_name="call_claims_tool",
                        metadata={"validation_type": "no_matching_api", "intent": intent},
                        user_id=log_ctx.get("user_id")
                    )
            except Exception:
                pass  # Never break tool flow if exception logging fails
            # Log state snapshot for debugging
            result_dict = {"tool_results": tool_result.dict()}
            if isinstance(state, dict):
                await log_state_snapshot(state, "call_claims_tool", result_dict)
            return result_dict

        logger.info(f"✅ [MATCH] Selected API → {getattr(api, 'name', '<unknown>')}")

        # 2) build request body
        logger.debug(f"🔨 Building request body for API: {api.name}")
        try:
            body = api.body_template(entities)
            logger.debug(f"✅ Request body built successfully")
        except Exception as e:
            logger.error(f"❌ Failed to build request body: {e}")
            ae = create_internal_error(
                error_message=f"Failed building request body: {e}",
                node_name=getattr(api, "name", None)
            )
            tool_result = ToolResult(
                tool_name=getattr(api, "name", "claims_api"),
                status=ToolExecutionStatus.FAILURE,
                data={},
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                api_endpoint=getattr(api, "full_url", None),
                is_retryable=False
            )
            # Log to exceptions collection for debugging (additive)
            try:
                if log_ctx and settings.enable_telemetry:
                    if persistence_store is None:
                        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
                    await persistence_store.log_exception(
                        error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                        category=ae.category.value if hasattr(ae.category, "value") else str(ae.category),
                        severity=ae.severity.value if hasattr(ae.severity, "value") else str(ae.severity),
                        message=ae.message,
                        user_message=ae.user_message,
                        session_id=log_ctx.get("session_id"),
                        request_id=log_ctx.get("request_id"),
                        node_name="call_claims_tool",
                        stacktrace=traceback.format_exc(),
                        metadata={"validation_type": "body_template_failure", "api_name": getattr(api, "name", "unknown")},
                        user_id=log_ctx.get("user_id")
                    )
            except Exception:
                pass  # Never break tool flow if exception logging fails
            # Log state snapshot for debugging
            result_dict = {"tool_results": tool_result.dict()}
            if isinstance(state, dict):
                await log_state_snapshot(state, "call_claims_tool", result_dict)
            return result_dict

        # Log API call attempt for audit trail
        if log_ctx and persistence_store:
            await persistence_store.log_audit(
                session_id=log_ctx.get("session_id"),
                request_id=log_ctx.get("request_id"),
                user_id=log_ctx.get("user_id"),
                node_name="call_claims_tool",
                event_type="api_call_attempt",
                data={
                    "intent": intent,
                    "api_name": getattr(api, "name", "unknown"),
                    "api_endpoint": getattr(api, "full_url", None)
                }
            )

        # 3) call external API (async call wrapped with retry decorator) - Thread-safe
        logger.info(f"🌐 Calling external API: {api.name} → {getattr(api, 'full_url', 'N/A')}")
        start = time.time()
        try:
            result = await call_external_api(api, body, auth_token)
            elapsed_ms = (time.time() - start) * 1000.0
            
            logger.info(f"✅ API call succeeded in {elapsed_ms:.2f}ms")

            # Log successful API call
            if log_ctx and persistence_store:
                await persistence_store.log_audit(
                    session_id=log_ctx.get("session_id"),
                    request_id=log_ctx.get("request_id"),
                    user_id=log_ctx.get("user_id"),
                    node_name="call_claims_tool",
                    event_type="api_call_success",
                    data={
                        "api_name": getattr(api, "name", "unknown"),
                        "execution_time_ms": elapsed_ms,
                        "status": "success"
                    }
                )

            # Return raw unmasked data - PII masking will be handled by safety layer
            # This follows separation of concerns: tools return data, safety layer handles security
            logger.info(f"✅ Returning raw API response (PII masking will be done by safety layer)")

            tool_result = ToolResult(
                tool_name=getattr(api, "name", "claims_api"),
                status=ToolExecutionStatus.SUCCESS,
                data=result,  # Raw unmasked data
                error_message=None,
                error_code=None,
                agent_error=None,
                execution_time_ms=elapsed_ms,
                api_endpoint=getattr(api, "full_url", None),
                http_status_code=200,
                is_retryable=False,
                metadata={}
            )
            result_dict = {"tool_results": tool_result.dict()}
            if isinstance(state, dict):
                await log_state_snapshot(state, "call_claims_tool", result_dict)
            return result_dict

        except Exception as exc:
            elapsed_ms = (time.time() - start) * 1000.0
            logger.error(f"❌ API call failed after {elapsed_ms:.2f}ms: {str(exc)}")
            
            # map exception -> AgentError and fill ToolResult
            ae = to_agent_error(exc, node=getattr(api, "name", None))

            # Log failed API call (existing log_audit preserved, log_exception added alongside)
            if log_ctx and persistence_store:
                await persistence_store.log_audit(
                    session_id=log_ctx.get("session_id"),
                    request_id=log_ctx.get("request_id"),
                    user_id=log_ctx.get("user_id"),
                    node_name="call_claims_tool",
                    event_type="api_call_failure",
                    data={
                        "api_name": getattr(api, "name", "unknown"),
                        "execution_time_ms": elapsed_ms,
                        "error": str(exc),
                        "status": "failure"
                    }
                )
                # Also log to exceptions collection for debugging
                try:
                    await persistence_store.log_exception(
                        error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                        category=ae.category.value if hasattr(ae.category, "value") else str(ae.category),
                        severity=ae.severity.value if hasattr(ae.severity, "value") else str(ae.severity),
                        message=ae.message,
                        user_message=ae.user_message,
                        session_id=log_ctx.get("session_id"),
                        request_id=log_ctx.get("request_id"),
                        node_name="call_claims_tool",
                        stacktrace=traceback.format_exc(),
                        metadata={"flow": "standard_api_failure", "api_name": getattr(api, "name", "unknown"), "execution_time_ms": elapsed_ms},
                        user_id=log_ctx.get("user_id")
                    )
                except Exception:
                    pass  # Never break tool flow if exception logging fails

            # Extract full API error response details from exception if available
            api_error_details = {}
            if hasattr(exc, 'details') and isinstance(exc.details, dict):
                api_error_details = {
                    "error_type": "api_error",
                    "status_code": exc.details.get("status"),
                    "response_body": exc.details.get("response_body"),
                    "response_json": exc.details.get("response_json"),
                    "api_url": exc.details.get("url"),
                    "api_name": exc.details.get("api")
                }
            
            tool_result = ToolResult(
                tool_name=getattr(api, "name", "claims_api"),
                status=ToolExecutionStatus.FAILURE,
                data=api_error_details,  # Put API error details in data field
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                api_endpoint=getattr(api, "full_url", None),
                http_status_code=ae.metadata.get("status_code") if isinstance(ae.metadata, dict) else None,
                is_retryable=ae.is_retryable
            )
            result_dict = {"tool_results": tool_result.dict()}
            if isinstance(state, dict):
                await log_state_snapshot(state, "call_claims_tool", result_dict)
            return result_dict
    except Exception as e:
        # Outer exception handler for unexpected errors
        tb = traceback.format_exc()
        logger.error(f"🚨 Unexpected exception in call_claims_tool_node: {e}\n{tb}")
        
        # Log exception to database (fix: initialize persistence_store if not yet created)
        if log_ctx:
            try:
                if persistence_store is None:
                    persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
                await persistence_store.log_exception(
                    session_id=log_ctx.get("session_id"),
                    request_id=log_ctx.get("request_id"),
                    user_id=log_ctx.get("user_id"),
                    node_name="call_claims_tool",
                    error_code="E9001",
                    category="system",
                    severity="critical",
                    message=f"Unexpected error in claims API tool: {str(e)}",
                    stacktrace=tb
                )
            except Exception as log_error:
                logger.error(f"Failed to log exception: {log_error}")
        
        # Return error result
        ae = create_internal_error(
            error_message=str(e),
            stacktrace=tb,
            node_name="call_claims_tool"
        )
        tool_result = ToolResult(
            tool_name="claims_api",
            status=ToolExecutionStatus.FAILURE,
            data={},
            error_message=ae.message,
            error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
            agent_error=ae,
            api_endpoint=None,
            http_status_code=None,
            is_retryable=False
        )
        # Log state snapshot for debugging
        result_dict = {"tool_results": tool_result.dict()}
        if isinstance(state, dict):
            await log_state_snapshot(state, "call_claims_tool", result_dict)
        return result_dict



# ============================================================================
# ENTITY NORMALIZATION
# ============================================================================
ENTITY_MAP = {
    "claim_number": "claimNumber",
    "member_id": "memberId",
    "prescription_number": "prescriptionNumber",
    "medication_name": "medicationName",
    "date_from": "dateFrom",
    "date_to": "dateTo",
    "claim_sequence": "claimSequence",
    "sequence_number": "claimSequence",  # LLM judge uses this name
    "claim_id": "claimNumber",       # Map to claimNumber (same as claim_number) - team's fix
    "claim_ids": "claimNumber",      # Map to claimNumber (same as claim_number)
    "claimId": "claimNumber",        # Normalize camelCase claimId to claimNumber
    "claim_sequences": "claimSequence",  # Map plural sequences
}

def _is_masked_token(value: Any) -> bool:
    """
    Check if a value is a masked PII token like [CLAIM_ID_ABC123] or [CLAIM_ID_253152631273000].
    
    These tokens are created by the PII protection layer and should not be sent to the API.
    In multi-turn conversations, the LLM may extract masked tokens from conversation history
    instead of the actual values.
    """
    import re
    if isinstance(value, str):
        # Match patterns like:
        # - [CLAIM_ID_ABC123] (hex hash)
        # - [CLAIM_ID_253152631273000] (actual claim number embedded)
        # - [PERSON_DEF456]
        # - Any token starting with [ and containing ENTITY_TYPE_ pattern
        return bool(re.match(r'^\[[A-Z_]+_[A-Za-z0-9]+\]$', value))
    return False

# --- Helper Function to handle Pydantic model extraction ---
def normalize_entities(entities_obj) -> Dict[str, Any]:
    """
    1. Merges top-level entities with raw_entities from the Pydantic model or dict.
    2. Normalizes all keys from snake_case to target API's camelCase format.
    3. Also accepts already-normalized camelCase keys (passthrough).
    4. Filters out masked PII tokens (e.g., [CLAIM_ID_ABC123]) that should not be sent to API.
    """
    
    # Handle both Pydantic models and plain dicts
    if isinstance(entities_obj, dict):
        # Already a dict from state
        all_entities = {k: v for k, v in entities_obj.items() if v is not None and k != 'raw_entities'}
        # Merge raw_entities if present
        if 'raw_entities' in entities_obj and isinstance(entities_obj['raw_entities'], dict):
            all_entities.update(entities_obj['raw_entities'])
    else:
        # Pydantic model - use .model_dump(exclude_none=True) for Pydantic v2 (or .dict() for v1)
        # Extract the main fields from the Pydantic model, excluding raw_entities itself initially
        all_entities = entities_obj.model_dump(exclude_none=True, exclude={'raw_entities'})
        
        # Merge the contents of raw_entities dictionary into the main dictionary
        all_entities.update(entities_obj.raw_entities)
    
    # CRITICAL: Filter out masked PII tokens that LLM may have extracted from conversation history
    # These look like [CLAIM_ID_ABC123] and should not be sent to the API
    filtered_entities = {}
    for k, v in all_entities.items():
        if _is_masked_token(v):
            logger.warning(f"⚠️ Filtering out masked token from entities: {k}={v}")
            continue
        # FIX: Also filter masked tokens inside list values (e.g., claim_ids: ["[CLAIM_ID_...]"])
        if isinstance(v, list):
            filtered_list = [item for item in v if not _is_masked_token(item)]
            if not filtered_list:
                logger.warning(f"⚠️ Filtering out list with all masked tokens: {k}={v}")
                continue
            if len(filtered_list) != len(v):
                logger.info(f"🔧 Removed masked tokens from list: {k}: {v} -> {filtered_list}")
                v = filtered_list
        filtered_entities[k] = v
    all_entities = filtered_entities
    
    # Create reverse mapping for camelCase -> camelCase (already normalized)
    # This allows function to accept both snake_case AND camelCase inputs
    REVERSE_ENTITY_MAP = {v: v for v in ENTITY_MAP.values()}
    COMBINED_MAP = {**ENTITY_MAP, **REVERSE_ENTITY_MAP}
    
    # Normalize keys to the target API format (camelCase)
    # Only include fields that are in ENTITY_MAP (filter out extra fields like person_names)
    normalized_entities = {}
    for k, v in all_entities.items():
        # Skip fields not in COMBINED_MAP (handles both snake_case and camelCase)
        if k not in COMBINED_MAP:
            continue
            
        # Use the mapped name
        target_key = COMBINED_MAP.get(k)
        
        # Handle list values - take first element for singular API parameters
        if isinstance(v, list) and len(v) > 0:
            # For plural entity names mapping to singular API params, use first value
            if k in ['claim_ids', 'member_ids', 'claim_sequences'] and target_key in ['claimId', 'memberId', 'claimSequence']:
                v = v[0]  # Take first claim/member ID or sequence
        
        normalized_entities[target_key] = v
    
    # FIX: Normalize claimNumber - strip "claim" prefix if present
    # PII masking preserves full text (e.g., "claim 233211748898001") for entity detection
    # But API expects only numeric ID, so we normalize here at API call time
    if "claimNumber" in normalized_entities:
        import re
        value = normalized_entities["claimNumber"]
        # Handle list values (claim_ids maps to claimNumber but may still be a list)
        if isinstance(value, list) and len(value) > 0:
            value = value[0]
        if isinstance(value, str):
            numeric_match = re.search(r'\d+$', value)
            if numeric_match:
                normalized_entities["claimNumber"] = numeric_match.group(0)
        
    return normalized_entities

# ============================================================================
# API MATCHING LOGIC (Hybrid: Entities + Intent)
# ============================================================================
def match_api(intent: str, entities: Dict[str, Any]):
    """
    Hybrid API matcher using ENTITIES (primary) + INTENT (tiebreaker):
    
    Step 1: Filter APIs by required entities (must all be present)
    Step 2: Score remaining APIs by intent keyword matches
    Step 3: Add specificity bonus (more required entities = higher score)
    
    This ensures:
    - Entities determine WHICH APIs are eligible
    - Intent determines WHICH eligible API to use (if multiple match)
    
    Example:
      Query: "claim status 242563401456001"
      - Entities: claimId="242563401456001"
      - Intent: "claim_status" (keywords: status, check, track...)
      - Both APIs require claimId, but list API has "status" keyword
      - Result: list API wins due to intent match
    """
    registry = get_api_repository()
    intent_lower = (intent or "").lower()

    matched_api = None
    matched_api_score = -1
    eligible_apis = []

    logger.debug(f"🔍 API Matching started - Intent: '{intent}', Entities: {list(entities.keys())}")
    
    for api in registry:
        api_name = getattr(api, 'name', 'unknown')
        required_entities = getattr(api, "required_entities", [])
        
        # STEP 1: Entity filter - required entities must all be present
        missing_entities = [req for req in required_entities if req not in entities]
        if missing_entities:
            logger.debug(f"   ❌ {api_name}: Missing required entities {missing_entities}")
            continue
        
        logger.debug(f"   ✅ {api_name}: Has all required entities {required_entities}")
        eligible_apis.append(api)

        # STEP 2: Intent scoring - count keyword matches
        intent_keywords = getattr(api, "intent_keywords", [])
        matched_keywords = [kw for kw in intent_keywords if kw in intent_lower]
        intent_score = len(matched_keywords)
        
        # STEP 3: Specificity bonus - more required entities = more specific
        specificity_score = len(required_entities)
        
        total_score = intent_score + specificity_score
        
        logger.debug(f"      📊 Scoring: intent_score={intent_score} (matched: {matched_keywords}), specificity={specificity_score}, total={total_score}")

        if total_score > matched_api_score:
            matched_api_score = total_score
            matched_api = api
            logger.debug(f"      🏆 New leader: {api_name} (score: {total_score})")
    
    # Log final decision
    if matched_api:
        logger.info(f"✅ [MATCH] Selected API: {getattr(matched_api, 'name', 'None')} (score: {matched_api_score}, eligible: {len(eligible_apis)})")
    else:
        logger.warning(f"❌ [NO MATCH] No eligible APIs found for intent '{intent}' with entities {list(entities.keys())}")

    return matched_api


# ============================================================================
# EXTERNAL API CALL WITH RETRY (ASYNC)
# ============================================================================
@async_retry(attempts=3, retry_on=(ExternalAPIError, ToolTimeoutError))
async def call_external_api(api, body: Dict[str, Any], auth_token: str = "") -> Dict[str, Any]:
    """
    Async HTTP call wrapper using httpx. Raises ExternalAPIError / ToolTimeoutError on issues.
    Returned value is parsed JSON.
    
    Args:
        api: API configuration object
        body: Request body as dict
        auth_token: Authorization token (passed per-request for thread-safety)
        
    Returns:
        Parsed JSON response as dict
    """
    url = getattr(api, "full_url", getattr(api, "endpoint", None))
    method = getattr(api, "method", "POST").upper()
    
    # Generate proper correlation ID in CVS format
    correlation_id = f"CVS-{uuid.uuid4()}"
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-correlation-id": correlation_id,
        "x-clientRefId": correlation_id,
        "Authorization": auth_token  # Thread-safe: passed as parameter
    }

    # Log request details for debugging (auth token masked to first 20 chars)
    _token_preview = (auth_token or "")[:20] + "..." if auth_token else "(empty)"
    logger.info(f"🌐 Making API request:")
    logger.info(f"   URL: {url}")
    logger.info(f"   Method: {method}")
    logger.info(f"   Auth: {_token_preview}")
    logger.info(f"   Body: {body}")
    
    # Use shared httpx.AsyncClient for connection pooling and memory efficiency
    # This prevents creating new clients for each request which can cause memory leaks
    client = await _get_shared_http_client()
    try:
        resp = await client.request(method, url, headers=headers, json=body)
        logger.info(f"   Response Status: {resp.status_code}")
        if resp.status_code != 200:
            logger.error(f"   Response Body: {resp.text[:500]}")
        resp.raise_for_status()
        
        # parse JSON or raise
        try:
            return resp.json()
        except (ValueError, TypeError) as e:
            # non-json response
            raise ExternalAPIError("Non-JSON response", details={"status": resp.status_code, "raw": resp.text, "api": getattr(api, "name", url)}, retriable=False)
            
    except httpx.TimeoutException as e:
        # timeout -> retriable
        raise ToolTimeoutError(str(e), details={"api": getattr(api, "name", url)}, retriable=True)
    except httpx.HTTPStatusError as e:
        # HTTP error (4xx, 5xx)
        status = e.response.status_code
        
        # Capture full error response body for detailed error reporting
        error_response_body = None
        error_response_json = None
        if e.response is not None:
            logger.error(f"❌ API Error Response:")
            logger.error(f"   Status: {e.response.status_code}")
            logger.error(f"   Response Text: {e.response.text[:500]}")
            
            # Store the full response body
            error_response_body = e.response.text
            
            # Try to parse as JSON for structured error details
            try:
                error_response_json = e.response.json()
            except:
                pass  # Not JSON, use raw text
        
        # Build comprehensive error details
        error_details = {
            "status": status,
            "api": getattr(api, "name", url),
            "response_body": error_response_body,
            "response_json": error_response_json,
            "url": url
        }
        
        retriable = status is None or (500 <= status < 600)
        raise ExternalAPIError(str(e), details=error_details, retriable=retriable)
    except httpx.RequestError as e:
        # Connection error or other request error
        logger.error(f"❌ Request Error: {e}")
        error_details = {
            "status": None,
            "api": getattr(api, "name", url),
            "response_body": None,
            "response_json": None,
            "url": url
        }
        # Connection errors are retriable
        raise ExternalAPIError(str(e), details=error_details, retriable=True)


# ============================================================================
# HELPER METHODS FOR ENRICHED CLAIM DETAILS
# ============================================================================
def get_first_non_none(dicts_and_keys):
    """
    Safely get first non-None value from list of (dict, key) tuples.
    This handles zero values correctly (unlike 'or' chain which treats 0 as falsy).
    
    Args:
        dicts_and_keys: List of (dictionary, key) tuples to check
        
    Returns:
        First non-None value found, or None if all are None/missing
    """
    for d, key in dicts_and_keys:
        if d and key in d and d[key] is not None:
            return d[key]
    return None


async def get_claim_details(claimNumber: str, claimSequence: str, auth_token: str = "") -> Dict[str, Any]:
    """
    Fetch claim details by claim number and sequence (async).
    Uses existing call_external_api() with retry logic.
    Falls back to JSON file on failure.
    
    Args:
        claimNumber: The claim number
        claimSequence: The claim sequence
        auth_token: Authorization token (passed per-request for thread-safety)
        
    Returns:
        Single claim details object as returned by API or fallback
    """
    logger.debug(f"📞 Fetching claim details for claimNumber={claimNumber}, claimSequence={claimSequence}")
    
    try:
        # Get the API definition from repository
        registry = get_api_repository()
        api = next((a for a in registry if getattr(a, 'name', '') == 'get_claim_details'), None)
        
        if not api:
            raise ValueError("get_claim_details API not found in repository")
        
        # Build request body using the API's body template
        entities = {
            "claimNumber": claimNumber,
            "claimSequence": claimSequence
        }
        body = api.body_template(entities)
        
        # Call external API with retry logic - pass auth_token for thread-safety
        result = await call_external_api(api, body, auth_token)
        logger.debug(f"✅ Successfully fetched claim details")
        return result
    
    except ExternalAPIError as e:
        # Check if error is retriable (server down: 5xx errors, network issues)
        if e.retriable and settings.enable_api_fallback:
            # Server is down AND fallback is enabled - use mock data for testing
            logger.warning(f"⚠️ Fallback: Claim Details API server error (retriable) - Using mock data for testing")
            logger.warning(f"   Error: {str(e)}")
            logger.warning(f"   💡 Fallback is ENABLED (enable_api_fallback=True) - Set to False in production")
            fallback_data = get_fallback_details(claimNumber, claimSequence)
            return fallback_data
        elif e.retriable:
            # Server is down but fallback is disabled - return error
            logger.error(f"❌ Claim Details API server error - Fallback disabled, returning error")
            logger.error(f"   Error: {str(e)}")
            raise
        else:
            # Client error (400, 401, 404, etc.) - NEVER use fallback
            # These indicate actual issues: invalid claim number, unauthorized, not found, etc.
            logger.error(f"❌ Claim Details API client error (non-retriable) - NO FALLBACK")
            logger.error(f"   Error: {str(e)}")
            logger.error(f"   📌 Client errors (4xx) indicate real issues like:")
            logger.error(f"      - 400: Invalid request/claim number format")
            logger.error(f"      - 401: Unauthorized access")
            logger.error(f"      - 404: Claim not found")
            logger.error(f"   ⚠️ Returning actual error to user (NO FAKE DATA)")
            raise
    
    except ToolTimeoutError as e:
        # Network timeout
        if settings.enable_api_fallback:
            # Fallback enabled - use mock data
            logger.warning(f"⚠️ Fallback: Claim Details API timeout - Using mock data for testing")
            logger.warning(f"   Error: {str(e)}")
            fallback_data = get_fallback_details(claimNumber, claimSequence)
            return fallback_data
        else:
            # Fallback disabled - return error
            logger.error(f"❌ Claim Details API timeout - Fallback disabled, returning error")
            raise
    
    except Exception as e:
        # Unexpected errors - always re-raise
        logger.error(f"❌ Unexpected error in get_claim_details: {str(e)}")
        raise


async def get_claim_list(claimId: str, claimSequence: str = "1", auth_token: str = "") -> Dict[str, Any]:
    """
    Fetch claim list by claim ID (async).
    Uses existing call_external_api() with retry logic.
    Falls back to dynamic generated data on failure.
    
    Args:
        claimId: The claim ID to search for
        claimSequence: The claim sequence (optional, used for fallback data generation)
        auth_token: Authorization token (passed per-request for thread-safety)
        
    Returns:
        List of claims as returned by API or fallback
    """
    logger.debug(f"📞 Fetching claim list for claimId={claimId} (type: {type(claimId).__name__}, repr: {repr(claimId)})")
    
    # CRITICAL: Ensure claimId is a string, not a list
    if isinstance(claimId, list):
        logger.warning(f"⚠️ claimId is a list {claimId}, converting to string")
        claimId = str(claimId[0]) if claimId else ""
        logger.info(f"   ✅ Converted to: claimId={claimId}")
    
    try:
        # Get the API definition from repository
        registry = get_api_repository()
        api = next((a for a in registry if getattr(a, 'name', '') == 'get_claim_list'), None)
        
        if not api:
            raise ValueError("get_claim_list API not found in repository")
        
        # Build request body using the API's body template
        entities = {
            "claimId": claimId
        }
        body = api.body_template(entities)
        
        # Call external API with retry logic - pass auth_token for thread-safety
        result = await call_external_api(api, body, auth_token)
        logger.debug(f"✅ Successfully fetched claim list")
        return result
    
    except ExternalAPIError as e:
        # Check if error is retriable (server down: 5xx errors, network issues)
        if e.retriable and settings.enable_api_fallback:
            # Server is down AND fallback is enabled - use mock data for testing
            logger.warning(f"⚠️ Fallback: Claim List API server error (retriable) - Using mock data for testing")
            logger.warning(f"   Error: {str(e)}")
            logger.warning(f"   💡 Fallback is ENABLED (enable_api_fallback=True) - Set to False in production")
            fallback_data = get_fallback_list(claimId, claimSequence)
            return fallback_data
        elif e.retriable:
            # Server is down but fallback is disabled - return error
            logger.error(f"❌ Claim List API server error - Fallback disabled, returning error")
            logger.error(f"   Error: {str(e)}")
            raise
        else:
            # Client error (400, 401, 404, etc.) - NEVER use fallback
            # These indicate actual issues: invalid claim ID, unauthorized, not found, etc.
            logger.error(f"❌ Claim List API client error (non-retriable) - NO FALLBACK")
            logger.error(f"   Error: {str(e)}")
            logger.error(f"   📌 Client errors (4xx) indicate real issues like:")
            logger.error(f"      - 400: Invalid request/claim ID format")
            logger.error(f"      - 401: Unauthorized access")
            logger.error(f"      - 404: Claim not found")
            logger.error(f"   ⚠️ Returning actual error to user (NO FAKE DATA)")
            raise
    
    except ToolTimeoutError as e:
        # Network timeout
        if settings.enable_api_fallback:
            # Fallback enabled - use mock data
            logger.warning(f"⚠️ Fallback: Claim List API timeout - Using mock data for testing")
            logger.warning(f"   Error: {str(e)}")
            fallback_data = get_fallback_list(claimId, claimSequence)
            return fallback_data
        else:
            # Fallback disabled - return error
            logger.error(f"❌ Claim List API timeout - Fallback disabled, returning error")
            raise
    
    except Exception as e:
        # Unexpected errors - always re-raise
        logger.error(f"❌ Unexpected error in get_claim_list: {str(e)}")
        raise


async def combine_claim_details_and_list(claimNumber: str, claimSequence: str, auth_token: str = "") -> Dict[str, Any]:
    """
    Enriched claim details: combines claim details with filtered claim list (async).
    
    OPTIMIZED Flow with concurrent API calls:
        Step 1: Fetch claim list and claim details CONCURRENTLY (for better performance)
        Step 2: Validate sequence exists in list (FAIL EARLY if not found)
        Step 3: Merge filtered list record with claim details
        
    Args:
        claimNumber: The claim number
        claimSequence: The claim sequence
        auth_token: Authorization token (passed per-request for thread-safety)
        
    Returns:
        Merged enriched claim details object
        
    Raises:
        ExternalAPIError: With full error details for response agent
    """
    logger.info(f"🔄 Starting enriched claim details flow (concurrent API calls)")
    logger.info(f"   📌 Input: claimNumber={claimNumber} (type: {type(claimNumber).__name__})")
    logger.info(f"   📌 Input: claimSequence={claimSequence} (type: {type(claimSequence).__name__})")
    
    # CRITICAL: Ensure parameters are strings, not lists (defensive programming)
    if isinstance(claimNumber, list):
        logger.warning(f"⚠️ claimNumber is a list {claimNumber}, converting to string")
        claimNumber = str(claimNumber[0]) if claimNumber else ""
        logger.info(f"   ✅ Converted to: claimNumber={claimNumber}")
    
    if isinstance(claimSequence, list):
        logger.warning(f"⚠️ claimSequence is a list {claimSequence}, converting to string")
        claimSequence = str(claimSequence[0]) if claimSequence else "1"
        logger.info(f"   ✅ Converted to: claimSequence={claimSequence}")
    
    # Step 1: Fetch claim list and claim details CONCURRENTLY for better performance
    logger.info(f"Step 1: Fetching claim list and claim details concurrently for claimId={claimNumber}")
    
    # Initialize variables for error handling
    claim_details_error = None
    
    try:
        # Execute both API calls concurrently using asyncio.gather
        claim_list_response, claim_details = await asyncio.gather(
            get_claim_list(claimNumber, claimSequence, auth_token),
            get_claim_details(claimNumber, claimSequence, auth_token),
            return_exceptions=True
        )
        
        # Check if claim_list_response is an exception
        if isinstance(claim_list_response, Exception):
            logger.error(f"   ❌ Claim list API failed: {claim_list_response}")
            if isinstance(claim_list_response, ExternalAPIError):
                raise claim_list_response
            raise ExternalAPIError(
                f"Failed to fetch claim list for claim number '{claimNumber}': {str(claim_list_response)}",
                details={
                    "claimNumber": claimNumber,
                    "claimSequence": claimSequence,
                    "status": 500,
                    "error_type": "list_api_error",
                    "api": "get_claim_list",
                    "message": f"Unable to retrieve claim information for claim number '{claimNumber}'."
                },
                retriable=True
            )
        
        logger.info(f"   ✅ Claim list fetched successfully")
        logger.debug(f"   Response type: {type(claim_list_response).__name__}")
        if isinstance(claim_list_response, dict):
            logger.debug(f"   Response keys: {list(claim_list_response.keys())}")
        
        # Check if claim_details is an exception
        if isinstance(claim_details, Exception):
            logger.error(f"   ❌ Claim details API failed: {claim_details}")
            # Even if details fails, we can still validate sequence from list
            # But we'll need to handle this case
            if isinstance(claim_details, ExternalAPIError):
                # Store the exception to raise later after sequence validation
                claim_details_error = claim_details
            else:
                claim_details_error = ExternalAPIError(
                    f"Failed to fetch claim details for claim number '{claimNumber}', sequence '{claimSequence}': {str(claim_details)}",
                    details={
                        "claimNumber": claimNumber,
                        "claimSequence": claimSequence,
                        "status": 500,
                        "error_type": "details_api_error",
                        "api": "get_claim_details",
                        "message": f"An unexpected error occurred while retrieving claim details."
                    },
                    retriable=True
                )
            claim_details = None  # Set to None to indicate failure
        else:
            logger.info(f"   ✅ Claim details fetched successfully")
    except ExternalAPIError as e:
        # List API failed - re-raise with enhanced details
        logger.error(f"   ❌ Claim list API failed: {e}")
        raise ExternalAPIError(
            f"Failed to fetch claim list for claim number '{claimNumber}': {str(e)}",
            details={
                "claimNumber": claimNumber,
                "claimSequence": claimSequence,
                "status": e.details.get("status") if hasattr(e, 'details') and e.details else 500,
                "error_type": "list_api_error",
                "response_body": e.details.get("response_body") if hasattr(e, 'details') and e.details else None,
                "response_json": e.details.get("response_json") if hasattr(e, 'details') and e.details else None,
                "url": e.details.get("url") if hasattr(e, 'details') and e.details else None,
                "api": "get_claim_list",
                "message": f"Unable to retrieve claim information for claim number '{claimNumber}'."
            },
            retriable=e.retriable if hasattr(e, 'retriable') else True
        )
    except Exception as e:
        logger.error(f"   ❌ Unexpected error fetching claim list: {e}")
        raise ExternalAPIError(
            f"Failed to fetch claim list for claim number '{claimNumber}': {str(e)}",
            details={
                "claimNumber": claimNumber,
                "claimSequence": claimSequence,
                "status": 500,
                "error_type": "list_api_error",
                "api": "get_claim_list",
                "message": f"An unexpected error occurred while retrieving claim information."
            },
            retriable=True
        )
    
    # Step 2: Extract list and validate sequence IMMEDIATELY (fail fast)
    logger.info("Step 2: Extracting claim list and validating sequence...")
    
    # Note: claim_details may be None if the concurrent call failed, but we validate sequence first
    
    claim_list = []
    if isinstance(claim_list_response, dict):
        logger.debug(f"   claim_list_response keys: {list(claim_list_response.keys())}")
        for key in ["claims", "data", "claimsList", "claimList"]:
            if key in claim_list_response and claim_list_response[key] is not None:
                claim_list = claim_list_response[key]
                logger.info(f"   ✅ Found claim list in key '{key}' with {len(claim_list) if isinstance(claim_list, list) else 'N/A'} items")
                break
    elif isinstance(claim_list_response, list):
        claim_list = claim_list_response
        logger.info(f"   ✅ Claim list response is a direct list with {len(claim_list)} claims")
    
    # Check if claim list is empty (claim number might not exist)
    if not claim_list or len(claim_list) == 0:
        logger.error(f"   ❌ Claim list is empty - claim number may not exist!")
        raise ExternalAPIError(
            f"No claims found for claim number '{claimNumber}'. Please verify the claim number and try again.",
            details={
                "claimNumber": claimNumber,
                "claimSequence": claimSequence,
                "status": 404,
                "error_type": "claim_not_found",
                "message": f"No claims found for claim number '{claimNumber}'. Please verify the claim number and try again."
            },
            retriable=False
        )
    
    # DEBUG: Log first claim structure
    if claim_list and len(claim_list) > 0 and isinstance(claim_list[0], dict):
        logger.info(f"📋 First claim keys: {list(claim_list[0].keys())}")
    
    # Find matching claim by sequence
    matched_claim = None
    available_sequences = []
    
    for claim in claim_list:
        if isinstance(claim, dict):
            claim_info = claim.get("claimInformation", {})
            claim_primary = claim.get("primary", {})
            
            claim_num = get_first_non_none([
                (claim_primary, "number"),
                (claim, "claimNumber"),
                (claim, "claimId"),
                (claim, "claim_number"),
                (claim, "claim_id"),
                (claim_info, "claimNumber"),
                (claim_info, "claimId")
            ])
            
            claim_seq = get_first_non_none([
                (claim_primary, "sequence"),
                (claim, "claimSequence"),
                (claim, "sequenceNumber"),
                (claim, "claimSequenceNumber"),
                (claim, "sequence"),
                (claim_info, "claimSequence"),
                (claim_info, "claimSequenceNumber"),
                (claim_info, "sequenceNumber")
            ])
            
            # Track available sequences for error message
            if claim_seq is not None:
                available_sequences.append(str(claim_seq))
            
            logger.debug(f"   Comparing: claim_seq={claim_seq} vs target={claimSequence}")
            
            if str(claim_num) == str(claimNumber) and str(claim_seq) == str(1000-int(claimSequence)):
                matched_claim = claim
                logger.info(f"   ✅ Found matching sequence: claimNumber={claim_num}, claimSequence={claim_seq}")
                break
    
    # FAIL EARLY if sequence not found (before calling details API)
    if matched_claim is None:
        logger.error(f"   ❌ Sequence '{claimSequence}' NOT FOUND in claim list!")
        logger.debug(f"   📋 Available sequences (for debugging): {available_sequences}")
        
        error_msg = f"Sequence number '{claimSequence}' does not exist for claim number '{claimNumber}'. Please verify the sequence number and try again."
        
        raise ExternalAPIError(
            error_msg,
            details={
                "claimNumber": claimNumber,
                "invalidSequence": claimSequence,
                "status": 404,
                "error_type": "sequence_not_found",
                "message": error_msg
            },
            retriable=False
        )
    
    logger.info(f"   ✅ Sequence validated")
    
    # Step 3: Check if claim details fetch failed (from concurrent call)
    if claim_details is None:
        if claim_details_error is not None:
            logger.error(f"   ❌ Claim details API failed during concurrent fetch")
            raise claim_details_error
        else:
            # This shouldn't happen, but handle it gracefully
            logger.error(f"   ❌ Claim details is None but no error was captured")
            raise ExternalAPIError(
                f"Failed to fetch claim details for claim number '{claimNumber}', sequence '{claimSequence}'",
                details={
                    "claimNumber": claimNumber,
                    "claimSequence": claimSequence,
                    "status": 500,
                    "error_type": "details_api_error",
                    "api": "get_claim_details",
                    "message": f"An unexpected error occurred while retrieving claim details."
                },
                retriable=True
            )
    
    # Step 4: Merge results
    logger.info("Step 4: Merging claim details with list data")
    enriched_details = claim_details.copy() if isinstance(claim_details, dict) else {}

    if matched_claim and isinstance(matched_claim, dict):
        enriched_details["list_data"] = matched_claim
    else:
        enriched_details["list_data"] = matched_claim

    # Step 5: Deep sequence transformation (999 → 001 throughout entire response)
    # This replaces the old single-field workaround that only fixed list_data.primary.sequence.
    # The transformer converts ALL matching sequence fields in the entire response
    # so the LLM never sees internal 999-form values.
    logger.info("Step 5: Transforming all internal sequences to user-facing format")
    enriched_details = transform_sequences_to_user_format(
        data=enriched_details,
        user_sequence=claimSequence,
        claim_number=claimNumber,
    )

    logger.info(f"✅ Successfully enriched claim details with list_data + sequence transform")
    logger.info(f"   📌 User requested: claimNumber={claimNumber}, claimSequence={claimSequence}")

    return enriched_details