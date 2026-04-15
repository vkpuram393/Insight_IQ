"""
Orchestrator Node - Entry point for all user inputs

This is a PURE PROCESSING NODE (no LLM calls).
It normalizes user input before passing to the workflow.

Key Responsibilities:
- Input normalization (6 steps: lowercase, space collapse, Unicode, zero-width removal, punctuation removal, strip)
- Preserve original text for logging/display purposes
- Add orchestration metadata for tracking
- Ensure JSON-serializable data throughout
- Graceful error handling with fallback to original input

Compatibility:
- Output must be compatible with safety_precheck_node input expectations
- Must return 'text' as a string (safety_precheck expects state["text"].lower())
- All additional data stored in 'metadata' dict to avoid schema conflicts
"""

import re
import unicodedata
import time
import traceback
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from state.schema import AgentState
from config.config import settings
from core.logger import get_logger
from core.logging_context import extract_logging_context, log_state_snapshot
from core.node_models import OrchestratorResult, create_orchestrator_result
from core.errors.models import (
    AgentError,
    create_orchestrator_empty_input_error,
    create_orchestrator_invalid_type_error,
    create_orchestrator_normalization_error
)
from persistence import EventType, PersistenceStoreFactory
from utils.serialization import to_dict

logger = get_logger(__name__)


# ============================================================================
# NORMALIZATION HELPER FUNCTIONS
# ============================================================================

def _normalize_text(raw_text: str) -> Tuple[str, List[str]]:
    """
    Apply normalization pipeline to input text.
    
    Normalization Steps (in order):
    1. Lowercase conversion
    2. Collapse multiple spaces
    3. Unicode normalization (NFD)
    4. Remove zero-width characters
    5. Remove punctuation (if enabled)
    6. Strip whitespace (final step to remove trailing spaces)
    
    Args:
        raw_text: Raw user input text
    
    Returns:
        Tuple of (normalized_text, normalization_steps_applied)
    """
    normalized = raw_text
    steps = []
    
    # Step 1: Convert to lowercase
    normalized = normalized.lower()
    steps.append("lowercase")
    logger.debug(f"📝 Normalization: After lowercase → {len(normalized)} chars")
    
    # Step 2: Collapse multiple spaces to single space
    # This also handles tabs, newlines, and other whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    steps.append("collapse_spaces")
    logger.debug(f"📝 Normalization: After space collapse → {len(normalized)} chars")
    
    # Step 3: Unicode normalization (NFD - Canonical Decomposition)
    # W3C/Unicode Consortium standard for consistent Unicode representation
    # Example: café (with é as single char) → café (with e + combining accent)
    normalized = unicodedata.normalize('NFD', normalized)
    steps.append("unicode_nfd")
    logger.debug(f"📝 Normalization: After Unicode NFD → {len(normalized)} chars")
    
    # Step 4: Remove zero-width characters (security measure)
    # These invisible characters can be used for spoofing attacks
    # Includes: zero-width space (\u200b), zero-width non-joiner (\u200c),
    #           zero-width joiner (\u200d), and zero-width no-break space (\ufeff)
    normalized = re.sub(r'[\u200b-\u200d\ufeff]', '', normalized)
    steps.append("remove_zero_width")
    logger.debug(f"📝 Normalization: After zero-width removal → {len(normalized)} chars")
    
    # Step 5: Remove punctuation (configurable)
    # Removes all punctuation marks while preserving alphanumeric characters and spaces
    # This improves cache hit rates and provides cleaner text for downstream processing
    if settings.remove_punctuation_in_normalization:
        normalized = re.sub(r'[^\w\s]', '', normalized)
        steps.append("remove_punctuation")
        logger.debug(f"📝 Normalization: After punctuation removal → {len(normalized)} chars")
    
    # Step 6: Strip whitespace (FINAL STEP - ensures no trailing spaces remain)
    normalized = normalized.strip()
    steps.append("strip_whitespace")
    logger.debug(f"📝 Normalization: After final strip → {len(normalized)} chars")
    
    return normalized, steps


# ============================================================================
# ENTITY ENRICHMENT HELPERS (UI Context for First Query of Session)
# ============================================================================

def _detect_entities_in_text(normalized_text: str) -> Dict[str, Any]:
    """
    Detect if user's normalized text already contains a claim ID and/or sequence number.
    
    Uses patterns CONSISTENT with downstream EntityExtractor to ensure:
    - If we detect it here, EntityExtractor will also find it (no false negatives downstream)
    - If we miss it here, EntityExtractor also misses it (enrichment rescues the case)
    
    Detection Methods for Sequence:
    - Method 1: Keyword-preceded (sequence/seq/line + optional space + 3 digits)
      Consistent with EntityExtractor context_pattern: r'(?:sequence|seq|line)\\s*(\\d{3})\\b'
    - Method 2: Adjacent to claim ID (15 digits + whitespace + 3 digits)
      Handles "201250305894000 027" pattern without explicit keyword
    
    Args:
        normalized_text: Text after normalization pipeline
    
    Returns:
        Dict with 'has_claim_id' (bool), 'has_sequence' (bool), 'detected_claim_ids' (list)
    """
    # Claim ID: exactly 15 digits, word-bounded
    # Consistent with EntityExtractor pattern: r'\\b(CLM\\d{3,10}|\\d{15})\\b'
    claim_id_matches = re.findall(r'\b(\d{15})\b', normalized_text)
    has_claim_id = len(claim_id_matches) > 0
    
    # Sequence detection (two methods, matching EntityExtractor patterns):
    # Method 1: Keyword-preceded (sequence/seq/line + optional space + 3 digits)
    has_keyword_sequence = bool(re.search(
        r'(?:sequence|seq|line)\s*\d{3}\b', normalized_text
    ))
    # Method 2: Adjacent to claim ID (15 digits + whitespace + 3 digits)
    has_adjacent_sequence = bool(re.search(
        r'\b\d{15}\s+\d{3}\b', normalized_text
    ))
    has_sequence = has_keyword_sequence or has_adjacent_sequence
    
    return {
        'has_claim_id': has_claim_id,
        'has_sequence': has_sequence,
        'detected_claim_ids': claim_id_matches
    }


def _enrich_first_query_entities(
    normalized_text: str,
    ui_claim_id: Optional[str],
    ui_claim_sequence: Optional[str],
    log_ctx: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Enrich the first query of a session with missing entities from the UI payload.
    
    When a user clicks "Ask AI" from the myClaims UI, the frontend sends the claim_id
    and claim_sequence from the UI screen in the request payload (first query only).
    If the user's text is missing those entities, we append them so that downstream
    entity extraction handles everything automatically.
    
    ONLY appends entities that are:
    1. Missing from the user's text (detected via _detect_entities_in_text)
    2. Available and valid in the UI payload
    
    Safety guarantees:
    - Never raises exceptions (returns original text on any error)
    - Never modifies text if UI payload is absent/invalid
    - Logs all enrichment decisions for observability
    
    Args:
        normalized_text: Text after normalization pipeline
        ui_claim_id: Claim ID from UI payload (or None)
        ui_claim_sequence: Claim sequence from UI payload (or None)
        log_ctx: Logging context (session_id, user_id, request_id)
        
    Returns:
        Tuple of (enriched_text, enrichment_metadata_dict)
    """
    enrichment_meta = {
        "entity_enrichment_applied": False,
        "ui_claim_id_available": bool(ui_claim_id),
        "ui_claim_sequence_available": bool(ui_claim_sequence),
    }
    
    try:
        # Validate UI claim_id format (defense-in-depth against malformed UI data)
        if ui_claim_id and not re.match(r'^\d{15}$', ui_claim_id):
            logger.warning(
                f"⚠️ Orchestrator enrichment: Invalid UI claim_id format: '{ui_claim_id}' "
                f"- skipping enrichment for claim_id | session={log_ctx.get('session_id', 'N/A')}"
            )
            ui_claim_id = None
        
        # Validate and normalize UI claim_sequence format (ensure 3-digit zero-padded)
        if ui_claim_sequence:
            ui_claim_sequence = ui_claim_sequence.strip().zfill(3)
            if not re.match(r'^\d{3}$', ui_claim_sequence):
                logger.warning(
                    f"⚠️ Orchestrator enrichment: Invalid UI claim_sequence format: '{ui_claim_sequence}' "
                    f"- skipping enrichment for sequence | session={log_ctx.get('session_id', 'N/A')}"
                )
                ui_claim_sequence = None
        
        # If no valid UI entities remain after validation, skip enrichment
        if not ui_claim_id and not ui_claim_sequence:
            logger.debug(f"🔍 Orchestrator enrichment: No valid UI entities to enrich with")
            return normalized_text, enrichment_meta
        
        # Detect what the user already provided in their text
        detection = _detect_entities_in_text(normalized_text)
        has_claim_id = detection['has_claim_id']
        has_sequence = detection['has_sequence']
        
        enrichment_meta["user_has_claim_id"] = has_claim_id
        enrichment_meta["user_has_sequence"] = has_sequence
        
        # Build list of missing entities to append
        append_parts = []
        
        # Append claim ID if user didn't provide one
        if not has_claim_id and ui_claim_id:
            append_parts.append(f"claim {ui_claim_id}")
            enrichment_meta["appended_claim_id"] = ui_claim_id
        
        # Append sequence if user didn't provide one
        if not has_sequence and ui_claim_sequence:
            append_parts.append(f"sequence {ui_claim_sequence}")
            enrichment_meta["appended_sequence"] = ui_claim_sequence
        
        # Apply enrichment if there are missing entities to append
        if append_parts:
            enriched_text = f"{normalized_text} for {' and '.join(append_parts)}"
            enrichment_meta["entity_enrichment_applied"] = True
            enrichment_meta["appended_text"] = f" for {' and '.join(append_parts)}"
            logger.info(
                f"✅ Orchestrator enrichment: Appended [{', '.join(append_parts)}] to first query "
                f"| session={log_ctx.get('session_id', 'N/A')}"
            )
            return enriched_text, enrichment_meta
        else:
            logger.debug(
                f"🔍 Orchestrator enrichment: No enrichment needed - user provided all entities "
                f"| session={log_ctx.get('session_id', 'N/A')}"
            )
            return normalized_text, enrichment_meta
    
    except Exception as e:
        # Safety net: enrichment failure should NEVER break the flow
        logger.warning(
            f"⚠️ Orchestrator enrichment: Unexpected error during enrichment: {str(e)} "
            f"- continuing with original text | session={log_ctx.get('session_id', 'N/A')}"
        )
        enrichment_meta["enrichment_error"] = str(e)
        return normalized_text, enrichment_meta


async def _handle_empty_input(
    state: AgentState,
    log_ctx: Dict[str, Any],
    original_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle empty input error case.
    
    Args:
        state: Current agent state
        log_ctx: Logging context (session_id, user_id, request_id)
        original_metadata: Original metadata from state
    
    Returns:
        State update dict with error information
    """
    logger.warning(f"⚠️ Orchestrator: Empty input received from session: {log_ctx['session_id']}")
    
    # Create structured error
    error = create_orchestrator_empty_input_error(
        session_id=log_ctx["session_id"],
        user_id=log_ctx["user_id"]
    )
    
    # Log to persistence store (consistent with safety.py pattern)
    persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
    await persistence_store.log_exception(
        error_code=error.error_code.value,
        category=error.category.value,
        severity=error.severity.value,
        message=error.message,
        user_message=error.user_message,
        session_id=log_ctx["session_id"],
        request_id=log_ctx["request_id"],
        node_name="orchestrator",
        stacktrace=error.stacktrace,
        metadata=error.metadata,
        user_id=log_ctx["user_id"]
    )
    
    # Create orchestrator result with error
    result = create_orchestrator_result(
        normalized_text="",
        original_text="",
        normalization_applied=False,
        original_length=0,
        normalized_length=0,
        chars_removed=0,
        error=to_dict(error),
        processing_time_ms=0.0
    )
    
    result_dict = {
        "text": "",  # CRITICAL: safety_precheck expects string
        "uuid": state["uuid"],
        "error": error.error_code.value,
        "metadata": {
            **original_metadata,
            "orchestrator_metadata": to_dict(result),
            "original_text": ""
        }
    }
    await log_state_snapshot(state, "orchestrator", result_dict)
    return result_dict


async def _handle_invalid_type(
    raw_text: Any,
    state: AgentState,
    log_ctx: Dict[str, Any]
) -> str:
    """
    Handle invalid input type case.
    
    Logs error to persistence store and converts input to string for graceful continuation.
    
    Args:
        raw_text: The invalid input
        state: Current agent state
        log_ctx: Logging context (session_id, user_id, request_id)
    
    Returns:
        String representation of the input
    """
    logger.warning(f"⚠️ Orchestrator: Invalid input type: {type(raw_text)} from session: {log_ctx['session_id']}")
    
    # Create structured error
    error = create_orchestrator_invalid_type_error(
        input_type=type(raw_text),
        session_id=log_ctx["session_id"],
        user_id=log_ctx["user_id"]
    )
    
    # Log to persistence store (consistent with safety.py pattern)
    persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
    await persistence_store.log_exception(
        error_code=error.error_code.value,
        category=error.category.value,
        severity=error.severity.value,
        message=error.message,
        user_message=error.user_message,
        session_id=log_ctx["session_id"],
        request_id=log_ctx["request_id"],
        node_name="orchestrator",
        stacktrace=error.stacktrace,
        metadata=error.metadata,
        user_id=log_ctx["user_id"]
    )
    
    # Attempt conversion (graceful fallback) - error logged but continue processing
    return str(raw_text)


async def _handle_normalization_error(
    exception: Exception,
    raw_text: Any,
    state: AgentState,
    log_ctx: Dict[str, Any],
    original_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle normalization failure with graceful fallback.
    
    Args:
        exception: The exception that occurred
        raw_text: The original input text
        state: Current agent state
        log_ctx: Logging context (session_id, user_id, request_id)
        original_metadata: Original metadata from state
    
    Returns:
        State update dict with error information and fallback text
    """
    logger.error(f"❌ Orchestrator FAILURE: {str(exception)}")
    logger.error(f"🔍 Orchestrator ERROR DETAILS: Session: {log_ctx['session_id']}, Type: {type(exception).__name__}")
    logger.warning(f"⚠️ Orchestrator: Falling back to original input")
    
    # Get stacktrace for debugging
    tb = traceback.format_exc()
    logger.debug(f"🔍 Orchestrator STACK TRACE:\n{tb}")
    
    # Create structured error
    error = create_orchestrator_normalization_error(
        exception=exception,
        session_id=log_ctx["session_id"],
        user_id=log_ctx["user_id"],
        stacktrace=tb
    )
    
    # Log to persistence store (consistent with safety.py pattern)
    persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
    await persistence_store.log_exception(
        error_code=error.error_code.value,
        category=error.category.value,
        severity=error.severity.value,
        message=error.message,
        user_message=error.user_message,
        session_id=log_ctx["session_id"],
        request_id=log_ctx["request_id"],
        node_name="orchestrator",
        stacktrace=error.stacktrace,
        metadata=error.metadata,
        user_id=log_ctx["user_id"]
    )
    
    # Graceful fallback
    fallback_text = str(raw_text) if raw_text else ""
    
    # Create orchestrator result with error
    result = create_orchestrator_result(
        normalized_text=fallback_text,
        original_text=fallback_text,
        normalization_applied=False,
        original_length=len(fallback_text) if fallback_text else 0,
        normalized_length=len(fallback_text) if fallback_text else 0,
        chars_removed=0,
        error=to_dict(error)
    )
    
    result_dict = {
        "text": fallback_text,
        "uuid": state["uuid"],
        "error": error.error_code.value,
        "metadata": {
            **original_metadata,
            "orchestrator_metadata": to_dict(result),
            "original_text": fallback_text,
            # Reset stale enrichment checkpoint data (consistent with success path fix)
            "enriched_text": None,
            "enrichment_metadata": None
        }
    }
    await log_state_snapshot(state, "orchestrator", result_dict)
    return result_dict


# ============================================================================
# MAIN ORCHESTRATOR NODE
# ============================================================================

async def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """
    Orchestrator Node - Entry point and input normalizer
    
    This is a PURE NODE - NO LLM calls. Pure text processing only.
    
    INPUT (reads from state):
        - text: str (user's raw input) - REQUIRED
        - session_id: str - REQUIRED
        - user_info: dict (optional user metadata)
        - metadata: dict (optional, will be merged)
    
    OUTPUT (writes to state):
        - text: str (normalized text) - REQUIRED for safety_precheck_node
        - metadata: dict (merged with orchestrator_metadata and original_text)
    
    NORMALIZATION OPERATIONS:
        1. Convert to lowercase
        2. Collapse multiple spaces to single space
        3. Unicode normalization (NFD - Canonical Decomposition)
        4. Remove zero-width characters (security measure)
        5. Remove punctuation (configurable via settings.remove_punctuation_in_normalization)
        6. Strip whitespace (FINAL - ensures no trailing spaces)
    
    ERROR HANDLING:
        - Invalid input → Returns original with error metadata
        - Normalization failure → Falls back to original text with error log
        - Always returns valid state for downstream nodes
    
    COMPATIBILITY GUARANTEE:
        - Output is 100% compatible with safety_precheck_node
        - Returns 'text' as string (required by safety node)
        - Preserves all required state fields
        - No breaking changes to existing schema
    """
    
    logger.info("🎯 Node: Orchestrator")
    
    # Generate UUID first (before extracting logging context)
    # This ensures log_ctx["request_id"] will have the UUID
    request_uuid = state.get("uuid") or str(uuid.uuid4())
    state["uuid"] = request_uuid
    
    # Extract logging context (consistent with other nodes)
    node_name = "orchestrator"
    log_ctx = extract_logging_context(state)
    
    # Start timing for processing_time_ms calculation
    start_time = time.perf_counter()
    
    # Extract current state values
    raw_text = state.get("text", "")
    original_metadata = state.get("metadata", {})
    
    # CRITICAL FIX: Clear stale enrichment data from checkpoint-persisted metadata.
    # When LangGraph checkpoints state between requests (same session_id/thread_id),
    # merge_metadata reducer preserves ALL keys from previous requests' checkpoints.
    # Without this reset, enriched_text from the first request's UI entity enrichment
    # persists indefinitely via **original_metadata spread, causing update_memory_node
    # to save the wrong (stale first-request) user message for every subsequent request.
    # Popping here ensures both success and error paths benefit from cleanup.
    original_metadata.pop("enriched_text", None)
    original_metadata.pop("enrichment_metadata", None)
    
    try:
        # ============================================================
        # INPUT VALIDATION
        # ============================================================
        
        if not raw_text:
            return await _handle_empty_input(state, log_ctx, original_metadata)
        
        if not isinstance(raw_text, str):
            raw_text = await _handle_invalid_type(raw_text, state, log_ctx)
        
        original_text = raw_text
        original_length = len(original_text)
        
        logger.info(f"📥 Orchestrator: Received {original_length} chars from session: {log_ctx['session_id']}")
        logger.debug(f"🔍 Orchestrator: Original text preview: '{original_text[:50]}...'")
        
        # ============================================================
        # NORMALIZATION PIPELINE
        # ============================================================
        
        normalized, normalization_steps = _normalize_text(raw_text)
        
        # Track normalization-only stats before enrichment
        post_normalization_length = len(normalized)
        chars_removed = original_length - post_normalization_length
        
        # ============================================================
        # ENTITY ENRICHMENT (first query of session only)
        # ============================================================
        # If UI sends claim_id/claim_sequence in payload and user's
        # text is missing those entities, append them so downstream
        # entity extraction handles everything automatically.
        # This block is naturally scoped to first query only because
        # the frontend sends these fields only for the first query.
        # ============================================================
        
        enrichment_metadata = {}
        ui_claim_id = (state.get("user_info") or {}).get("claim_id")
        ui_claim_sequence = (state.get("user_info") or {}).get("claim_sequence")
        
        if ui_claim_id or ui_claim_sequence:
            normalized, enrichment_metadata = _enrich_first_query_entities(
                normalized, ui_claim_id, ui_claim_sequence, log_ctx
            )
            if enrichment_metadata.get("entity_enrichment_applied"):
                normalization_steps.append("entity_enrichment")
        
        normalized_length = len(normalized)
        
        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"✅ Orchestrator: Input normalized successfully")
        logger.info(f"📊 Orchestrator: Original {original_length} → Normalized {post_normalization_length} chars ({chars_removed} removed)")
        if enrichment_metadata.get("entity_enrichment_applied"):
            logger.info(f"📊 Orchestrator: After enrichment → {normalized_length} chars (+{normalized_length - post_normalization_length} added)")
        logger.info(f"⏱️ Orchestrator: Processing time: {processing_time_ms:.2f}ms")
        
        if chars_removed > 0 or enrichment_metadata.get("entity_enrichment_applied"):
            logger.debug(f"🔍 Orchestrator: Final text preview: '{normalized[:80]}...'")
        
        # ============================================================
        # RETURN STATE UPDATE - SUCCESS PATH
        # ============================================================
        
        # Create structured orchestrator result
        result = create_orchestrator_result(
            normalized_text=normalized,
            original_text=original_text,
            normalization_applied=True,
            original_length=original_length,
            normalized_length=normalized_length,
            chars_removed=chars_removed,
            normalization_steps=normalization_steps,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Return partial state update (LangGraph will merge with existing state)
        result_dict = {
            "text": normalized,  # CRITICAL: safety_precheck_node expects this as string
            "uuid": state["uuid"],
            "metadata": {
                **original_metadata,
                "orchestrator_metadata": to_dict(result),  # Use serialization helper for consistency
                "original_text": original_text,  # Preserve raw text for logging/display
                # CRITICAL FIX: Always set enriched_text and enrichment_metadata explicitly.
                # When enrichment IS applied (first query with UI entities): store enriched text
                # so entities are available in conversation history for subsequent extraction.
                # When enrichment is NOT applied: set to None to ensure merge_metadata reducer
                # overwrites any stale checkpoint values, preventing corrupted session history.
                "enriched_text": normalized if enrichment_metadata.get("entity_enrichment_applied") else None,
                "enrichment_metadata": enrichment_metadata if enrichment_metadata else None
            }
        }
        
        # Log full AgentState snapshot after this node
        await log_state_snapshot(state, node_name, result_dict)
        
        return result_dict
        
    except Exception as e:
        # ============================================================
        # ERROR HANDLING - GRACEFUL FALLBACK WITH STRUCTURED ERRORS
        # ============================================================
        
        return await _handle_normalization_error(
            e, raw_text, state, log_ctx, original_metadata
        )

