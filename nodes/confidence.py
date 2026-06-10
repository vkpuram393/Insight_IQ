"""
Confidence Check Node - Checks confidence and routes accordingly
"""

import json
import traceback
from pathlib import Path
from typing import Dict, Any, Literal
from state.schema import AgentState
from config.config import settings
from core.logger import get_logger
from core.errors.models import (
    AgentError,
    ErrorCode,
    ErrorCategory,
    ErrorSeverity,
    create_low_confidence_error,
    create_internal_error
)
from core.logging_context import extract_logging_context, log_state_snapshot
from persistence import PersistenceStoreFactory
# ADDED: Import conversation context service for entity checking in history
from services.conversation_context import has_entities_in_history as service_has_entities

logger = get_logger(__name__)

# Confidence threshold is now in settings (config.py)

_config_cache = None

def _load_config() -> Dict[str, Any]:
    """Load domain config from JSON file"""
    global _config_cache
    config_path = Path(__file__).parent.parent / "config" / "domain_config.json"
    try:
        with open(config_path, 'r') as f:
            _config_cache = json.load(f)
        return _config_cache
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Return defaults
        return {
            "confidence_threshold": 0.7,
            "clarification_messages": {
                "low_confidence": "I'm not quite sure what you're asking. Could you rephrase your question?",
                "missing_entity_template": "Could you provide your {missing_entity}?"
            }
        }

def _quick_check_history_for_entities(conversation_history: list, intent: str) -> bool:
    """
    Quick check if conversation history contains entities needed for this intent.
    
    UPDATED: Delegates to ConversationContextService for pattern matching.
    This keeps pattern logic centralized in services/conversation_context.py
    
    Args:
        conversation_history: List of messages
        intent: The classified intent
        
    Returns:
        True if history likely contains required entities
    """
    # ADDED: Delegate to conversation context service (pattern logic centralized)
    return service_has_entities(conversation_history, intent)

def confidence_check_router(state: AgentState) -> Literal["clarification", "build_context", "llm_judge", "response_safety_pii_precheck"]:
    """Route based on confidence, entity completeness, query complexity, and intent_reclassified flag.

    UPDATED ROUTING WITH DIRECT RESPONSE PATH:
      1. response_safety_pii_precheck: Greeting/help/out_of_scope (no API needed, direct to LLM)
      2. llm_judge: Low confidence/complex/mixed-intent AND intent_reclassified == False (needs re-classification)
      3. clarification: Missing entities OR low confidence after LLM judge (template-based)
      4. build_context: High confidence + has entities (standard API flow)

    Rules (Priority Order):
      IF intent_reclassified == False (initial classifier result):
        1. If is_complex=True → llm_judge (needs expert review)
        2. If confidence < threshold → llm_judge (needs expert review)
        3. If high confidence + has entities → build_context (proceed directly)
      
      IF intent_reclassified == True (LLM judge already ran):
        1. If confidence >= threshold AND entities present → build_context (expert is confident)
        2. If missing entities OR confidence < threshold → clarification (template-based, no LLM call)
    """
    logger.info("\n" + "="*70)
    logger.info("🔀 CONFIDENCE_CHECK_ROUTER CALLED")
    logger.info("="*70)
    
    # PRIORITY CHECK: Handle embedding classifier failure
    embedding_failed = state.get("embedding_failed", False)
    intent_reclassified = state.get("intent_reclassified", False)
    if embedding_failed and not intent_reclassified:
        logger.warning("❌ Embedding classifier failed - routing to llm_judge for classification")
        logger.info("   Reason: Google/Azure embeddings unavailable, cannot classify intent semantically")
        logger.info("   Fallback: Will use LLM judge (Gemini) to classify intent from scratch")
        return "llm_judge"  # Let LLM judge classify the intent when embeddings fail
    
    config = _load_config()
    threshold = config.get("confidence_threshold", 0.7)
    confidence = state.get("confidence", 0.0)
    
    # PRIORITY CHECK: LLM fallback already ran inside the multidomain classifier.
    # If its confidence is below the LLM fallback threshold, route straight to clarification
    # instead of llm_judge (avoids a redundant second LLM call).
    # EXCEPTION: claim_history_search queries must NOT use this short-circuit.
    # The internal LLM fallback uses per-domain prompts that may lack claim-history
    # context, while llm_judge uses claim_prompt_template with conversation history
    # and is far more capable for these queries.  Skipping llm_judge for CHS domains
    # was the primary cause of correct queries reaching the clarification node.
    llm_fallback_confidence = state.get("llm_fallback_confidence")
    llm_fb_threshold = config.get(
        "llm_fallback_confidence_threshold",
        settings.LLM_FALLBACK_CONFIDENCE_THRESHOLD,
    )
    domain = state.get("domain", "")
    if (
        llm_fallback_confidence is not None
        and llm_fallback_confidence < llm_fb_threshold
        and confidence < threshold          # only clarify when ensemble is also uncertain
        and not state.get("intent_reclassified", False)
        and domain != "claim_history_search"   # let llm_judge handle CHS queries
    ):
        logger.info(
            f"⚠️ LLM fallback confidence ({llm_fallback_confidence:.2f}) "
            f"< threshold ({llm_fb_threshold}) AND ensemble confidence "
            f"({confidence:.2f}) < threshold ({threshold}) -> Clarification "
            f"(LLM already ran inside multidomain classifier)"
        )
        return "clarification"

    # ── Vague-query safety net for CHS domain ──────────────────────────────
    # Fix B (above) exempts CHS queries from the early-exit so that medium-
    # confidence queries reach llm_judge instead of clarification.  But when
    # both the internal LLM fallback AND the classifier are DEEPLY uncertain
    # (below vague_query_confidence_threshold, default 0.4) AND the current
    # query contains no extractable entities, the intent cannot be determined
    # and the system must ask the user to clarify what they need.
    #
    # Note: confidence in state may be boosted to 0.85 for CHS by the intent
    # agent (Fix A), so we use llm_fallback_confidence as the unmodified signal.
    # This guard fires only when llm_fallback ran (not None) and was truly vague.
    vague_threshold = config.get("vague_query_confidence_threshold", 0.4)
    _quick_entities = state.get("entities") or {}
    _has_any_entity = bool(_quick_entities and any(v for v in _quick_entities.values()))
    if (
        domain == "claim_history_search"
        and not state.get("intent_reclassified", False)
        and llm_fallback_confidence is not None
        and llm_fallback_confidence < vague_threshold   # deeply uncertain — not just below gate
        and not _has_any_entity                         # no claim number or other entity extracted
    ):
        logger.info(
            f"❓ Genuinely vague CHS query: llm_fallback={llm_fallback_confidence:.2f} "
            f"< vague_threshold={vague_threshold} with no extractable entities "
            f"→ Clarification (cannot determine intent or member context)"
        )
        return "clarification"

    # TEMPORARY: Check if LLM judge path is disabled (for testing)
    disable_llm_judge = settings.temporarily_disable_llm_judge_path_for_testing
    intent = state.get("intent", "")  # ✅ Get intent from state
    needs_clarification = state.get("needs_clarification", False)
    is_complex = state.get("is_complex", False)
    entities = state.get("entities") or {}
    conversation_history = state.get("conversation_history", [])
    missing_slots = state.get("missing_slots") or []  # Required entities that are missing (computed by entity extractor)
    required_entities_list = state.get("required_entities_list") or []  # Required entities for this intent (from API config)

    # INTENTS_WITHOUT_ENTITIES: Intents that don't require entity extraction
    INTENTS_WITHOUT_ENTITIES = {'out_of_scope', 'greeting', 'help'}

    # Check if we have required entities (check for required entities specifically, not just any entities)
    # FIXED: Changed from `v is not None` to `v` to properly handle empty lists ([] is falsy)
    # FIXED: Check if required entities are present, not just if ANY entities exist
    # Use missing_slots if available (most reliable), otherwise check required_entities_list
    if missing_slots:
        # Entity extractor already computed missing slots - if any are missing, we don't have required entities
        has_entities = False
        logger.debug(f"   Missing required slots: {missing_slots}")
    elif required_entities_list:
        # Check if all required entities are present (handle both list format and single value format)
        has_required_entities = True
        for req_entity in required_entities_list:
            # ONLY check claim_number and sequence - these are the ONLY two entities
            # Skip any other required_entities (like date_range, claim_numbers, etc.)
            entity_key = None
            if req_entity == "claim_number":
                entity_key = "claim_ids"  # Entity extractor uses claim_ids (list format)
            elif req_entity == "sequence":
                entity_key = "claim_sequences"  # Entity extractor uses claim_sequences (list format)
            else:
                continue  # Skip any other entity - we only care about claim_number and sequence
            
            # Check if entity exists and has values
            if entity_key not in entities or not entities[entity_key] or (isinstance(entities[entity_key], list) and len(entities[entity_key]) == 0):
                has_required_entities = False
                logger.debug(f"   Missing required entity: {req_entity} (checked as {entity_key})")
                break
        has_entities = has_required_entities
    else:
        # Fallback: Check if any entities exist (for intents without required_entities_list)
        has_entities = bool(entities and any(v for v in entities.values()))
    
    # CRITICAL: Check conversation history for entities before routing to clarification
    # This enables follow-up questions without re-asking for information user already provided
    has_entities_in_history = False
    if not has_entities and conversation_history:
        has_entities_in_history = _quick_check_history_for_entities(conversation_history, intent)
        if has_entities_in_history:
            logger.info(f"🔗 Context-aware routing: Found entities in conversation history for intent '{intent}'")
            logger.info(f"   → Will route to build_context for full extraction")
    
    # Combined check: has entities in current message OR in history
    has_entities_anywhere = has_entities or has_entities_in_history

    # PRIORITY RULE 0: Handle greeting/help/out_of_scope FIRST (before other rules)
    if intent in INTENTS_WITHOUT_ENTITIES:
        # Special case: If entities are present, this might be a mixed intent
        # Example: "Hello, please check claim 253152732536005"
        if has_entities and not intent_reclassified:
            if disable_llm_judge:
                logger.info(f"⚠️ Mixed intent detected: '{intent}' WITH entities ({list(entities.keys())}) -> Clarification (LLM judge disabled)")
                logger.info("   Reason: User combined greeting/help with claim request, but LLM judge path is disabled")
                return "clarification"
            else:
                logger.info(f"⚠️ Mixed intent detected: '{intent}' WITH entities -> LLM Judge")
                logger.info(f"   Reason: User combined greeting/help with claim request (e.g., 'Hello, check claim 123')")
                logger.info(f"   Entities found: {list(entities.keys())}")
                logger.info(f"   Action: Route to LLM judge to determine actual intent")
                return "llm_judge"
        
        # Pure greeting/help/out_of_scope without entities
        # Check if low confidence first
        if confidence < threshold and not intent_reclassified:
            if disable_llm_judge:
                logger.info(f"⚠️ Low confidence ({confidence:.2f}) for '{intent}' -> Clarification (LLM judge disabled)")
                return "clarification"
            else:
                logger.info(f"⚠️ Low confidence ({confidence:.2f}) for '{intent}' -> LLM Judge")
                logger.info("   Reason: Even though no entities needed, confidence is too low to proceed")
                return "llm_judge"
        
        # High confidence, no entities - direct to response (skip API!)
        logger.info(f"✅ Intent '{intent}' (high confidence, no entities) - routing directly to response_safety_pii_precheck")
        logger.info("   Reason: Greeting/help/out-of-scope queries don't require claim data")
        logger.info("   Action: Skip build_context and call_claims_tool entirely")
        return "response_safety_pii_precheck"

    # DECISION LOGIC: Check intent_reclassified flag first
    if not intent_reclassified:
        # Initial classifier result - can route to LLM judge if not disabled
        # RULE 1: Complex query → LLM Judge (if enabled) OR Clarification (if disabled)
        if is_complex:
            if disable_llm_judge:
                logger.info(f"🧠 Complex query detected (confidence: {confidence:.2f}) -> Clarification (LLM judge path temporarily disabled for testing)")
                logger.info("   Reason: LLM judge path is disabled - routing to clarification instead")
                return "clarification"
            else:
                logger.info(f"🧠 Complex query detected (confidence: {confidence:.2f}) -> LLM Judge")
                logger.info("   Reason: Query contains aggregations, comparisons, or multiple conditions")
                logger.info("   Flag: intent_reclassified=False (initial classification)")
                return "llm_judge"

        # RULE 2: Low confidence → LLM Judge (if enabled) OR Clarification (if disabled)
        # Skip if multidomain LLM fallback already ran inside the classifier with
        # sufficient confidence (>= llm_fb_threshold). In that case the LLM has
        # already re-evaluated the intent; routing to llm_judge would run it twice.
        if confidence < threshold:
            llm_already_confident = (
                llm_fallback_confidence is not None
                and llm_fallback_confidence >= llm_fb_threshold
            )
            if llm_already_confident:
                logger.info(
                    f"ℹ️ Low ensemble confidence ({confidence:.2f}) but multidomain "
                    f"LLM fallback already ran with sufficient confidence "
                    f"({llm_fallback_confidence:.2f} >= {llm_fb_threshold}) — "
                    f"skipping llm_judge, proceeding to entity check"
                )
                # Fall through to RULE 3 / RULE 4 (entity-based routing)
            elif disable_llm_judge:
                logger.info(f"⚠️ Low confidence ({confidence:.2f}) < {threshold} -> Clarification (LLM judge path temporarily disabled for testing)")
                logger.info("   Reason: LLM judge path is disabled - routing to clarification instead")
                return "clarification"
            else:
                logger.info(f"⚠️ Low confidence ({confidence:.2f}) < {threshold} -> LLM Judge")
                logger.info("   Reason: Uncertain intent - route to LLM Judge for re-classification")
                logger.info("   Flag: intent_reclassified=False (initial classification)")
                return "llm_judge"

        # RULE 3: No entities (current OR history) → Clarification (except for whitelisted intents)
        if not has_entities_anywhere and intent not in INTENTS_WITHOUT_ENTITIES:
            # ========================================================================
            # 🚨 CLARIFICATION DECISION LOG - Router Path (GCP only)
            # ========================================================================
            logger.warning(
                f"\n{'='*70}\n"
                f"🚨 CLARIFICATION_DECISION [confidence_check_router]: NO_ENTITIES_ANYWHERE\n"
                f"{'='*70}\n"
                f"SESSION: {state.get('session_id', 'unknown')[:20] if state.get('session_id') else 'unknown'}\n"
                f"INTENT: {intent}\n"
                f"CONFIDENCE: {confidence:.2f}\n"
                f"\n"
                f"ENTITY CHECK:\n"
                f"  • Current message entities: {has_entities}\n"
                f"  • History has entities: {has_entities_in_history}\n"
                f"  • Combined (has_entities_anywhere): {has_entities_anywhere}\n"
                f"\n"
                f"RESULT: Routing to clarification node\n"
                f"{'='*70}"
            )
            return "clarification"

        # RULE 4: High confidence + Has entities anywhere (or doesn't need them) → Build Context (direct path)
        if confidence >= threshold and (has_entities_anywhere or intent in INTENTS_WITHOUT_ENTITIES):
            if has_entities:
                logger.info(f"✅ High confidence ({confidence:.2f}) + entities present in current message -> Build Context")
            elif has_entities_in_history:
                logger.info(f"✅ High confidence ({confidence:.2f}) + entities found in history -> Build Context")
            else:
                logger.info(f"✅ Intent '{intent}' doesn't require entities -> Build Context")
            logger.info("   Reason: Initial classifier is confident enough")
            logger.info("   Flag: intent_reclassified=False (initial classification)")
            return "build_context"

    else:
        # LLM judge already ran - route based on updated intent
        
        # PRIORITY: Check if LLM judge re-classified to greeting/help/out_of_scope
        if intent in INTENTS_WITHOUT_ENTITIES:
            logger.info(f"✅ LLM Judge re-classified as '{intent}' - routing directly to response_safety_pii_precheck")
            logger.info("   Reason: Greeting/help/out-of-scope don't need API or context")
            logger.info("   Action: Skip build_context and call_claims_tool entirely")
            return "response_safety_pii_precheck"
        
        # RULE 1: High confidence + Has entities anywhere → Build Context (needs API)
        if confidence >= threshold and has_entities_anywhere:
            if has_entities:
                logger.info(f"✅ High confidence ({confidence:.2f}) + entities present in current message -> Build Context")
            elif has_entities_in_history:
                logger.info(f"✅ High confidence ({confidence:.2f}) + entities found in history -> Build Context")
            logger.info("   Reason: LLM Judge is confident enough")
            logger.info("   Flag: intent_reclassified=True (LLM judge already ran)")
            return "build_context"

        # RULE 2: Missing entities (current AND history) OR low confidence → Clarification (template-based)
        if (not has_entities_anywhere and intent not in INTENTS_WITHOUT_ENTITIES) or confidence < threshold:
            # ========================================================================
            # 🚨 CLARIFICATION DECISION LOG - Router Path after LLM Judge (GCP only)
            # ========================================================================
            reason = "NO_ENTITIES_ANYWHERE" if not has_entities_anywhere else "LOW_CONFIDENCE"
            logger.warning(
                f"\n{'='*70}\n"
                f"🚨 CLARIFICATION_DECISION [confidence_check_router]: {reason}\n"
                f"{'='*70}\n"
                f"SESSION: {state.get('session_id', 'unknown')[:20] if state.get('session_id') else 'unknown'}\n"
                f"INTENT: {intent}\n"
                f"CONFIDENCE: {confidence:.2f} (threshold: {threshold})\n"
                f"LLM_JUDGE_RAN: True\n"
                f"\n"
                f"ENTITY CHECK:\n"
                f"  • Current message entities: {has_entities}\n"
                f"  • History has entities: {has_entities_in_history}\n"
                f"  • Combined (has_entities_anywhere): {has_entities_anywhere}\n"
                f"\n"
                f"RESULT: Routing to clarification node (template-based)\n"
                f"{'='*70}"
            )
            return "clarification"

    # Fallback: Default to build_context if all else fails
    logger.info(f"✅ Default routing -> Build Context")
    logger.info("="*70 + "\n")
    return "build_context"

async def confidence_checker_node(state: AgentState) -> Dict[str, Any]:
    """
    Confidence Checker Node - Checks confidence against threshold and routes accordingly
    
    If low confidence:
        - Sets needs_clarification flag
        - Routes to clarification node (which will generate the question)
    
    If high confidence:
        - Constructs context builder input object
        - Logs to SQLite
        - Returns state to proceed to context builder
    
    Note: Entity checking is handled by the router, not this node.
    """
    node_name = "confidence_checker"
    log_ctx = extract_logging_context(state)
    
    try:
        # CRITICAL: Load conversation history BEFORE routing decisions
        # This enables the router to check history for entities
        from memory import MemoryStoreFactory
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        conversation_history = await memory_store.get_session_history(log_ctx["session_id"])
        
        # Add history to state so router can access it
        state["conversation_history"] = conversation_history
        
        logger.info(f"📚 Loaded {len(conversation_history)} messages from conversation history")
        
        # Load config
        config = _load_config()
        threshold = config.get("confidence_threshold", 0.7)
        
        # Get state values
        intent = state.get("intent")
        confidence = state.get("confidence", 0.0)
        intent_reclassified = state.get("intent_reclassified", False)
        
        logger.info(f"🔍 Confidence Check: intent={intent}, confidence={confidence:.2f}, threshold={threshold:.2f}, intent_reclassified={intent_reclassified}")
        
        # ========================================================================
        # CALCULATE MISSING SLOTS (for clarification node)
        # ONLY checks for claim_number and sequence - these are the ONLY two entities
        # ========================================================================
        required_entities_list = list(state.get("required_entities_list") or [])  # copy, don't mutate state
        entities = state.get("entities") or {}
        missing_slots = []

        # ── CHS safety: sequence is NEVER required for claim_history_search ──────
        # api_routing_config sets required_entities=["claim_number","sequence"] for
        # ALL cap_api intents.  When the classifier misidentifies a CHS query as a
        # cap_api intent (e.g. DrugLast→fill_date_info) and the is_claim_history
        # normalisation in the intent agent didn't fire, missing_slots ends up with
        # "sequence", producing a wrong clarification asking for a sequence number
        # that the claims-history pipeline never needs.  Guard against this here by
        # checking the canonical INTENT_TO_DOMAIN mapping regardless of what the
        # intent agent produced.
        _current_intent = state.get("intent", "")
        _current_domain = state.get("domain", "")
        try:
            from multidomain_intent_detection.config import INTENT_TO_DOMAIN as _I2D
            _resolved_domain = _I2D.get(_current_intent, _current_domain)
        except Exception:
            _resolved_domain = _current_domain
        if _resolved_domain == "claim_history_search" and "sequence" in required_entities_list:
            required_entities_list = [e for e in required_entities_list if e != "sequence"]
            logger.info(
                f"🔧 CHS intent '{_current_intent}': removed 'sequence' from required_entities "
                f"(claim_history_search never requires a sequence number)"
            )

        # ONLY these two entities matter - claim_number and sequence
        # All other required_entities are ignored
        ENTITY_MAP = {
            "claim_number": "claim_ids",
            "sequence": "claim_sequences"
        }

        for req_entity in required_entities_list:
            # ONLY check claim_number and sequence - skip anything else
            if req_entity not in ENTITY_MAP:
                continue

            entity_key = ENTITY_MAP[req_entity]
            if entity_key not in entities or not entities[entity_key] or (isinstance(entities[entity_key], list) and len(entities[entity_key]) == 0):
                missing_slots.append(req_entity)

        if missing_slots:
            logger.info(f"⚠️ Missing required entities: {missing_slots}")
        
        # Determine if confidence is low
        confidence_low = confidence < threshold
        
        if confidence_low:
            # Low confidence detected
            # NOTE: DON'T set needs_clarification here! The router will decide where to go:
            # - If llm_judge path: will boost confidence, then re-evaluate
            # - If clarification path: clarification_node will set the flag
            logger.info(f"⚠️ Low confidence detected ({confidence:.2f} < {threshold})")
            logger.info(f"   Router will decide: llm_judge (if enabled) or clarification")
            
            result = {
                "conversation_history": conversation_history,  # Pass history to next node
                "missing_slots": missing_slots,  # Pass missing slots to clarification node
                "metadata": {
                    **state.get("metadata", {}),
                    "confidence_low_detected": True,
                    "confidence": confidence,
                    "threshold": threshold
                }
            }
            
            # Log full AgentState snapshot after this node
            await log_state_snapshot(state, node_name, result)
            
            return result
        
        else:
            # High confidence -> proceed to context builder
            logger.info(f"✅ High confidence -> Context Builder")
            
            # Return state to proceed (context builder will be called next)
            proceed_result = {
                "conversation_history": conversation_history,  # Pass history to next node
                "missing_slots": missing_slots,  # Pass missing slots (may be used if routed to clarification)
                "metadata": {
                    **state.get("metadata", {}),
                    "confidence_check_passed": True
                }
            }
            
            # Log full AgentState snapshot after this node
            await log_state_snapshot(state, node_name, proceed_result)
            
            return proceed_result
            
    except Exception as e:
        # Log exception
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Confidence checker failed: {str(e)}",
            stacktrace=tb,
            session_id=log_ctx["session_id"],
            node_name=node_name
        )
        
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        await persistence_store.log_exception(
            error_code=error.error_code.value,
            category=error.category.value,
            severity=error.severity.value,
            message=error.message,
            user_message=error.user_message,
            session_id=log_ctx["session_id"],
            request_id=log_ctx["request_id"],
            node_name=node_name,
            stacktrace=error.stacktrace,
            metadata=error.metadata,
            user_id=log_ctx["user_id"]
        )
        
        logger.error(f"🚨 Exception in confidence checker: {e}\n{tb}")
        
        # Return error state (will stop graph)
        result = {
            "error": error.user_message,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result
