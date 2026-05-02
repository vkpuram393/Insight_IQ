"""
Extended Intent Agent Node (NO LLM REQUIRED)
Alternative to intent_agent.py that uses keyword-based or embedding-based classification

🚀 BENEFITS:
- NO LLM costs for intent classification
- Faster (< 10ms for keyword, ~200ms for embeddings vs 500ms+ for LLM)
- 30+ CVS-specific intents
- Production-ready with conversation continuity
- Can run side-by-side with existing LLM agent

🔧 TO USE:
In langgraph_agent.py, replace:
    workflow.add_node("intent_agent", intent_agent_node)
With:
    workflow.add_node("intent_agent", extended_intent_agent_node)
"""

import traceback
from typing import Dict, Any
from state.schema import AgentState
from core.logger import get_logger
from core.errors.models import create_internal_error
from core.logging_context import extract_logging_context, log_state_snapshot
from classifiers.intent_classifier_wrapper import classify_intent_unified, extract_entities_unified
from config.api_routing_config import get_api_config  # NEW: Get API endpoint from config
from config.config import settings  # NEW: For classifier_type metadata
from persistence import PersistenceStoreFactory

# Safety net: detect claim-history-search queries even if the upstream
# classifier doesn't know about them.  This guarantees that queries like
# "show me all rejected claims" or "manufactured by MACLEODS" don't get
# misrouted to the single-claim flow when the multidomain classifier is
# unavailable / not trained.
try:
    from Claims_search_api.intent_router import _CLAIMS_SEARCH_PATTERNS
except Exception:  # pragma: no cover - defensive
    _CLAIMS_SEARCH_PATTERNS = []

logger = get_logger(__name__)


def _looks_like_claim_history_query(text: str) -> bool:
    """Regex-based heuristic — mirror of intent_router.is_claims_search_query."""
    if not text:
        return False
    for pat in _CLAIMS_SEARCH_PATTERNS:
        if pat.search(text):
            return True
    return False


async def extended_intent_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Classify user intent using CVS keyword-based classifier
    NO LLM required - fast and cost-effective
    """
    node_name = "intent_agent"
    log_ctx = extract_logging_context(state)
    
    try:
        logger.info("🤖 CVS AGENT: Intent Classification (NO LLM)")
        
        text = state["text"]
        
        # Classify intent using wrapper (respects settings.use_embedding_classifier)
        # Now async - reuses MongoDB connection like team's pattern
        intent_result = await classify_intent_unified(text)
        
        # ========== CHECK 1: Handle embedding classifier failure ==========
        if intent_result.get('embedding_failed', False):
            logger.warning("❌ Embedding classifier failed - setting flag to route to response_agent (LLM)")
            fallback_reason = intent_result.get('fallback_reason', 'Unknown')
            
            result = {
                "intent": None,
                "confidence": 0.0,
                "entities": {},
                "embedding_failed": True,  # KEY: Router will check this flag
                "is_complex": False,
                "needs_clarification": False,
                "metadata": {
                    "embedding_failure": True,
                    "fallback_reason": fallback_reason
                }
            }
            
            # Log to telemetry
            await log_state_snapshot(state, node_name, result)
            
            logger.info("🔄 Routing to response_agent (LLM) due to embedding failure")
            return result
        
        # ========== Normal flow (embedding succeeded) ==========
        intent = intent_result['intent']
        confidence = intent_result['confidence']
        # NEW: domain (populated by multidomain classifier; may be None for legacy classifiers)
        domain = intent_result.get('domain')
        
        logger.info(f"🎯 Intent: {intent} ({confidence:.2f}) | Domain: {domain}")
        
        # Extract entities
        entity_result = extract_entities_unified(text, intent=intent)
        entities = entity_result['entities']

        # If the multidomain classifier extracted entities from the query
        # (e.g. claim_number) merge those in too — they often catch IDs the
        # generic entity extractor misses.
        md_entities = intent_result.get('entities_from_query') or {}
        if md_entities:
            for k, v in md_entities.items():
                if not v:
                    continue
                # Map common keys to the system's expected list-format
                if k in ("claim_number", "claimNumber", "claim_id", "claimId"):
                    existing = entities.get("claim_ids") or []
                    if isinstance(existing, str):
                        existing = [existing]
                    val_list = v if isinstance(v, list) else [v]
                    merged = list(dict.fromkeys(existing + val_list))
                    if merged:
                        entities["claim_ids"] = merged
                elif k in ("sequence", "claimSequence", "claim_sequence"):
                    existing = entities.get("claim_sequences") or []
                    if isinstance(existing, str):
                        existing = [existing]
                    val_list = v if isinstance(v, list) else [v]
                    entities["claim_sequences"] = list(dict.fromkeys(existing + val_list))
                else:
                    entities.setdefault(k, v)
        
        logger.info(f"📦 Entities: {entities}")
        
        # ========== NEW: Get API configuration from config file ==========
        api_config = get_api_config(intent)
        api_endpoint = api_config.get("api_endpoint")
        required_entities_list = api_config.get("required_entities", [])
        requires_llm = api_config.get("requires_llm", False)

        # ------------------------------------------------------------------
        # SAFETY NET: when the upstream classifier did NOT label this as a
        # claim_history_search query (e.g. fallback embedding classifier
        # doesn't know about NDC / RejectCode / DrugLast labels) but the
        # raw text clearly looks like a member-history search, override
        # the routing fields so the rest of the graph treats it as one:
        #   - domain = claim_history_search        (router will dispatch)
        #   - required_entities = ["claim_number"] (no sequence required)
        #   - api_endpoint = /claims/search         (member-history endpoint)
        # ------------------------------------------------------------------
        if (domain != "claim_history_search"
                and api_config.get("domain") != "claim_history_search"
                and _looks_like_claim_history_query(text)):
            from config.api_routing_config import CLAIM_HISTORY_SEARCH_ENDPOINT
            logger.info(
                "🛟 Claim-history pattern detected — overriding domain to "
                "'claim_history_search' (claim_number only, no sequence)"
            )
            domain = "claim_history_search"
            api_endpoint = CLAIM_HISTORY_SEARCH_ENDPOINT
            required_entities_list = ["claim_number"]
            requires_llm = False
            # Boost confidence so confidence_check_router proceeds straight
            # to build_context instead of detouring through llm_judge — the
            # regex match is itself a high-precision signal.
            if confidence < 0.75:
                logger.info(
                    f"🛟 Boosting confidence from {confidence:.2f} → 0.85 "
                    f"(claim-history regex is a strong signal)"
                )
                confidence = 0.85

        # Log API routing decision
        if api_endpoint:
            logger.info(f"🔗 API Endpoint: {api_endpoint}")
            logger.info(f"📋 Required Entities: {required_entities_list}")
        else:
            logger.info(f"💬 No API needed - Intent will use {'LLM' if requires_llm else 'FAQ/Knowledge Base'}")
        
        # ========== USE CLASSIFIER'S COMPLEXITY DETECTION (no duplicate calculation) ==========
        is_complex = intent_result.get('is_complex', False)
        
        if is_complex:
            logger.info(f"🧠 Complex query detected by classifier (is_complex=True)")
            logger.info("   Query contains aggregations, comparisons, date ranges, or multiple conditions")
        
        # ========== SLOT VALIDATION REMOVED (Old System 1) ==========
        # Required slots are now checked by confidence_check_router using:
        # - required_entities_list from api_routing_config.py (set below)
        # The router compares required_entities_list against extracted entities
        # and routes to clarification if any are missing.
        # ============================================================
        
        result = {
            "intent": intent,
            "confidence": confidence,
            "domain": domain,  # NEW: propagate domain (used by claim_history_search router)
            "entities": entities,
            "is_complex": is_complex,  # Complexity detection for LLM routing
            # NOTE: needs_clarification and missing_slots are now determined by
            # confidence_check_router based on required_entities_list (from api_routing_config.py)
            # API routing info from config (System 2 - the ONLY system now)
            "api_endpoint": api_endpoint,
            "required_entities_list": required_entities_list,  # Used by confidence_check_router
            "requires_llm": requires_llm,
            # NEW: Add observability metadata
            "metadata": {
                **state.get("metadata", {}),  # Preserve existing metadata
                "all_scores": intent_result.get('all_scores', {}),  # All intent similarities for debugging
                "is_complex": is_complex,  # Complex query flag (single source of truth)
                "classifier_type": (
                    "multidomain" if getattr(settings, "use_multidomain_classifier", False)
                    else ("embedding" if settings.use_embedding_classifier else "keyword")
                ),
                "intent_domain": domain,
                "intent_classification_metadata": {
                    "top_intent": intent,
                    "top_confidence": confidence,
                    "domain": domain,
                    "num_intents_evaluated": len(intent_result.get('all_scores', {})),
                }
            }
        }
        
        # Log to telemetry database (same as remote MVP-1's intent_agent.py)
        await log_state_snapshot(state, node_name, result)
        
        return result
        
    except Exception as e:
        # Standard exception handler (pattern from clarification.py, confidence.py, etc.)
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Intent classification failed: {str(e)}",
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
        
        logger.error(f"🚨 Exception in intent classification: {e}\n{tb}")
        
        # Return graceful error state compatible with downstream confidence_check_router
        # embedding_failed=True ensures router routes to llm_judge (existing handling at confidence.py lines 91-97)
        result = {
            "intent": None,
            "confidence": 0.0,
            "entities": {},
            "embedding_failed": True,
            "is_complex": False,
            "needs_clarification": False,
            "error": error.user_message,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result

