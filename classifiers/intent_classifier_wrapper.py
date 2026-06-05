"""
Intent Classifier Wrapper
Provides unified interface to switch between:
- Embedding-based Classifier (semantic understanding) - use_embedding_classifier=True
- Keyword-based Classifier (fast, rule-based) - use_embedding_classifier=False

Usage:
    from classifiers.intent_classifier_wrapper import classify_intent_unified
    
    result = classify_intent_unified(query="What's my claim status?")
    print(result['intent'], result['confidence'])
"""

from typing import Dict, Any
from config.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


async def classify_intent_unified(query: str) -> Dict[str, Any]:
    """
    Classify intent using configured classifier (ASYNC version - like team's pattern)
    
    Switches between:
    - Multidomain Classifier (if settings.use_multidomain_classifier=True) - PCA + Ensemble + LLM fallback
    - Embedding Classifier (if settings.use_embedding_classifier=True) - semantic understanding
    - Keyword Classifier (if settings.use_embedding_classifier=False) - fast, rule-based
    
    Args:
        query: User's natural language query
        
    Returns:
        Dict with:
            - intent: str
            - confidence: float
            - domain: str (optional, populated by multidomain classifier)
            - needs_clarification: bool (optional)
            - all_scores: dict (optional)
            - is_simple: bool (optional)
            - is_complex: bool (optional)
    """
    # ------------------------------------------------------------------
    # Option 0: Multidomain classifier (PCA + Ensemble + LLM fallback)
    # Provides the `domain` field used to dispatch claim_history_search
    # queries to the member-history search pipeline.
    # ------------------------------------------------------------------
    if getattr(settings, "use_multidomain_classifier", False):
        logger.info("🟪 Using Multidomain Intent Classifier (PCA + Ensemble + LLM)")
        try:
            from multidomain_intent_detection import get_classifier
            md_classifier = get_classifier()
            # Run synchronous predict_single in an executor
            import asyncio as _asyncio
            loop = _asyncio.get_event_loop()
            md_result = await loop.run_in_executor(None, md_classifier.classify, query)

            # Normalize to the same shape the rest of the system expects
            result = {
                "intent": md_result.get("intent"),
                "confidence": float(md_result.get("confidence") or 0.0),
                "domain": md_result.get("domain"),
                "domain_name": md_result.get("domain_name"),
                "api_endpoint": md_result.get("api_endpoint"),
                "needs_clarification": md_result.get("needs_clarification", False),
                "is_complex": False,  # multidomain classifier doesn't compute this; let downstream decide
                "all_scores": dict(md_result.get("top_5") or []),
                # Wider top-N list (with intent + score) for response observability.
                # Each item is (intent_name, confidence_score). Falls back to top_5 if absent.
                "top_n": md_result.get("top_n") or md_result.get("top_5") or [],
                # LLM-fallback chain-of-thought (only present when the LLM path was used
                # AND settings.enable_llm_fallback_thinking=True).
                "llm_thinking": md_result.get("llm_thinking"),
                "llm_reasoning": md_result.get("llm_reasoning"),
                "source": md_result.get("source"),
                "entities_from_query": md_result.get("entities") or {},
                "llm_fallback_confidence": md_result.get("llm_fallback_confidence"),
            }
            return result
        except Exception as e:
            logger.error(
                "[INTENT-CLASSIFIER] ============================================\n"
                "[INTENT-CLASSIFIER]  MULTIDOMAIN CLASSIFIER FAILED\n"
                f"[INTENT-CLASSIFIER]  Error  : {type(e).__name__}: {e}\n"
                "[INTENT-CLASSIFIER]  Action : raising — no fallback when\n"
                "[INTENT-CLASSIFIER]           use_multidomain_classifier=True\n"
                "[INTENT-CLASSIFIER] ============================================"
            )
            raise

    if settings.use_embedding_classifier:
        # Use Embedding-based Classifier (semantic understanding)
        logger.info("🟣 Using CVS Embedding Intent Classifier (Semantic - Async)")
        from classifiers.embedded_classifier import get_embedded_classifier
        
        try:
            # Use singleton to prevent memory leaks (embeddings loaded once, ~800MB)
            classifier = get_embedded_classifier()
            # Use async version - reuses MongoDB connection like team's pattern
            result = await classifier.classify_async(query)
        except RuntimeError as e:
            # Embeddings unavailable - return special flag to route to LLM
            logger.error(f"❌ Embedding classifier failed: {e}")
            logger.info("🔄 Routing query directly to Response LLM Agent")
            return {
                'intent': 'embedding_failed',
                'confidence': 0.0,
                'needs_clarification': False,
                'embedding_failed': True,  # Special flag for router
                'fallback_reason': str(e)
            }
    else:
        # Use Keyword-based Classifier (fast, rule-based) - sync is fine
        logger.info("🔵 Using CVS Keyword Intent Classifier (Fast)")
        from classifiers.keyword_classifier import get_cvs_intent_classifier
        
        classifier = get_cvs_intent_classifier()
        result = classifier.classify(query)
    
    # Ensure 'needs_clarification' key exists (for compatibility)
    if 'needs_clarification' not in result:
        result['needs_clarification'] = result['confidence'] < 0.4
    
    return result


def extract_entities_unified(query: str, intent: str = None) -> Dict[str, Any]:
    """
    Extract entities from query
    
    Uses the CVS Entity Extractor which provides:
    - Claim IDs (CLM123, CLM456, ...)
    - Member IDs (MEM123)
    - Prescription IDs (RX123)
    - Dates (October, last month, yesterday, etc.)
    - Amounts ($45.20)
    
    Args:
        query: User's natural language query
        intent: Optional intent (used for slot validation)
        
    Returns:
        Dict with:
            - entities: dict (claim_id, member_id, dates, etc.)
            - validation: dict (validation status)
            - needs_validation: bool
            - missing_required: list
    """
    from utils.entity_extractor import get_entity_extractor
    
    extractor = get_entity_extractor()
    result = extractor.extract(query)
    
    # NOTE: Removed extract_required_slots() call - this was the OLD system (System 1)
    # Required slots are now checked via api_routing_config.py (System 2)
    # The confidence_check_router uses required_entities_list from api_routing_config
    # which is set in extended_intent_agent_node.py via get_api_config(intent)
    
    return result


# ========== FOR TESTING / COMPARISON ==========

def compare_classifiers(query: str) -> Dict[str, Any]:
    """
    Compare both classifiers side-by-side (for testing)
    
    Returns:
        {
            'query': str,
            'original': {intent, confidence, ...},
            'cvs': {intent, confidence, ...},
            'agreement': bool,
            'confidence_diff': float
        }
    """
    from classifiers.intent_classifier import get_intent_classifier as get_original
    from classifiers.keyword_classifier import get_cvs_intent_classifier
    
    original_classifier = get_original()
    cvs_classifier = get_cvs_intent_classifier()
    
    original_result = original_classifier.classify(query)
    cvs_result = cvs_classifier.classify(query)
    
    agreement = original_result['intent'] == cvs_result['intent']
    confidence_diff = abs(
        original_result['confidence'] - cvs_result['confidence']
    )
    
    return {
        'query': query,
        'original': original_result,
        'cvs': cvs_result,
        'agreement': agreement,
        'confidence_diff': confidence_diff
    }

