"""
Intent Classifier Wrapper
Provides unified interface to switch between:
- Original EDGAR-style classifier (existing)
- CVS Production-Ready Classifier (new, 28+ intents)

Usage:
    from classifiers.intent_classifier_wrapper import classify_intent_unified
    
    result = classify_intent_unified(query="What's my claim status?")
    print(result['intent'], result['confidence'])
"""

from typing import Dict, Any
from config.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


def classify_intent_unified(query: str) -> Dict[str, Any]:
    """
    Classify intent using configured classifier
    
    Switches between:
    - CVS Embedding Classifier (if settings.use_cvs_intent_classifier=True and use_embedding_classifier=True)
    - CVS Keyword Classifier (if settings.use_cvs_intent_classifier=True and use_embedding_classifier=False)
    - Original EDGAR Classifier (if settings.use_cvs_intent_classifier=False)
    
    Args:
        query: User's natural language query
        
    Returns:
        Dict with:
            - intent: str
            - confidence: float
            - needs_clarification: bool (optional)
            - all_scores: dict (optional)
            - is_simple: bool (optional)
            - is_complex: bool (optional)
    """
    if settings.use_cvs_intent_classifier:
        # Use CVS Production Classifier (30 intents, production-ready)
        
        if settings.use_embedding_classifier:
            # Use Embedding-based Classifier (semantic understanding)
            logger.info("🟣 Using CVS Embedding Intent Classifier (Semantic)")
            from classifiers.embedded_classifier import CVSIntentEmbedded
            
            try:
                classifier = CVSIntentEmbedded()
                result = classifier.classify(query)
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
            # Use Keyword-based Classifier (fast, rule-based)
            logger.info("🔵 Using CVS Keyword Intent Classifier (Fast)")
            from classifiers.keyword_classifier import get_cvs_intent_classifier
            
            classifier = get_cvs_intent_classifier()
            result = classifier.classify(query)
        
        # Ensure 'needs_clarification' key exists (for compatibility)
        if 'needs_clarification' not in result:
            result['needs_clarification'] = result['confidence'] < 0.4
        
        return result
    
    else:
        # Use Original EDGAR-style Classifier
        logger.info("🟢 Using Original EDGAR Intent Classifier")
        from classifiers.intent_classifier import get_intent_classifier
        
        classifier = get_intent_classifier()
        result = classifier.classify(query)
        
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
    
    # If intent provided, check for required slots
    if intent:
        slot_validation = extractor.extract_required_slots(intent, result['entities'])
        result['slot_validation'] = slot_validation
    
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

