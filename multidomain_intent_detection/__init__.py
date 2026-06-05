"""
Multidomain Intent Detection
==============================

Clean, consolidated package for multi-domain intent classification
across the PBM (Pharmacy Benefit Manager) system.

Quick start:
    from multidomain_intent_detection import get_classifier, classify_query

    # Option 1: Full classifier object
    classifier = get_classifier()
    result = classifier.classify("What is the copay on claim 132435151040074?")

    # Option 2: One-liner convenience
    result = classify_query("Show all rejected claims for this member")

    # Result dict
    print(result["intent"])       # "Status"
    print(result["domain"])       # "claim_history_search"
    print(result["confidence"])   # 0.95
    print(result["entities"])     # {"claim_number": "132435151040074"}

Modules:
    config       — Intent/domain mappings, descriptions, endpoints
    normalizer   — Query normalization + entity extraction
    embeddings   — Vertex AI embedding client (singleton)
    pipeline     — PCA + Ensemble (SVM-RBF / LogReg / kNN)
    llm_fallback — Gemini Flash fallback for low-confidence queries
    classifier   — Production classifier (orchestrates all above)
    training     — Training, augmentation, evaluation

Domains (6):
    cap_api, benefits_api, claim_history_search,
    member_domain, override_domain, general
"""

from multidomain_intent_detection.classifier import (
    MultidomainIntentClassifier,
    get_classifier,
)
from multidomain_intent_detection.config import (
    INTENT_TO_DOMAIN,
    DOMAIN_ENDPOINTS,
    DOMAIN_NAMES,
    INTENT_DESCRIPTIONS,
    get_domain_for_intent,
    get_endpoint_for_domain,
    get_all_intents,
    get_all_domains,
    get_intents_for_domain,
)
from multidomain_intent_detection.normalizer import (
    normalize_query,
    extract_entities,
)

__all__ = [
    # Main classifier
    "MultidomainIntentClassifier",
    "get_classifier",
    "classify_query",
    # Config lookups
    "INTENT_TO_DOMAIN",
    "DOMAIN_ENDPOINTS",
    "DOMAIN_NAMES",
    "INTENT_DESCRIPTIONS",
    "get_domain_for_intent",
    "get_endpoint_for_domain",
    "get_all_intents",
    "get_all_domains",
    "get_intents_for_domain",
    # Normalizer
    "normalize_query",
    "extract_entities",
]


def classify_query(query: str, **kwargs) -> dict:
    """Convenience function — classify a single query using the singleton.

    Equivalent to:
        get_classifier().classify(query)

    Accepts the same keyword arguments as get_classifier() on first call
    (confidence_threshold, margin_threshold, use_llm_fallback).
    """
    return get_classifier(**kwargs).classify(query)
