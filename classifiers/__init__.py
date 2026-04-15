"""
Intent Classifiers

Provides unified interface for intent classification:
- Keyword-based classifier (fast, rule-based)
- Embedding-based classifier (semantic understanding)
- Original EDGAR-style classifier (legacy)
"""

from classifiers.intent_classifier_wrapper import (
    classify_intent_unified,
    extract_entities_unified
)

__all__ = [
    "classify_intent_unified",
    "extract_entities_unified"
]

