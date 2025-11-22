"""
Intent Classifier - EDGAR-inspired
Classifies user queries by intent using keyword weighting and pattern matching
NO LLM required for classification - fast and cost-effective
"""

import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Classifies natural language queries by intent using keyword scoring
    Inspired by EDGAR's approach but adapted for MyClaim domain
    """
    
    def __init__(self):
        # Define intent-specific keyword weights
        # Higher weight = stronger signal for that intent
        self.keyword_weights = {
            # ========== CLAIM INTENTS ==========
            'claim_status': {
                'claim': 0.6, 'status': 0.7, 'where': 0.4,
                'track': 0.5, 'my': 0.2, 'check': 0.3,
                'what': 0.2, 'is': 0.1
            },
            
            'rejection_reasons': {
                'rejected': 0.8, 'deny': 0.7, 'denied': 0.7,
                'why': 0.5, 'reason': 0.6, 'rejection': 0.8,
                'claim': 0.4, 'refused': 0.7
            },
            
            'claim_pending': {
                'pending': 0.8, 'submitted': 0.6, 'waiting': 0.6,
                'claim': 0.5, 'processing': 0.5, 'review': 0.4
            },
            
            'claim_details': {
                'details': 0.7, 'information': 0.5, 'about': 0.3,
                'claim': 0.6, 'show': 0.4, 'get': 0.3,
                'find': 0.3, 'tell': 0.3
            },
            
            'claim_summary': {
                'summarize': 0.9, 'summary': 0.9, 'overview': 0.7,
                'total': 0.6, 'all': 0.4, 'aggregate': 0.7,
                'claim': 0.4, 'claims': 0.4
            },
            
            'expensive_claims': {
                'expensive': 0.9, 'costly': 0.8, 'high': 0.6,
                'most': 0.5, 'largest': 0.7, 'biggest': 0.7,
                'amount': 0.4, 'claim': 0.3, 'claims': 0.3
            },
            
            'claim_list': {
                'list': 0.7, 'all': 0.5, 'claims': 0.6,
                'show': 0.4, 'history': 0.5, 'previous': 0.5,
                'past': 0.5
            },
            
            'date_range_search': {
                'october': 0.7, 'november': 0.7, 'december': 0.7,
                'january': 0.7, 'february': 0.7, 'march': 0.7,
                'april': 0.7, 'may': 0.7, 'june': 0.7,
                'july': 0.7, 'august': 0.7, 'september': 0.7,
                'last': 0.5, 'month': 0.6, 'period': 0.5,
                'between': 0.6, 'from': 0.4, 'to': 0.3,
                'claim': 0.3, 'claims': 0.3
            },
            
            # ========== PRESCRIPTION INTENTS ==========
            'prescription_info': {
                'prescription': 0.8, 'medication': 0.7, 'drug': 0.6,
                'refill': 0.6, 'pharmacy': 0.5, 'medicine': 0.6,
                'rx': 0.7
            },
            
            'prescription_status': {
                'prescription': 0.7, 'status': 0.7, 'refill': 0.6,
                'active': 0.5, 'expired': 0.5, 'remaining': 0.5
            },
            
            'refill_info': {
                'refill': 0.8, 'renew': 0.7, 'renewal': 0.7,
                'prescription': 0.5, 'medication': 0.5,
                'doctor': 0.4, 'visit': 0.3
            },
            
            'medication_coverage': {
                'covered': 0.8, 'coverage': 0.8, 'formulary': 0.7,
                'medication': 0.6, 'drug': 0.5, 'plan': 0.5,
                'covered': 0.8
            },
            
            # ========== MEMBER INTENTS ==========
            'member_info': {
                'member': 0.7, 'benefits': 0.6, 'plan': 0.6,
                'my': 0.3, 'information': 0.5, 'account': 0.5
            },
            
            'benefits_info': {
                'benefits': 0.8, 'coverage': 0.7, 'deductible': 0.7,
                'copay': 0.7, 'out': 0.4, 'pocket': 0.4,
                'maximum': 0.5
            },
            
            'copay_info': {
                'copay': 0.9, 'copayment': 0.9, 'cost': 0.5,
                'pay': 0.5, 'amount': 0.4, 'much': 0.3
            },
            
            'deductible_info': {
                'deductible': 0.9, 'out': 0.5, 'pocket': 0.5,
                'met': 0.6, 'remaining': 0.5
            },
            
            # ========== GENERAL INTENTS ==========
            'help': {
                'help': 0.9, 'assistance': 0.8, 'support': 0.7,
                'what': 0.3, 'can': 0.3, 'do': 0.2
            },
            
            'greeting': {
                'hi': 0.9, 'hello': 0.9, 'hey': 0.8,
                'good': 0.5, 'morning': 0.4, 'afternoon': 0.4,
                'evening': 0.4
            },
            
            'appeal_info': {
                'appeal': 0.9, 'dispute': 0.7, 'challenge': 0.6,
                'overturn': 0.6, 'reconsider': 0.7
            }
        }
        
        # Define which intents are SIMPLE vs COMPLEX
        # SIMPLE → Can use pattern matching and templates
        # COMPLEX → Need LLM processing
        self.simple_intents = [
            'claim_status', 'claim_details', 'claim_pending',
            'claim_list', 'prescription_info', 'prescription_status',
            'member_info', 'benefits_info', 'copay_info', 'deductible_info',
            'greeting', 'help'
        ]
        
        self.complex_intents = [
            'claim_summary', 'expensive_claims', 'date_range_search',
            'rejection_reasons', 'appeal_info'
        ]
        
        # Regex patterns for additional detection
        self.patterns = {
            'claim_status': [
                r'\bstatus\s+of\s+claim\b',
                r'\bwhere\s+is\s+(my|the)\s+claim\b',
                r'\btrack\s+(my|the)\s+claim\b'
            ],
            'rejection_reasons': [
                r'\bwhy\s+.*\brejected\b',
                r'\brejection\s+reason\b',
                r'\bdenied\s+claim\b'
            ],
            'expensive_claims': [
                r'\bmost\s+expensive\b',
                r'\bhighest\s+(amount|cost|price)\b',
                r'\bcostly\s+claims\b'
            ]
        }
        
        # Compile patterns
        self.compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.patterns.items()
        }
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Main classification method
        Returns dict with intent, confidence, and metadata
        """
        query_lower = query.strip().lower()
        
        # Handle empty/very short queries
        if not query_lower or len(query_lower) < 2:
            return {
                'intent': 'empty_query',
                'confidence': 0.0,
                'needs_clarification': True,
                'all_scores': {},
                'top_candidates': []
            }
        
        # Handle common greetings/help
        if query_lower in ['hi', 'hello', 'hey', 'help', '?']:
            return {
                'intent': 'greeting' if query_lower != 'help' else 'help',
                'confidence': 1.0,
                'needs_clarification': False,
                'all_scores': {},
                'top_candidates': []
            }
        
        # Calculate keyword scores
        keyword_scores = self._calculate_keyword_scores(query_lower)
        
        # Calculate pattern scores
        pattern_scores = self._calculate_pattern_scores(query_lower)
        
        # Combine scores (take maximum of keyword or pattern)
        all_scores = {}
        for intent in set(list(keyword_scores.keys()) + list(pattern_scores.keys())):
            all_scores[intent] = max(
                keyword_scores.get(intent, 0.0),
                pattern_scores.get(intent, 0.0)
            )
        
        if not all_scores:
            return {
                'intent': 'out_of_scope',
                'confidence': 0.0,
                'needs_clarification': True,
                'all_scores': {},
                'top_candidates': []
            }
        
        # Get top candidates
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_score = sorted_scores[0]
        
        # Check for ambiguity (top 2 scores too close)
        needs_clarification = False
        candidates = []
        
        if len(sorted_scores) > 1:
            second_score = sorted_scores[1][1]
            if top_score - second_score < 0.05:  # Too close!
                needs_clarification = True
                candidates = sorted_scores[:2]
        
        # Check confidence threshold
        if top_score < 0.1:  # Very low confidence
            return {
                'intent': 'out_of_scope',
                'confidence': top_score,
                'needs_clarification': True,
                'all_scores': all_scores,
                'top_candidates': []
            }
        
        return {
            'intent': top_intent,
            'confidence': top_score,
            'needs_clarification': needs_clarification,
            'all_scores': all_scores,
            'top_candidates': candidates,
            'is_simple': top_intent in self.simple_intents,
            'is_complex': top_intent in self.complex_intents
        }
    
    def _calculate_keyword_scores(self, query: str) -> Dict[str, float]:
        """
        Calculate scores based on keyword presence and weights
        EDGAR's core algorithm
        """
        scores = {}
        
        for intent, keywords in self.keyword_weights.items():
            score = 0.0
            for keyword, weight in keywords.items():
                if keyword in query:
                    score += weight
            
            # Normalize by number of keywords
            if score > 0:
                scores[intent] = min(score / len(keywords), 1.0)
        
        return scores
    
    def _calculate_pattern_scores(self, query: str) -> Dict[str, float]:
        """Calculate scores based on regex pattern matching"""
        scores = {}
        
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    # Pattern match gives high score
                    scores[intent] = max(scores.get(intent, 0.0), 0.8)
        
        return scores
    
    def get_top_intents(self, query: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """Get top N intents for a query"""
        result = self.classify(query)
        all_scores = result['all_scores']
        sorted_intents = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_intents[:top_n]
    
    def is_intent_match(self, query: str, expected_intent: str, threshold: float = 0.5) -> bool:
        """Check if query matches expected intent with minimum threshold"""
        result = self.classify(query)
        return result['intent'] == expected_intent and result['confidence'] >= threshold


# Global singleton instance
_intent_classifier = None


def get_intent_classifier() -> IntentClassifier:
    """Get global intent classifier instance"""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier


# Async wrapper for utils test endpoints
async def classify_intent(text: str, user_info: Dict) -> Dict[str, Any]:
    """
    Async wrapper for intent classification
    Used by test endpoints and other async code
    """
    classifier = get_intent_classifier()
    result = classifier.classify(text)

    return {
        "intent": result.get("intent", "unknown"),
        "confidence": result.get("confidence", 0.0),
        "reasoning": f"Classified as {result.get('intent')} with confidence {result.get('confidence', 0.0):.2f}",
        "all_scores": result.get("all_scores", {}),
        "needs_clarification": result.get("needs_clarification", False)
    }


