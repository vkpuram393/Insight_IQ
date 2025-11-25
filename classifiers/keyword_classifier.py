"""
CVS Intent Classifier - Aligned with CVS Claim API
Classifies user queries based on CVS API response structure
NO LLM required for classification - fast and cost-effective
"""

import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CVSIntentClassifier:
    """
    Intent classifier optimized for CVS Claim API
    
    CVS API provides:
    - Primary claim info (number, sequence, status)
    - Beneficiary (patient) details
    - Drug information
    - Pharmacy details
    - Prescriber information
    - Pricing (patient_pay)
    - Status details (reject_details, settlement_codes, messages)
    - Prior authorization info
    - Submission details (dateOfFill, rxNumber, quantity, daysSupply)
    """
    
    def __init__(self):
        """Initialize with CVS API-aligned intents"""
        self.keyword_weights = self._build_keyword_weights()
        self.confidence_threshold = 0.50  # Increased to avoid false positives
        logger.info(f"CVS Intent Classifier initialized with {len(self.keyword_weights)} intents")
    
    def _build_keyword_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Build keyword weights aligned with CVS API fields
        
        Returns:
            Dict mapping intent to keyword weights
        """
        return {
            # ========== GENERAL CLAIM STATUS (maps to: status, statusDescription) ==========
            # SMART SUMMARY: Moderate weight catches general "claim summary" only
            'claim_status': {
                'status': 0.9, 'claim': 0.8, 'claims': 0.8, 'check': 0.6,
                # 'what' removed - too generic, dilutes score
                'where': 0.7, 'track': 0.6, 'tracking': 0.6,  # 'where' boosted
                'find': 0.5, 'look': 0.4, 'up': 0.3,
                'show': 0.7, 'number': 0.6,  # Added for "claim number"
                'summary': 0.5, 'summarize': 0.5,  # MODERATE weight - won't override specific summaries
                'generate': 0.3  # Helper keyword
            },
            
            # ========== REJECTION DETAILS (maps to: statusDetails.rejectDetails[]) ==========
            'rejection_reasons': {
                'rejected': 1.0, 'reject': 0.9, 'rejection': 0.9,
                'denied': 0.9, 'deny': 0.8,
                # 'why' removed - too generic, dilutes score when paired with strong keywords
                'reason': 0.9, 'reasons': 0.9,
                'refused': 0.8, 'decline': 0.7, 'declined': 0.7,
                # 'claim' removed - too generic, causes false positives for "where is my claim"
                'not': 0.4, 'approved': 0.4,
                'summary': 0.3, 'summarize': 0.3,  # LOW weight - "reject" keywords dominate
                'fail': 0.9, 'failed': 0.9, 'edits': 0.8  # UC11: "What edits did the claim fail?"
            },
            
            # ========== DRUG/MEDICATION INFO (maps to: drug{}, submitted{}) ==========
            'drug_info': {
                'drug': 0.9, 'medication': 0.9, 'medicine': 0.8,
                'prescription': 0.7, 'pill': 0.7, 'tablet': 0.7,
                'name': 0.3, 
                'taking': 0.5, 'prescribed': 0.3, 'rx': 0.8,
                'product': 0.5, 'gpi': 0.8
            },
            
            # ========== PHARMACY INFO (maps to: pharmacy{}) ==========
            'pharmacy_info': {
                'pharmacy': 0.9, 'pharmacies': 0.9, 'store': 0.7,  # Lowered "pharmacy" to 0.9
                'where': 0.3, 'filled': 0.8, 'dispensed': 0.7,  # Lowered "dispensed" to 0.7
                'location': 0.7, 'address': 0.7, 'cvs': 0.6,
                'picked': 0.6, 'pick': 0.5, 'got': 0.5,
                'dispense': 1.0  # UC31: "Where did the member dispense this prescription?" - BOOSTED to 1.0
            },
            
            # ========== PRESCRIBER INFO (maps to: prescriber{}) ==========
            'prescriber_info': {
                'prescriber': 1.0, 'doctor': 0.9, 'dr': 0.9,
                'physician': 0.9, 'who': 0.7, 'prescribed': 1.0,
                'wrote': 0.7, 'ordered': 0.6, 'provider': 0.8,
                'npi': 0.9, 'prescribing': 0.9
            },
            
            # ========== PRICING/COST INFO (maps to: pricing{patientPay}) ==========
            'pricing_info': {
                'cost': 0.9, 'price': 0.9, 'pricing': 0.9, 'pay': 0.8,
                'paid': 0.8, 'payment': 0.8, 'amount': 0.7,
                'much': 0.8, 'money': 0.7,  # Boosted to 0.8 for consistency with other money intents
                'copay': 0.9, 'coinsurance': 0.9, 'deductible': 0.9,
                'patient': 0.5, 'responsibility': 0.6,
                'owe': 0.7, 'owed': 0.7, 'charge': 0.7,
                'summary': 0.5, 'summarize': 0.5,  # BOOSTED - "pricing summary" wins over general summary
                'generate': 0.3,  # Helper keyword
                'schedule': 0.9, 'schedules': 0.9,  # UC20: "Explain the pricing schedule"
                'manufacturer': 0.9, 'discount': 0.9,  # UC22: "What was the manufacturer discount?"
                'modifier': 0.9, 'modifiers': 0.9  # UC25: "What co-pay modifier applied?"
            },
            
            # ========== RX DETAILS (maps to: submitted{rxNumber, fillNumber, quantity, daysSupply}) ==========
            'rx_details': {
                'rx': 1.0, 'prescription': 0.9, 'number': 0.3,  # Boosted all
                'quantity': 0.9, 'many': 0.6,
                'pills': 0.7, 'tablets': 0.7, 'days': 0.7,
                'supply': 0.8, 'fill': 0.6, 'refill': 0.6,
                'dispensed': 0.7, 'rxnumber': 1.0, 'rx#': 1.0  # Added variants
            },
            
            # ========== PRIOR AUTHORIZATION (maps to: priorAuthorization{}) ==========
            'prior_auth_info': {
                'prior': 0.9, 'authorization': 1.0, 'pa': 1.0,  # Boosted PA to ensure dominance
                'auth': 0.8, 'approval': 0.7, 'required': 0.6,
                'need': 0.5, 'needed': 0.5, 'preauth': 0.9,
                'precertification': 0.8,
                'summary': 0.6, 'summarize': 0.6,  # BOOSTED - "PA summary" wins
                'smart': 0.8, 'member': 0.5,  # LOWERED to 0.5 - only wins when combined with "PA" or "smart"
                'generate': 0.4  # Helper keyword
            },
            
            # ========== PATIENT/BENEFICIARY INFO (maps to: beneficiary{}) ==========
            'beneficiary_info': {
                'patient': 0.9, 'member': 0.3, 'beneficiary': 1.0,  # LOWERED "member" to 0.3 - only wins with other beneficiary keywords
                'my': 0.4, 'info': 0.7, 'information': 0.8,  # Boosted info-related keywords
                'details': 0.6, 'who': 0.7, 'name': 0.5,
                'id': 0.7, 'cardholder': 0.8, 'profile': 0.7,
                'accumulation': 0.9, 'benefit': 0.8, 'phase': 0.9,  # UC18: "Which accumulation benefit phase is member in?"
                'coverage': 0.9, 'type': 0.5,  # UC28: "What type of coverage does member have?"
                'medical': 0.7, 'dollars': 0.6,  # UC33: "Does member accumulation consider medical dollars?"
                'fml': 0.9, 'loe': 0.9, 'loes': 0.9, 'linked': 0.7  # UC34: "Was FML used? What are the linked member LOEs?"
            },
            
            # ========== FILL DATE (maps to: submitted.dateOfFill, submitted.date) ==========
            'fill_date_info': {
                'when': 0.8, 'date': 0.9, 'filled': 0.9,
                'dispensed': 0.7, 'got': 0.6, 'picked': 0.7,  # Lowered "dispensed" to 0.7
                'received': 0.7, 'fill': 0.7, 'time': 0.5,
                'day': 0.5
            },
            
            # ========== APPROVAL INFO (maps to: statusDetails.approvedMessages[]) ==========
            'approval_info': {
                'approved': 1.0, 'approval': 0.9, 'accepted': 0.8,
                'authorize': 0.7, 'covered': 0.7, 'paid': 0.6,
                'claim': 0.4, 'status': 0.5,
                'summary': 0.3, 'summarize': 0.3,  # LOW weight
                'tf': 0.9, 'transition': 0.9,  # For "TF summary", "transition fill summary" - BOOSTED to 0.9
                'fill': 0.6,  # UC23: for "transition fill"
                'qualify': 1.0, 'qualified': 1.0,  # UC23: "Did claim qualify for transition fill?" - BOOSTED to 1.0
                'executed': 1.0, 'options': 0.9, 'plan': 0.6,  # UC32: "Which plan options executed?" - BOOSTED
                'bypass': 1.0, 'by-pass': 1.0,  # UC41: "Why did claim by-pass accumulations?" - BOOSTED to 1.0
                'accumulations': 0.9, 'configuration': 0.7, 'setup': 0.7, 'set-up': 0.7, 'lead': 0.6,  # UC41: additional keywords
                'bpg': 0.9, 'adjudication': 0.9,  # UC44: "What BPG was used for adjudication?"
                'override': 0.9, 'overrides': 0.9, 'applied': 0.6  # For override queries
            },
            
            # ========== SETTLEMENT CODES (maps to: statusDetails.settlementCodes[]) ==========
            'settlement_info': {
                'settlement': 1.0, 'code': 0.7, 'codes': 0.7,
                'payment': 0.6, 'processed': 0.7,
                'much': 0.8, 'insurance': 0.8, 'paid': 0.8, 'pay': 0.8,  # Money-related keywords
                'amount': 0.7, 'covered': 0.6,
                'response': 1.0,  # UC8: "What was the response to the pharmacy?" - BOOSTED to 1.0
                'summary': 0.3, 'summarize': 0.3  # LOW weight - "settlement" keyword dominates
            },
            
            # ========== DATE RANGE QUERIES (requires dateOfFill filtering) ==========
            'date_range_claims': {
                'october': 0.9, 'november': 0.9, 'december': 0.9,
                'january': 0.9, 'february': 0.9, 'march': 0.9,
                'april': 0.9, 'may': 0.9, 'june': 0.9,
                'july': 0.9, 'august': 0.9, 'september': 0.9,
                # Removed 'last' - too generic, causes false positives with "last week"
                'month': 0.8, 'year': 0.7, 'months': 0.8, 'years': 0.7,
                'between': 0.9, 'from': 0.8, 'to': 0.7,  # Boosted date range indicators
                'during': 0.7, 'in': 0.4, 'claims': 0.6,
                'period': 0.8, 'range': 0.9  # Boosted range
            },
            
            # ========== MULTI-CLAIM SUMMARY (for multiple claim IDs) ==========
            # This should ONLY match for truly multiple claims: "all my claims", "claims CLM1 and CLM2"
            # NOT for single claim summaries like "pricing summary for CLM123"
            'multi_claim_summary': {
                'claims': 0.9, 'all': 0.8, 'multiple': 0.9,
                # REMOVED 'summary' and 'summarize' - they should route to specific intents!
                'list': 0.7,
                'show': 0.6, 'my': 0.3, 'both': 0.7,
                'and': 0.4, 'together': 0.6, 'combined': 0.7
            },
            
            # ========== SPECIAL INTENTS ==========
            'greeting': {
                'hi': 1.0, 'hello': 1.0, 'hey': 1.0,
                'good': 0.5, 'morning': 0.5, 'afternoon': 0.5,
                'evening': 0.5, 'greetings': 1.0
            },
            
            'help': {
                'help': 0.6, 'assist': 0.6, 'support': 0.6,  # Lowered to not override domain intents
                'guide': 0.7, 'tutorial': 0.7, 'instructions': 0.7
            },
            
            'appeal_info': {
                'appeal': 1.0, 'dispute': 0.9, 'challenge': 0.8,
                'disagree': 0.8, 'reconsider': 0.9, 'review': 0.7,
                'overturn': 0.8, 'resubmit': 0.8,
                'overcome': 1.0, 'done': 0.6  # UC13: "What can be done to overcome the reject?" - BOOSTED to 1.0
            },
            
            # ========== ADDITIONAL CVS-SPECIFIC FIELDS ==========
            
            # COMPOUND MEDICATIONS (maps to: compound)
            'compound_info': {
                'compound': 1.0, 'compounded': 0.9, 'custom': 0.8,
                'mixed': 0.8, 'specially': 0.7, 'prepared': 0.7,
                'formula': 0.8, 'mixture': 0.8,
                # Removed 'medication' - it was causing drug_info to win over compound_info
                # "compound" alone is a strong enough indicator
                'customized': 0.7, 'combination': 0.7,
                'mic': 1.0, 'ingredients': 0.9, 'ingredient': 0.9  # UC42: "Is this claim for MIC? What are the ingredients?"
            },
            
            # MEDICARE PART D (maps to: medD)
            'medicare_part_d': {
                'medicare': 1.0, 'part': 0.9, 'd': 0.9,
                'medD': 1.0, 'medd': 1.0, 'government': 0.6, 'federal': 0.6,
                'cms': 0.8, 'partd': 1.0,
                'summary': 0.5, 'summarize': 0.5,  # BOOSTED - "MEDD summary" wins
                'pde': 0.9,  # For "PDE summary"
                'lics': 0.7, 'n1': 0.7,  # For "LICS and N1's"
                'generate': 0.3  # Helper keyword
            },
            
            # DISPENSE AS WRITTEN / DAW (maps to: dispenseAsWritten, dawproductSelectionCode)
            'daw_info': {
                'brand': 0.9, 'generic': 0.8, 'substitute': 0.9,  # Boosted substitute
                'daw': 1.0, 'substitution': 0.9, 'dispense': 0.8,  # Boosted dispense
                'dispensed': 0.8, 'written': 0.7, 'required': 0.5, 'allowed': 0.8,  # Added dispensed
                'as': 0.6, 'prescribed': 0.7  # Boosted context keywords
            },
            
            # COORDINATION OF BENEFITS (maps to: cobClaimIndicator, linkedClaims.stcob)
            'cob_info': {
                'coordination': 1.0, 'benefits': 0.8, 'cob': 1.0,
                'other': 0.5, 'insurance': 0.7, 'primary': 0.7,
                'secondary': 0.8, 'multiple': 0.6, 'plans': 0.6,
                'much': 0.8, 'paid': 0.7, 'covered': 0.6,  # Money-related keywords
                'member': 0.6,  # Member with multiple insurance
                'summary': 0.3, 'summarize': 0.3,  # LOW weight - "cob"/"stcob" dominates
                'stcob': 1.0,  # For "STCOB summary"
                'coverage': 0.4  # UC28: LOW weight - beneficiary_info should win for "member coverage type"
            },
            
            # PHARMACY NETWORK (maps to: pharmacyNetwork)
            'network_info': {
                'network': 1.0, 'in-network': 1.0, 'out-of-network': 1.0,  # Hyphenated versions
                'in': 0.9, 'out': 0.9,  # Boosted to 0.9 to compete with "pharmacy"
                'preferred': 0.9, 'participating': 0.9,
                'tier': 0.8, 'coverage': 0.4  # UC28: LOW weight - beneficiary_info should win for "member coverage type"
                # Removed "pharmacy" - it was causing pharmacy_info to win over network_info
            },
            
            # REIMBURSEMENT TYPE (maps to: reimbursementType)
            'reimbursement_info': {
                'reimbursement': 1.0, 'reimbursed': 0.9, 'paid': 0.6,
                'payment': 0.7, 'type': 0.5, 'method': 0.6,
                'processed': 0.5,
                'much': 0.8, 'back': 0.6, 'refund': 0.7, 'return': 0.6  # Money-related keywords
            },
            
            # GOVERNMENT CLAIM TYPE (maps to: governmentClaimType)
            'government_claim_type': {
                'government': 1.0, 'medicaid': 0.9, 'medicare': 0.6,
                'federal': 0.8, 'state': 0.7, 'program': 0.7,
                'type': 0.8,  # UC45: BOOSTED - only wins when combined with "government"
                'claim': 0.3  # UC45: LOW weight - don't catch all "claim" queries, only when "government" present
            },
            
            # MAIL ORDER (maps to: mail)
            'mail_order_info': {
                'mail': 1.0, 'order': 0.8, 'delivery': 0.8,
                'shipped': 0.8, 'mailed': 0.9, 'home': 0.6,
                'delivered': 0.7, 'sent': 0.6
            },
            
            # MULTI-SOURCE INDICATOR (maps to: multiSourceInd)
            'generic_availability': {
                'generic': 1.0, 'available': 0.9, 'alternative': 0.9,  # Boosted all keywords
                'multi-source': 1.0, 'cheaper': 0.8, 'option': 0.7,
                'substitute': 0.7, 'equivalent': 0.9,
                'availability': 0.8, 'exist': 0.6, 'other': 0.5  # Added context
            },
            
            # DRUG UTILIZATION REVIEW (maps to: durExistenceStatus)
            'drug_interaction_info': {
                'interaction': 1.0, 'interactions': 1.0, 'dur': 1.0,
                'conflict': 0.8, 'warning': 0.9, 'alert': 0.9,
                'safety': 0.7, 'contraindication': 0.9,
                'compatible': 0.7, 'safe': 0.6
            },
            
            # REVERSAL INFO (maps to: rnR)
            'reversal_info': {
                'reversal': 1.0, 'reversed': 0.9, 'reverse': 0.9,
                'undo': 0.7, 'cancelled': 0.8, 'cancel': 0.7,
                'voided': 0.8, 'void': 0.7,
                'adjustments': 1.0, 'adjustment': 1.0,  # UC38: "Did this claim have any adjustments?" - BOOSTED to 1.0
                'r&r': 1.0, 'rnr': 1.0, 'manual': 0.9  # UC38: "R&R, Manual etc." - BOOSTED manual to 0.9
            },
            
            # AUDIT TRAIL (maps to: audit{})
            'audit_info': {
                'audit': 1.0, 'history': 0.8, 'changes': 0.8,
                'modified': 0.7, 'updated': 0.7, 'changed': 0.7,
                'who': 0.7, 'when': 0.8, 'trail': 0.8
            },
            
            # OUT OF SCOPE (fallback intent - no keywords, used when no match found)
            'out_of_scope': {
                # No keywords - this is a fallback intent when confidence is too low
                # or no intent matches. Handled programmatically in classify() method.
            },
        }
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify user query by intent
        
        Args:
            query: User's natural language query
            
        Returns:
            Dict with intent, confidence, and metadata
        """
        query_lower = query.lower().strip()
        
        # Check for empty query
        if not query_lower:
            return {
                'intent': 'out_of_scope',
                'confidence': 0.0,
                'all_scores': {},
                'is_complex': False
            }
        
        # Calculate scores for all intents
        intent_scores = self._calculate_intent_scores(query_lower)
        
        # Get top intent
        if not intent_scores:
            return {
                'intent': 'out_of_scope',
                'confidence': 0.0,
                'all_scores': {},
                'is_complex': False
            }
        
        top_intent = max(intent_scores, key=intent_scores.get)
        top_score = intent_scores[top_intent]
        
        # ========== SPECIAL CASE: Greeting as conversation starter ==========
        # If "greeting" wins but there's a strong real intent after it, use that instead
        # E.g., "hi where is my claim CLM123" → claim_status, not greeting
        if top_intent == 'greeting':
            # Check if query has multiple words (not just "hi" or "hello")
            query_words = query_lower.split()
            greeting_words = {'hi', 'hey', 'hello', 'greetings', 'good', 'morning', 'afternoon', 'evening'}
            
            # If there are non-greeting words in the query
            if len(query_words) > 2:
                # Find the best NON-GREETING intent
                sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
                for intent_name, score in sorted_intents:
                    if intent_name != 'greeting' and score >= 0.40:
                        # Found a real intent with reasonable confidence
                        logger.info(f"Detected greeting as conversation starter, using {intent_name} ({score:.2f}) instead")
                        top_intent = intent_name
                        top_score = score
                        break
        
        # Check if below threshold
        if top_score < self.confidence_threshold:
            return {
                'intent': 'out_of_scope',
                'confidence': top_score,
                'all_scores': intent_scores,
                'is_complex': False
            }
        
        # Classify complexity (CRITICAL: Detects aggregations, date ranges, comparisons)
        is_complex = self._is_complex_query(query_lower, top_intent)
        
        logger.info(f"Intent: {top_intent}, Confidence: {top_score:.2f}, Complex: {is_complex}")
        
        return {
            'intent': top_intent,
            'confidence': top_score,
            'all_scores': intent_scores,
            'is_complex': is_complex,
            'needs_clarification': top_score < 0.4 and top_intent != 'out_of_scope'
        }
    
    def _calculate_intent_scores(self, query: str) -> Dict[str, float]:
        """
        Calculate scores for all intents
        
        Returns AVERAGE keyword quality (total score / matched keywords)
        This measures the average strength of matched keywords, preventing
        verbose queries from being penalized and keyword stuffing from inflating scores.
        """
        scores = defaultdict(float)
        match_counts = defaultdict(int)
        query_lower = query.lower()
        
        # Strip punctuation and create word set for whole-word matching
        # Remove common punctuation to handle "copay?" → "copay"
        query_cleaned = re.sub(r'[^\w\s]', ' ', query_lower)
        query_words = set(query_cleaned.split())
        
        for intent, keywords in self.keyword_weights.items():
            for keyword, weight in keywords.items():
                # Use whole-word matching to avoid substring matches
                # (e.g., "hi" shouldn't match "this")
                if keyword in query_words:
                    scores[intent] += weight
                    match_counts[intent] += 1
            
            # Normalize by number of MATCHED keywords (average quality)
            if match_counts[intent] > 0:
                scores[intent] = scores[intent] / match_counts[intent]
        
        return dict(scores)
    
    def _is_simple_query(self, query: str, intent: str) -> bool:
        """
        Determine if query is simple (can be handled with pattern matching)
        
        Simple queries:
        - Have claim ID explicitly mentioned
        - Single intent, clear phrasing
        - No complex filters or aggregations
        """
        # Has explicit claim ID
        if re.search(r'\b(CLM|claim)\s*#?\s*\w+', query, re.IGNORECASE):
            return True
        
        # Short, simple queries
        if len(query.split()) <= 5 and intent in ['greeting', 'help']:
            return True
        
        return False
    
    def _is_complex_query(self, query: str, intent: str) -> bool:
        """
        Determine if query is complex (requires LLM processing)
        
        CRITICAL: Even if confidence is HIGH, complex queries should route to Master LLM!
        
        SMART ROUTING: "summary" keyword is now handled by keyword weights!
        - "pricing summary" → routes to pricing_info (via weights)
        - "summarize all my claims" → complex (aggregation)
        
        Complex queries contain:
        - Aggregations (summarize ALL, total, sum, average)
        - Comparisons (compare, versus, most expensive)
        - Explanations (explain how, why did)
        - Multiple conditions ("all claims in October over $100")
        
        SCANNING METHOD: Uses substring matching for multi-word phrases
        - Checks if entire phrase exists anywhere in query
        - E.g., 'summarize all' in 'please summarize all my claims' → True
        - Works for 1-word, 2-word, 3+ word phrases
        """
        query_lower = query.lower()
        
        # Aggregation keywords (truly complex)
        # Multi-word phrases: 'summarize all', 'all my', etc.
        aggregation_indicators = [
            'summarize all', 'summarize my', 'all my', 'all claims', 'every claim',
            'total', 'sum', 'average', 'count'
        ]
        
        for indicator in aggregation_indicators:
            if indicator in query_lower:  # Substring search for phrase
                return True
        
        # Comparison keywords (truly complex)
        # Multi-word phrases: 'more than', 'less than', 'difference between'
        comparison_indicators = [
            'compare', 'comparison', 'versus', 'vs', 'difference between',
            'most', 'least', 'expensive', 'cheapest',
            'more than', 'less than', 'greater', 'fewer',
            'highest', 'lowest', 'top', 'bottom'
        ]
        
        for indicator in comparison_indicators:
            if indicator in query_lower:  # Substring search for phrase
                return True
        
        # Explanation/reasoning keywords (truly complex)
        # Multi-word phrases: 'explain how', 'why did', 'what caused'
        explanation_indicators = [
            'explain how', 'explain why', 'how was', 'how did',
            'why did', 'why was', 'what caused', 'what led to',
            'which setup', 'which configuration', 'what can be done',
            'how to', 'what should'
        ]
        
        for indicator in explanation_indicators:
            if indicator in query_lower:  # Substring search for phrase
                return True
        
        # Very long queries are often complex
        query_words = query.split()
        if len(query_words) > 15:
            return True
        
        return False
    
    def get_supported_intents(self) -> List[str]:
        """Get list of all supported intents"""
        return list(self.keyword_weights.keys())
    
    def get_intent_description(self, intent: str) -> str:
        """Get description of what an intent covers"""
        descriptions = {
            # Core Intents
            'claim_status': 'General claim status inquiry (maps to: status, statusDescription)',
            'rejection_reasons': 'Why claim was rejected (maps to: statusDetails.rejectDetails[])',
            'drug_info': 'Medication/drug details (maps to: drug{}, submitted{})',
            'pharmacy_info': 'Where prescription was filled (maps to: pharmacy{})',
            'prescriber_info': 'Doctor/prescriber details (maps to: prescriber{})',
            'pricing_info': 'Cost/payment information (maps to: pricing.patientPay)',
            'rx_details': 'Prescription details (maps to: submitted.rxNumber, quantity, daysSupply)',
            'prior_auth_info': 'Prior authorization info (maps to: priorAuthorization{})',
            'beneficiary_info': 'Patient/member information (maps to: beneficiary{})',
            'fill_date_info': 'When prescription was filled (maps to: submitted.dateOfFill)',
            'approval_info': 'Approval messages (maps to: statusDetails.approvedMessages[])',
            'settlement_info': 'Settlement codes (maps to: statusDetails.settlementCodes[])',
            'date_range_claims': 'Claims in date range (filter on: submitted.dateOfFill)',
            'multi_claim_summary': 'Summary of multiple claims (maps to: multiple claimNumber requests)',
            
            # Additional CVS-Specific Intents
            'compound_info': 'Compound medication details (maps to: compound)',
            'medicare_part_d': 'Medicare Part D information (maps to: medD)',
            'daw_info': 'Brand vs generic dispensing (maps to: dispenseAsWritten, dawproductSelectionCode)',
            'cob_info': 'Coordination of benefits (maps to: cobClaimIndicator, linkedClaims.stcob)',
            'network_info': 'Pharmacy network status (maps to: pharmacyNetwork)',
            'reimbursement_info': 'How claim was reimbursed (maps to: reimbursementType)',
            'government_claim_type': 'Government program type (maps to: governmentClaimType)',
            'mail_order_info': 'Mail order pharmacy indicator (maps to: mail)',
            'generic_availability': 'Generic alternatives available (maps to: multiSourceInd)',
            'drug_interaction_info': 'Drug utilization review/interactions (maps to: durExistenceStatus)',
            'reversal_info': 'Claim reversal information (maps to: rnR)',
            'audit_info': 'Audit trail and history (maps to: audit{})',
            
            # Special Intents
            'greeting': 'User greeting',
            'help': 'User needs help',
            'appeal_info': 'Appeal/dispute information',
        }
        return descriptions.get(intent, 'No description available')


# ========== SINGLETON ==========

_cvs_intent_classifier = None


def get_cvs_intent_classifier() -> CVSIntentClassifier:
    """Get CVS intent classifier singleton"""
    global _cvs_intent_classifier
    
    if _cvs_intent_classifier is None:
        _cvs_intent_classifier = CVSIntentClassifier()
    
    return _cvs_intent_classifier

