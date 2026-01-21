"""
Conversation Context Service

Handles entity extraction from conversation history for context-aware follow-up questions.

This service provides:
- Pattern-based entity extraction from conversation history
- Quick entity checks for routing decisions
- Support for claim IDs, member IDs, dates, and other pharmacy entities

Author: Created for context-aware follow-up question functionality
"""

import re
from typing import Dict, List, Any
from core.logger import get_logger

logger = get_logger(__name__)


class ConversationContextService:
    """Service for extracting entities from conversation history"""
    
    def extract_entities_from_history(
        self,
        conversation_history: List[Dict[str, str]],
        current_entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract entities from conversation history that might not be in current message.
        
        This enables follow-up questions without re-asking for information:
        - User: "What is status of claim 253152732536005?"
        - User: "How much did I pay?" ← Should use 253152732536005 from history
        
        Args:
            conversation_history: List of conversation messages
            current_entities: Entities already extracted from current message
            
        Returns:
            Merged entities dict (current entities take precedence over history)
        """
        if not conversation_history:
            logger.debug("No conversation history to extract from")
            return current_entities
        
        # Combine ONLY USER messages into text (exclude assistant messages)
        # CRITICAL: Including assistant messages can cause entity contamination from API responses
        history_text = " ".join([
            msg.get("content", "") 
            for msg in conversation_history 
            if isinstance(msg, dict) and msg.get("role") == "user"
        ])
        
        logger.debug(f"Extracting entities from {len(conversation_history)} messages ({len(history_text)} chars)")
        
        extracted = {}
        
        # ========================================================================
        # Pattern 1: Claim IDs - Context-aware detection
        # ========================================================================
        # Key principle: If "claim" keyword is present + digits → ALWAYS treat as claim ID
        # No rigid digit count restriction - claim IDs vary by system (8, 10, 12, 15, 18+ digits)
        # The keyword requirement prevents false positives (won't match random numbers)
        # 
        # Matches: "claim 253152732536005" (15 digits), "claim 12345" (5 digits), 
        #          "claim number 999888777" (9 digits), "CLM123456789012345" (any length)
        claim_pattern = r'(?:claim|clm)\s*(?:number|id|#)?\s*:?\s*(\d+)\b'
        claim_matches = re.findall(claim_pattern, history_text, re.IGNORECASE)
        
        if claim_matches:
            # Use longest match (real claim IDs are typically longer than codes)
            # Example: If both "12345" and "253152732536005" found, use the 15-digit one
            claim_matches_sorted = sorted(claim_matches, key=len, reverse=True)
            extracted["claim_number"] = claim_matches_sorted[0]
            logger.info(f"✓ Extracted claim_number from history: {extracted['claim_number']}")
        
        # ========================================================================
        # FALLBACK: Standalone 15-digit claim ID (no keyword required)
        # ========================================================================
        # Aligned with EntityExtractor pattern: r'\b(CLM\d{3,10}|\d{15})\b'
        # This catches user input like "253016267966353 - 1" without "claim" keyword
        # Safe: 15-digit numbers are highly specific, low false positive risk
        # Only runs if keyword-based extraction above found nothing
        if "claim_number" not in extracted:
            standalone_claim_pattern = r'\b(\d{15})\b'
            standalone_claim_matches = re.findall(standalone_claim_pattern, history_text)
            if standalone_claim_matches:
                # Use most recent 15-digit number (last in list)
                extracted["claim_number"] = standalone_claim_matches[-1]
                logger.info(f"✓ Extracted claim_number from history (standalone 15-digit): {extracted['claim_number']}")
        
        # ========================================================================
        # Pattern 2: Member IDs - MUST have "member"/"patient" keyword
        # ========================================================================
        # Matches: "member ID ABC123", "patient 78318GG3001"
        # Prevents matching city names, generic text
        member_pattern = r'(?:member|patient)\s*(?:id|number)?\s*:?\s*([A-Z0-9\-]{6,20})\b'
        member_matches = re.findall(member_pattern, history_text, re.IGNORECASE)
        
        if member_matches:
            extracted["member_id"] = member_matches[-1]  # Most recent
            logger.debug(f"✓ Extracted member_id from history: {extracted['member_id']}")
        
        # ========================================================================
        # Pattern 3: Dates (various formats)
        # ========================================================================
        # Matches: "01/15/2024", "2024-01-15", "1-15-24"
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b'
        date_matches = re.findall(date_pattern, history_text)
        
        if date_matches:
            extracted["date"] = date_matches[-1]  # Most recent
            logger.debug(f"✓ Extracted date from history: {extracted['date']}")
        
        # ========================================================================
        # Pattern 4: Prescription/RX IDs (if no claim ID found)
        # ========================================================================
        # Context-aware: if "prescription"/"rx" keyword + digits → prescription ID (any length)
        if "claim_number" not in extracted:
            rx_pattern = r'(?:prescription|rx)\s*(?:number|id|#)?\s*:?\s*(\d+)\b'
            rx_matches = re.findall(rx_pattern, history_text, re.IGNORECASE)
            
            if rx_matches:
                extracted["prescription_number"] = rx_matches[-1]
                logger.debug(f"✓ Extracted prescription_number from history: {extracted['prescription_number']}")
        
        # ========================================================================
        # Pattern 5: Sequence Numbers - ALIGNED WITH EntityExtractor
        # ========================================================================
        # Step 1: Try keyword-based extraction (with seq/sequence)
        # Matches: "sequence 001", "seq 001", "sequence number 001", "seq# 001"
        sequence_pattern = r'(?:seq(?:uence)?)\s*(?:number|num|#)?\s*:?\s*(\d{3})\b'
        sequence_matches = re.findall(sequence_pattern, history_text, re.IGNORECASE)
        
        # Step 2: FALLBACK - If no keyword match, try standalone 3-digit (like EntityExtractor)
        # This catches cases where user says "claim 260158058207352 and 001" without "seq" keyword
        if not sequence_matches:
            # Mask claim numbers (10+ digits) to avoid extracting sequences from them
            masked_history = re.sub(r'\d{10,}', 'CLAIM_MASKED', history_text)
            standalone_pattern = r'(?<!\d)\b(\d{3})\b(?!\d)'
            sequence_matches = re.findall(standalone_pattern, masked_history)
            if sequence_matches:
                logger.debug(f"   Found sequence via standalone fallback: {sequence_matches}")
        
        if sequence_matches:
            # Use the most recent sequence number
            extracted["claim_sequence"] = sequence_matches[-1]
            logger.info(f"✓ Extracted claim_sequence from history: {extracted['claim_sequence']}")
        
        # Log summary
        if extracted:
            logger.info(f"📦 Extracted {len(extracted)} entities from history: {list(extracted.keys())}")
        else:
            logger.debug("No entities extracted from history")
        
        # Merge: current entities take precedence over history
        merged = {**extracted, **current_entities}
        
        return merged
    
    def has_entities_in_history(
        self,
        conversation_history: List[Dict[str, str]],
        intent: str
    ) -> bool:
        """
        Quick check if conversation history contains entities needed for this intent.
        
        This is a lightweight check used by the router to decide:
        - If entities exist in history → route to build_context (full extraction)
        - If no entities anywhere → route to clarification (ask user)
        
        Args:
            conversation_history: List of messages
            intent: The classified intent
            
        Returns:
            True if history likely contains required entities
        """
        if not conversation_history:
            return False
        
        # FIXED: Use USER messages only (consistent with extract_entities_from_history)
        # Previously used ALL messages which caused router/extraction mismatch:
        # - Router would find "claim X" in assistant response → return True
        # - Extraction only looked at user messages → return {} (empty)
        # - Result: API call failed with "No entities provided"
        history_text = " ".join([
            msg.get("content", "") 
            for msg in conversation_history 
            if isinstance(msg, dict) and msg.get("role") == "user"
        ])
        
        # Check for claim ID (context-aware: if "claim" keyword + digits, treat as claim ID)
        # No rigid digit count - context matters more than pattern
        claim_pattern = r'(?:claim|clm)\s*(?:number|id|#)?\s*:?\s*\d+'
        has_claim_id = bool(re.search(claim_pattern, history_text, re.IGNORECASE))
        
        # ADDED: Also check for standalone 15-digit claim IDs (consistent with extraction)
        # This ensures router and extraction agree on what constitutes "entities in history"
        if not has_claim_id:
            standalone_claim_pattern = r'\b\d{15}\b'
            has_claim_id = bool(re.search(standalone_claim_pattern, history_text))
        
        # Check for dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        has_date = bool(re.search(date_pattern, history_text))
        
        # Intent-specific checks
        # These intents require claim ID
        if intent in ['claim_status', 'rejection_reasons', 'pricing_info', 
                      'drug_info', 'pharmacy_info', 'prescriber_info', 
                      'fill_date_info', 'rx_details', 'prior_auth_info',
                      'approval_info', 'settlement_info', 'reversal_info',
                      'compound_info', 'medicare_part_d', 'mail_order_info',
                      'generic_availability', 'daw_info', 'cob_info',
                      'network_info', 'reimbursement_info', 'government_claim_type',
                      'drug_interaction_info', 'audit_info', 'beneficiary_info']:
            return has_claim_id
        
        # These intents require dates
        if intent in ['date_range_claims']:
            return has_date
        
        # Default: if we found any entity pattern, consider it sufficient
        return has_claim_id or has_date


# ============================================================================
# MODULE-LEVEL FUNCTIONS (SINGLETON PATTERN)
# ============================================================================

_service_instance = None


def get_conversation_service() -> ConversationContextService:
    """
    Get global conversation context service instance (singleton)
    
    Returns:
        ConversationContextService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = ConversationContextService()
        logger.info("🔧 ConversationContextService initialized")
    return _service_instance


def extract_entities_from_history(
    conversation_history: List[Dict[str, str]],
    current_entities: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function: Extract entities from conversation history
    
    Args:
        conversation_history: List of messages
        current_entities: Entities from current message
        
    Returns:
        Merged entities
    """
    service = get_conversation_service()
    return service.extract_entities_from_history(conversation_history, current_entities)


def has_entities_in_history(
    conversation_history: List[Dict[str, str]],
    intent: str
) -> bool:
    """
    Convenience function: Quick check if history has required entities
    
    Args:
        conversation_history: List of messages
        intent: The classified intent
        
    Returns:
        True if history contains required entities
    """
    service = get_conversation_service()
    return service.has_entities_in_history(conversation_history, intent)

