"""
Entity Extractor
Extracts entities from user queries: claim IDs, claim sequences, member IDs, 
prescription IDs, and date ranges.
Uses regex patterns with validation.

Based on MVP-1 simplified version with essential additions for response_agent compatibility.
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import calendar
import logging

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts structured entities from natural language queries"""
    
    def __init__(self):
        # Entity patterns
        self.patterns = {
            # Claim ID: CLM prefix (CLM12345) OR pure numeric (253152732536005 - exactly 15 digits)
            'claim_id': r'\b(CLM\d{3,10}|\d{15})\b',
            
            # Claim sequence: 3 digits only (e.g., 001, 002, 003, 999)
            'claim_sequence': r'\b(\d{3})\b',
            
            # Liberal Claim ID: Captures numbers (4+ digits) after "claim" keyword
            # Purpose: Better user experience - detect user intent even with invalid format
            # Examples: "claim 1234", "claim summary of 12345", "claim details for 99999"
            # Pattern allows words between "claim" and digits for natural language
            # Required by: response_agent.py (lines 493-506) for contextual error messages
            'potential_claim_id': r'(?:claim|clm|claims)\s+(?:\w+\s+)*?(\d{4,})\b',
            
            # Member ID: Flexible pattern supporting various lengths (3-10 digits)
            # Required by: response_agent.py (line 431) for member-related queries
            'member_id': r'\b(MEM\d{3,10})\b',
            
            # Prescription ID: RX prefix + 3-10 digits
            # Required by: response_agent.py (line 435) for prescription-related queries
            'prescription_id': r'\b(RX\d{3,10})\b',
        }
        
        # Compile patterns for performance
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }
        
        # Month names for date extraction
        # Required by: response_agent.py (line 440) for date_range queries
        self.months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
    
    def extract(self, query: str) -> Dict[str, Any]:
        """
        Extract all entities from query
        Returns dict with extracted entities and validation status
        """
        query_lower = query.lower()
        
        entities = {}
        
        # Extract IDs (claim, member, prescription)
        entities.update(self._extract_ids(query))
        
        # Extract dates (months, relative dates, quarters)
        # Required by: response_agent.py (line 440)
        date_info = self._extract_dates(query_lower)
        if date_info:
            entities.update(date_info)
        
        # Validate extracted entities
        validation = self._validate_entities(entities)
        
        return {
            'entities': entities,
            'validation': validation,
            'needs_validation': not validation['all_valid'],
            'missing_required': validation.get('missing', [])
        }
    
    def _extract_ids(self, query: str) -> Dict[str, Any]:
        """Extract claim IDs, claim sequences, member IDs, and prescription IDs"""
        result = {}
        
        # Claim IDs (support multiple for batch queries)
        claim_ids = self.compiled_patterns['claim_id'].findall(query)
        if claim_ids:
            result['claim_ids'] = claim_ids
            logger.debug(f"Extracted claim IDs: {claim_ids}")
        else:
            # If no exact claim IDs found, check for potential claim IDs
            # This enables better error messages: "please provide a VALID claim ID"
            # instead of: "please provide claim ID"
            # Required by: response_agent.py (lines 493-506)
            potential_ids = self.compiled_patterns['potential_claim_id'].findall(query)
            if potential_ids:
                result['potential_claim_ids'] = potential_ids
                result['claim_id_format_invalid'] = True
                logger.debug(f"Found potential claim IDs with invalid format: {potential_ids}")
        
        # Claim Sequences - SMART EXTRACTION
        # Step 1: Try context-aware extraction (with keywords)
        context_pattern = re.compile(r'(?:sequence|seq|line)\s*(\d{3})\b', re.IGNORECASE)
        claim_sequences = context_pattern.findall(query)
        
        # Step 2: If no context-aware match, try standalone 3-digit NOT part of longer number
        if not claim_sequences:
            # Create a query with claim IDs masked to avoid extracting from them
            masked_query = query
            if claim_ids:
                for claim_id in claim_ids:
                    masked_query = masked_query.replace(claim_id, 'CLAIM_MASKED')
            
            # Now extract 3-digit sequences that aren't part of longer numbers
            standalone_pattern = re.compile(r'(?<!\d)\b(\d{3})\b(?!\d)')
            claim_sequences = standalone_pattern.findall(masked_query)
        
        if claim_sequences:
            result['claim_sequences'] = claim_sequences
            logger.debug(f"Extracted claim sequences: {claim_sequences}")
        
        # Member IDs - Required by: response_agent.py (line 431)
        member_ids = self.compiled_patterns['member_id'].findall(query)
        if member_ids:
            result['member_ids'] = member_ids
            logger.debug(f"Extracted member IDs: {member_ids}")
        
        # Prescription IDs - Required by: response_agent.py (line 435)
        prescription_ids = self.compiled_patterns['prescription_id'].findall(query)
        if prescription_ids:
            result['prescription_ids'] = prescription_ids
            logger.debug(f"Extracted prescription IDs: {prescription_ids}")
        
        return result
    
    def _extract_dates(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extract dates and date ranges from query
        Required by: response_agent.py (line 440)
        
        Supports:
        - Month names (january, feb, etc.)
        - Relative dates (yesterday, last week, last month)
        - Quarters (Q1, Q2, Q3, Q4)
        
        Note: If multiple date patterns match, the last match takes precedence
        """
        result = {}
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Check for month names (use word boundaries to avoid false positives)
        # Example: "summary" contains "mar" but shouldn't match March
        for month_name, month_num in self.months.items():
            # Use regex with word boundaries to match standalone month names only
            month_pattern = r'\b' + re.escape(month_name) + r'\b'
            if re.search(month_pattern, query):
                # Infer year
                if month_num > current_month:
                    year = current_year - 1  # Past month, likely last year
                else:
                    year = current_year
                
                start_date = datetime(year, month_num, 1)
                end_date = datetime(year, month_num, calendar.monthrange(year, month_num)[1])
                
                result['date_range'] = {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'display': f"{month_name.capitalize()} {year}"
                }
                logger.debug(f"Extracted month date range: {result['date_range']['display']}")
                break
        
        # Check for relative dates (overrides month if present)
        if 'yesterday' in query:
            yesterday = datetime.now() - timedelta(days=1)
            result['date_range'] = {
                'start': yesterday.replace(hour=0, minute=0, second=0).isoformat(),
                'end': yesterday.replace(hour=23, minute=59, second=59).isoformat(),
                'display': 'Yesterday'
            }
            logger.debug("Extracted relative date: Yesterday")
        
        elif 'last week' in query:
            today = datetime.now()
            last_week_start = today - timedelta(days=7)
            result['date_range'] = {
                'start': last_week_start.isoformat(),
                'end': today.isoformat(),
                'display': 'Last 7 days'
            }
            logger.debug("Extracted relative date: Last week")
        
        elif 'last month' in query:
            today = datetime.now()
            last_month = today.replace(day=1) - timedelta(days=1)
            result['date_range'] = {
                'start': last_month.replace(day=1).isoformat(),
                'end': last_month.replace(day=calendar.monthrange(last_month.year, last_month.month)[1]).isoformat(),
                'display': f"{calendar.month_name[last_month.month]} {last_month.year}"
            }
            logger.debug(f"Extracted relative date: {result['date_range']['display']}")
        
        # Check for quarters (overrides other dates if present)
        quarter_match = re.search(r'\bq([1-4])\b', query)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            start_month = (quarter - 1) * 3 + 1
            end_month = quarter * 3
            
            result['date_range'] = {
                'start': datetime(current_year, start_month, 1).isoformat(),
                'end': datetime(current_year, end_month, calendar.monthrange(current_year, end_month)[1]).isoformat(),
                'display': f"Q{quarter} {current_year}"
            }
            logger.debug(f"Extracted quarter date range: {result['date_range']['display']}")
        
        return result if result else None
    
    def _validate_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted entities
        
        Rules:
        - Multiple claim IDs: VALID (for batch queries)
        - Multiple member IDs: INVALID (ambiguous)
        - Multiple claim sequences: INVALID (ambiguous)
        """
        validation = {
            'all_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Multiple claim IDs are VALID for batch queries (CVS API supports this!)
        claim_ids = entities.get('claim_ids', [])
        if len(claim_ids) > 1:
            validation['warnings'].append({
                'type': 'multiple_claims',
                'entity': 'claim_ids',
                'count': len(claim_ids),
                'message': f"Processing {len(claim_ids)} claims: {claim_ids}"
            })
            logger.info(f"Batch query detected: {len(claim_ids)} claim IDs")
        
        # Multiple member IDs are AMBIGUOUS (error)
        member_ids = entities.get('member_ids', [])
        if len(member_ids) > 1:
            validation['all_valid'] = False
            validation['errors'].append({
                'type': 'multiple_entities',
                'entity': 'member_ids',
                'message': f"Multiple member IDs found: {member_ids}. Please specify one."
            })
            logger.warning(f"Ambiguous query: multiple member IDs detected: {member_ids}")
        
        # Multiple claim sequences are AMBIGUOUS (error)
        claim_sequences = entities.get('claim_sequences', [])
        if len(claim_sequences) > 1:
            validation['all_valid'] = False
            validation['errors'].append({
                'type': 'multiple_entities',
                'entity': 'claim_sequences',
                'message': f"Multiple claim sequences found: {claim_sequences}. Please specify one."
            })
            logger.warning(f"Ambiguous query: multiple claim sequences detected: {claim_sequences}")
        
        return validation
    
    def extract_required_slots(self, intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if all required slots are present for given intent
        Returns missing slots and validation status
        
        DEPRECATED: This method was part of System 1 (old slot-filling approach).
        The new system (System 2) uses api_routing_config.py for required entity validation.
        See: extended_intent_agent_node.py -> get_api_config(intent)
        
        Kept for backward compatibility and potential future use.
        """
        logger.debug(f"[DEPRECATED] extract_required_slots called for intent: {intent}")
        
        # Define required slots per intent (using list format)
        required_slots_map = {
            'claim_status': ['claim_ids'],
            'claim_details': ['claim_ids'],
            'rejection_reasons': ['claim_ids'],
            'reversal_info': ['claim_ids'],
            'claim_pending': ['claim_ids'],
            'prescription_info': ['prescription_ids'],
            'prescription_status': ['prescription_ids'],
            'member_info': ['member_ids'],
            'benefits_info': ['member_ids']
        }
        
        required = required_slots_map.get(intent, [])
        missing = []
        
        for slot in required:
            # Check if slot exists and has at least one value
            if slot not in entities or not entities[slot] or len(entities[slot]) == 0:
                missing.append(slot)
        
        return {
            'required_slots': required,
            'missing_slots': missing,
            'has_all_slots': len(missing) == 0
        }
    
    def format_entity_for_api(self, entity_name: str, entity_value: Any) -> str:
        """
        Format entity value for API parameter
        
        Rules:
        - IDs: Uppercase (CLM123 -> CLM123, mem456 -> MEM456)
        - Date ranges: Return as-is (dict format)
        - Others: String conversion
        
        Reserved for future use in API integration layer.
        """
        if entity_name.endswith('_id'):
            # Ensure uppercase for IDs
            return str(entity_value).upper()
        
        if entity_name == 'date_range':
            # Return ISO format dates as-is
            if isinstance(entity_value, dict):
                return entity_value
        
        return str(entity_value)


# Global singleton
_entity_extractor = None


def get_entity_extractor() -> EntityExtractor:
    """Get global entity extractor instance (singleton pattern)"""
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = EntityExtractor()
        logger.info("Entity extractor initialized")
    return _entity_extractor
