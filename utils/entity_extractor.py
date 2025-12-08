"""
Entity Extractor
Extracts entities from user queries: claim IDs and claim sequences ONLY.
Uses regex patterns with validation.

SIMPLIFIED: Only extracts claim_ids and claim_sequences.
All other entity extraction has been removed as it's not used.
"""

import re
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts structured entities from natural language queries"""
    
    def __init__(self):
        # Entity patterns - ONLY claim_id and claim_sequence
        self.patterns = {
            # Claim ID: CLM prefix (CLM12345) OR pure numeric (253152732536005 - exactly 15 digits)
            'claim_id': r'\b(CLM\d{3,10}|\d{15})\b',
            # Claim sequence: 3 digits only (e.g., 001, 002, 003, 999)
            'claim_sequence': r'\b(\d{3})\b',
        }
        
        # Compile patterns
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }
    
    def extract(self, query: str) -> Dict[str, Any]:
        """
        Extract all entities from query
        Returns dict with extracted entities and validation status
        
        SIMPLIFIED: Only extracts claim_ids and claim_sequences.
        """
        entities = {}
        
        # Extract claim IDs and sequences
        entities.update(self._extract_ids(query))
        
        # Validate extracted entities
        validation = self._validate_entities(entities)
        
        return {
            'entities': entities,
            'validation': validation,
            'needs_validation': not validation['all_valid'],
            'missing_required': validation.get('missing', [])
        }
    
    def _extract_ids(self, query: str) -> Dict[str, Any]:
        """Extract claim IDs and claim sequences ONLY"""
        result = {}
        
        # Claim IDs (support multiple for batch queries)
        claim_ids = self.compiled_patterns['claim_id'].findall(query)
        if claim_ids:
            result['claim_ids'] = claim_ids
        
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
        
        return result
    
    def _validate_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted entities
        
        SIMPLIFIED: Only validates claim_ids and claim_sequences
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
        
        # Multiple claim sequences are AMBIGUOUS (error)
        claim_sequences = entities.get('claim_sequences', [])
        if len(claim_sequences) > 1:
            validation['all_valid'] = False
            validation['errors'].append({
                'type': 'multiple_entities',
                'entity': 'claim_sequences',
                'message': f"Multiple claim sequences found: {claim_sequences}. Please specify one."
            })
        
        return validation


# Global singleton
_entity_extractor = None


def get_entity_extractor() -> EntityExtractor:
    """Get global entity extractor instance"""
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = EntityExtractor()
    return _entity_extractor
