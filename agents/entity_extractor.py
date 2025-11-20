"""
Entity Extractor
Extracts entities from user queries: claim IDs, member IDs, dates, amounts, etc.
Uses regex patterns with validation
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import calendar
import logging

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts structured entities from natural language queries"""
    
    def __init__(self):
        # Entity patterns
        self.patterns = {
            'claim_id': r'\b(CLM\d{3,10})\b',
            'member_id': r'\b(MEM\d{3,10})\b',
            'prescription_id': r'\b(RX\d{3,10})\b',
            'amount': r'\$?\d+(?:\.\d{2})?',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        }
        
        # Compile patterns
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }
        
        # Month names for date extraction
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
        
        # Extract IDs
        entities.update(self._extract_ids(query))
        
        # Extract dates
        date_info = self._extract_dates(query_lower)
        if date_info:
            entities.update(date_info)
        
        # Extract amounts
        amounts = self._extract_amounts(query)
        if amounts:
            entities['amounts'] = amounts
        
        # Validate extracted entities
        validation = self._validate_entities(entities)
        
        return {
            'entities': entities,
            'validation': validation,
            'needs_validation': not validation['all_valid'],
            'missing_required': validation.get('missing', [])
        }
    
    def _extract_ids(self, query: str) -> Dict[str, Any]:
        """Extract claim IDs, member IDs, prescription IDs"""
        result = {}
        
        # Claim IDs (support multiple for batch queries)
        claim_ids = self.compiled_patterns['claim_id'].findall(query)
        if claim_ids:
            result['claim_ids'] = claim_ids
            result['claim_id'] = claim_ids[0]  # Primary claim ID
            result['has_multiple_claims'] = len(claim_ids) > 1  # Flag for multi-claim handling
        
        # Member IDs
        member_ids = self.compiled_patterns['member_id'].findall(query)
        if member_ids:
            result['member_ids'] = member_ids
            result['member_id'] = member_ids[0]  # Primary member ID
            if len(member_ids) > 1:
                result['multiple_member_ids'] = True
        
        # Prescription IDs
        prescription_ids = self.compiled_patterns['prescription_id'].findall(query)
        if prescription_ids:
            result['prescription_ids'] = prescription_ids
            result['prescription_id'] = prescription_ids[0]  # Primary prescription ID
            if len(prescription_ids) > 1:
                result['multiple_prescription_ids'] = True
        
        return result
    
    def _extract_dates(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract dates and date ranges from query"""
        result = {}
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Check for month names
        for month_name, month_num in self.months.items():
            if month_name in query:
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
                result['needs_confirmation'] = True
                result['inferred_year'] = True
                break
        
        # Check for relative dates
        if 'yesterday' in query:
            yesterday = datetime.now() - timedelta(days=1)
            result['date_range'] = {
                'start': yesterday.replace(hour=0, minute=0, second=0).isoformat(),
                'end': yesterday.replace(hour=23, minute=59, second=59).isoformat(),
                'display': 'Yesterday'
            }
        
        elif 'last week' in query:
            today = datetime.now()
            last_week_start = today - timedelta(days=7)
            result['date_range'] = {
                'start': last_week_start.isoformat(),
                'end': today.isoformat(),
                'display': 'Last 7 days'
            }
        
        elif 'last month' in query:
            today = datetime.now()
            last_month = today.replace(day=1) - timedelta(days=1)
            result['date_range'] = {
                'start': last_month.replace(day=1).isoformat(),
                'end': last_month.replace(day=calendar.monthrange(last_month.year, last_month.month)[1]).isoformat(),
                'display': f"{calendar.month_name[last_month.month]} {last_month.year}"
            }
        
        # Check for quarters
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
            result['needs_confirmation'] = True
        
        return result if result else None
    
    def _extract_amounts(self, query: str) -> Optional[List[float]]:
        """Extract dollar amounts from query"""
        amounts = []
        
        # Find all amount patterns
        amount_pattern = r'\$?(\d+(?:\.\d{2})?)'
        matches = re.finditer(amount_pattern, query)
        
        for match in matches:
            try:
                amount = float(match.group(1))
                # Filter out obvious non-amounts (like years or IDs)
                if 1 <= amount <= 100000:  # Reasonable claim amounts
                    amounts.append(amount)
            except ValueError:
                continue
        
        return amounts if amounts else None
    
    def _validate_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted entities"""
        validation = {
            'all_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Multiple claim IDs are VALID for batch queries (CVS API supports this!)
        if entities.get('has_multiple_claims'):
            validation['warnings'].append({
                'type': 'multiple_claims',
                'entity': 'claim_id',
                'count': len(entities.get('claim_ids', [])),
                'message': f"Processing {len(entities.get('claim_ids', []))} claims: {entities.get('claim_ids', [])}"
            })
        
        # Multiple member IDs are AMBIGUOUS (error)
        if entities.get('multiple_member_ids'):
            validation['all_valid'] = False
            validation['errors'].append({
                'type': 'multiple_entities',
                'entity': 'member_id',
                'message': f"Multiple member IDs found: {entities['member_ids']}. Please specify one."
            })
        
        # Check for date confirmation needed
        if entities.get('needs_confirmation'):
            validation['warnings'].append({
                'type': 'needs_confirmation',
                'entity': 'date_range',
                'message': f"Please confirm date range: {entities.get('date_range', {}).get('display')}"
            })
        
        return validation
    
    def extract_required_slots(self, intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if all required slots are present for given intent
        Returns missing slots and validation status
        """
        # Define required slots per intent
        required_slots_map = {
            'claim_status': ['claim_id'],
            'claim_details': ['claim_id'],
            'rejection_reasons': ['claim_id'],
            'reversal_info': ['claim_id'],
            'claim_pending': ['claim_id'],
            'prescription_info': ['prescription_id'],
            'prescription_status': ['prescription_id'],
            'member_info': ['member_id'],
            'benefits_info': ['member_id']
        }
        
        required = required_slots_map.get(intent, [])
        missing = []
        
        for slot in required:
            if slot not in entities or entities[slot] is None:
                missing.append(slot)
        
        return {
            'required_slots': required,
            'missing_slots': missing,
            'has_all_slots': len(missing) == 0
        }
    
    def format_entity_for_api(self, entity_name: str, entity_value: Any) -> str:
        """Format entity value for API parameter"""
        if entity_name.endswith('_id'):
            # Ensure uppercase for IDs
            return str(entity_value).upper()
        
        if entity_name == 'date_range':
            # Return ISO format dates
            if isinstance(entity_value, dict):
                return entity_value
        
        return str(entity_value)


# Global singleton
_entity_extractor = None


def get_entity_extractor() -> EntityExtractor:
    """Get global entity extractor instance"""
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = EntityExtractor()
    return _entity_extractor

