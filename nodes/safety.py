"""
Safety Nodes - Unified Safety Check with PII Protection

ARCHITECTURE:
    User Query
        ↓
    [safety_precheck_node] ← Unified safety check
        ├─ Violence pattern check
        ├─ Mask PII/PHI
        ├─ Gemini filters (on masked data)
        └─ Unmask PII/PHI
    Returns: Safe query with PII/PHI intact
        ↓
    ... (cache, context, intent_agent, tool_calls) ...
        ↓
    [response_safety_pii_precheck_node] ← Mask before response LLM
        ↓
    [response_agent] ← LLM (works with masked data)
        ↓
    [response_safety_pii_postcheck_node] ← Unmask for user
"""

import re
import traceback
import time
from typing import Dict, Any
from langgraph.graph import END
from state.schema import AgentState
from config.config import settings
from core.logger import get_logger
from core.logging_context import extract_logging_context, log_state_snapshot
from core.node_models import SafetyResult, SafetyCheckType, SafetyViolationType, create_safety_result
from core.errors.models import create_safety_error, create_internal_error
from core.telemetry import log_event
from utils.serialization import to_dict
from persistence import PersistenceStoreFactory, EventType
from services.pii_protection import (
    get_safety_checker,
    get_pii_service
)

logger = get_logger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _is_masked_token(text: str) -> bool:
    """
    Check if text is a masked PII/PHI token like [CLAIM_ID_ABC123]
    
    Tokens have the format: [ENTITY_TYPE_HEXHASH]
    Examples: [CLAIM_ID_5460C7F2], [PERSON_348253B6], [NDC_F80D0950]
    
    Args:
        text: Text to check
        
    Returns:
        bool: True if text matches token pattern
    """
    import re
    # Match: [UPPERCASE_LETTERS_HEXDIGITS]
    # Allow underscores in entity type (e.g., US_DRIVER_LICENSE)
    return bool(re.match(r'^\[[A-Z_]+_[A-F0-9]+\]$', text.strip()))


def _is_contextual_entity(entity_text: str, entity_type: str) -> bool:
    """
    Determine if a detected PII entity is contextual data vs actual leakage.
    
    COMPREHENSIVE FALSE POSITIVE PREVENTION for pharmacy claims domain.
    
    This function is the PRIMARY defense against false positive PII leakage detection.
    It allows common contextual patterns that appear in legitimate responses:
    - Drug names, dosages, and forms (500MG TAB, Atorvastatin 40mg)
    - Response section headers (SUMMARY, FINANCIAL, DRUG, MEMBER)
    - Table headers and labels (Category, Before, After, Remaining)
    - Pharmacy terms (COPAY, DEDUCTIBLE, accumulation)
    - System codes and field names (date8, APNCP321, submittedProductId)
    - Numeric patterns (dates, IDs, prices, NPI numbers)
    - Generic location and medical terms
    
    Args:
        entity_text: The detected PII text
        entity_type: The type of PII entity (PERSON, LOCATION, etc.)
        
    Returns:
        True if this is likely contextual data (NOT a leak)
        False if this is potential real PII leakage
    """
    import re
    
    # Normalize for comparison
    text_lower = entity_text.lower()
    text_upper = entity_text.upper()
    text_stripped = entity_text.strip()
    
    # ================================================================
    # UNIVERSAL FILTERS (apply to ALL entity types first)
    # ================================================================
    
    # 1. Very short text (≤3 chars) - almost never real PII (preserves original behavior)
    if len(text_stripped) <= 3:
        return True
    
    # 2. All whitespace or empty
    if not text_stripped:
        return True
    
    # 3. Contains only special characters (bullet points, symbols)
    if re.match(r'^[\s•\-\*\|\:\;\,\.]+$', text_stripped):
        return True
    
    # 4. Placeholder/null values (all zeros, all nines, sequential)
    placeholder_patterns = [
        r'^0+$',                    # 000000000
        r'^9+$',                    # 999999999
        r'^1234567890?$',           # 1234567890, 123456789
        r'^0?987654321$',           # 987654321
        r'^1{3,}$',                 # 111111...
        r'^(null|none|n/a|na|tbd|unknown|undefined)$',  # Null values
    ]
    for pattern in placeholder_patterns:
        if re.match(pattern, text_lower):
            return True
    
    # 5. Drug dosage patterns (CRITICAL: fixes "500MG TAB" false positive)
    # Matches: 500MG, 40MG TAB, 10MCG CAP, 100ML SOLUTION, etc.
    drug_dosage_patterns = [
        r'^\d+\.?\d*\s*(mg|mcg|ml|g|iu|units?|meq)\b',           # Starts with dosage
        r'\b\d+\.?\d*\s*(mg|mcg|ml|g|iu|units?|meq)\b',           # Contains dosage
        r'\b(tab|tablet|tablets|cap|caps|capsule|capsules)\b',    # Dosage forms
        r'\b(sol|solution|susp|suspension|inj|injection)\b',      # Liquid forms
        r'\b(cream|oint|ointment|gel|patch|spray|drops)\b',       # Topical forms
        r'\b(syrup|powder|granules?|lozenge|suppository)\b',      # Other forms
        r'\b(extended|delayed|controlled)\s*release\b',           # Release types
        r'\b(immediate|sustained|modified)\s*release\b',          # Release types
        r'\b(oral|topical|injectable|inhaled|nasal|ophthalmic)\b',  # Routes
    ]
    for pattern in drug_dosage_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # 6. YYYYMMDD date format (8 digits starting with 19 or 20)
    if re.match(r'^(19|20)\d{6}$', text_stripped):
        return True
    
    # 7. Decimal numbers (prices, measurements, coordinates)
    if re.match(r'^\d+\.\d+$', text_stripped):
        return True
    
    # 8. Dollar amounts
    if re.match(r'^\$?\d+\.?\d*$', text_stripped):
        return True
    
    # 9. Contains underscore (field names, system codes, token refs)
    if '_' in text_stripped:
        return True
    
    # 10. Contains hashtag with number (#1234)
    if re.match(r'^#\d+$', text_stripped):
        return True
    
    # ================================================================
    # PERSON ENTITY - Most common false positive type
    # ================================================================
    if entity_type == "PERSON":
        
        # --- Response Format Terms ---
        # Section headers from pharmacy claims responses
        response_headers = {
            'summary', 'financial', 'drug', 'member', 'pharmacy', 'rejection',
            'pricing', 'accumulation', 'next steps', 'details', 'information',
            'status', 'overview', 'history', 'notes', 'comments',
        }
        if text_lower in response_headers:
            return True
        
        # Table headers and labels
        table_terms = {
            'category', 'before', 'after', 'remaining', 'individual', 'family',
            'this claim', 'total', 'amount', 'value', 'description', 'type',
            'date', 'code', 'message', 'quantity', 'days', 'supply',
        }
        if text_lower in table_terms:
            return True
        
        # --- Pharmacy/Medical Terms ---
        pharmacy_terms = {
            # Financial terms
            'copay', 'copayment', 'coinsurance', 'deductible', 'premium',
            'accumulation', 'out-of-pocket', 'oop', 'moop', 'patient pay',
            'plan paid', 'patient paid', 'balance', 'cost', 'price',
            # Status terms  
            'paid', 'denied', 'rejected', 'reversed', 'pending', 'approved',
            'processed', 'submitted', 'completed', 'in process', 'on hold',
            # Coverage terms
            'primary', 'secondary', 'tertiary', 'generic', 'brand', 'specialty',
            'retail', 'mail', 'mail order', 'maintenance', 'formulary',
            # Role terms
            'pharmacy', 'pharmacist', 'prescriber', 'patient', 'member',
            'doctor', 'physician', 'nurse', 'provider', 'caregiver',
            'cardholder', 'dependent', 'subscriber', 'beneficiary',
            # Descriptor terms
            'male', 'female', 'adult', 'child', 'senior', 'minor',
            # Rejection/action terms
            'refill', 'refill too soon', 'previous fill', 'next fill',
            'prior authorization', 'step therapy', 'quantity limit',
            'days supply', 'early refill', 'coverage', 'not covered',
        }
        if text_lower in pharmacy_terms:
            return True
        
        # Contains pharmacy keyword
        pharmacy_keywords = ['pharmacy', 'pharm', 'drug', 'rx', 'prescription', 'medication']
        if any(kw in text_lower for kw in pharmacy_keywords):
            return True
        
        # --- Drug Name Patterns (Generic drug suffixes) ---
        # Common drug name endings that indicate pharmaceutical names
        drug_suffixes = (
            'statin', 'pril', 'olol', 'sartan', 'dipine', 'prazole', 'tidine',
            'cillin', 'mycin', 'cycline', 'floxacin', 'azole', 'sone', 'olone',
            'mab', 'nib', 'afil', 'formin', 'gliptin', 'gliflozin', 'tide',
            'semide', 'thiazide', 'pam', 'lam', 'zepam', 'done', 'idol',
            'profen', 'oxicam', 'triptan', 'setron', 'lukast', 'phylline',
            'dronate', 'vaptan', 'oxetine', 'racetam', 'barb', 'caine',
        )
        if any(text_lower.endswith(suffix) for suffix in drug_suffixes):
            return True
        
        # --- Common Drug Names (that don't follow suffix patterns) ---
        common_drugs = {
            'metformin', 'lisinopril', 'atorvastatin', 'amlodipine', 'metoprolol',
            'omeprazole', 'losartan', 'gabapentin', 'hydrochlorothiazide', 'sertraline',
            'simvastatin', 'montelukast', 'escitalopram', 'rosuvastatin', 'bupropion',
            'pantoprazole', 'duloxetine', 'tamsulosin', 'carvedilol', 'trazodone',
            'meloxicam', 'pravastatin', 'clopidogrel', 'potassium', 'prednisone',
            'albuterol', 'insulin', 'warfarin', 'aspirin', 'ibuprofen', 'acetaminophen',
            'levothyroxine', 'furosemide', 'alprazolam', 'hydrocodone', 'tramadol',
            'amoxicillin', 'azithromycin', 'ciprofloxacin', 'metronidazole', 'doxycycline',
            'liothyronine', 'synthroid', 'lipitor', 'norvasc', 'zoloft', 'lexapro',
            'xanax', 'vicodin', 'percocet', 'oxycodone', 'morphine', 'fentanyl',
        }
        if text_lower in common_drugs:
            return True
        
        # --- Pharmacy Chain Names ---
        pharmacy_chains = {
            'cvs', 'walgreens', 'walmart', 'rite aid', 'riteaid', 'kroger', 'publix',
            'costco', 'target', 'safeway', 'albertsons', 'wegmans', 'heb', 'meijer',
            'sams club', "sam's club", 'amazon', 'express scripts', 'caremark',
            'optumrx', 'humana pharmacy', 'cigna pharmacy', 'wellcare', 'centene',
            'rite-aid', 'duane reade', 'kinney drugs', 'bartell', 'fred meyer',
            'giant eagle', 'stop & shop', 'stop and shop', 'shoprite', 'winn-dixie',
        }
        if text_lower in pharmacy_chains:
            return True
        # Partial match for pharmacy chains (e.g., "WALGREENS PHARMACY #12345")
        chain_keywords = ['walgreens', 'cvs', 'rite aid', 'walmart', 'costco', 'kroger']
        if any(chain in text_lower for chain in chain_keywords):
            return True
        
        # --- Insurance/Plan Company Names ---
        insurance_names = {
            'aetna', 'cigna', 'humana', 'united', 'anthem', 'kaiser', 'bcbs',
            'blue cross', 'blue shield', 'medicare', 'medicaid', 'tricare',
            'wellcare', 'centene', 'molina', 'ambetter', 'oscar', 'clover',
            'devoted', 'alignment', 'devoted health', 'scan', 'unitedhealthcare',
            'unitedhealth', 'emblem', 'emblemhealth', 'healthfirst', 'fidelis',
            'amerihealth', 'carefirst', 'highmark', 'premera', 'regence', 'cambia',
            'geisinger', 'upmc', 'sentara', 'priority health', 'selecthealth',
        }
        if text_lower in insurance_names:
            return True
        
        # --- Drug Manufacturer Names ---
        manufacturers = {
            'pfizer', 'merck', 'teva', 'mylan', 'sandoz', 'novartis', 'abbvie',
            'lilly', 'eli lilly', 'amgen', 'bristol', 'bms', 'johnson', 'jnj',
            'glaxo', 'gsk', 'glaxosmithkline', 'astrazeneca', 'roche', 'sanofi',
            'bayer', 'boehringer', 'takeda', 'regeneron', 'gilead', 'biogen',
            'vertex', 'moderna', 'alexion', 'aurobindo', 'lupin', 'dr reddy',
            'sun pharma', 'cipla', 'apotex', 'zydus', 'torrent', 'hikma',
            'viatris', 'perrigo', 'amneal', 'par pharmaceutical', 'actavis',
            'watson', 'mallinckrodt', 'endo', 'allergan', 'bausch', 'valeant',
        }
        if text_lower in manufacturers:
            return True
        
        # --- Medical Titles (often appear before/after names in data) ---
        medical_titles = {
            'dr', 'md', 'do', 'pharmd', 'rph', 'np', 'pa', 'rn', 'lpn', 'cna',
            'dds', 'dpm', 'od', 'dc', 'dpt', 'aprn', 'fnp', 'cnp', 'crna',
            'physician', 'pharmacist', 'technician', 'intern', 'resident',
        }
        if text_lower in medical_titles:
            return True
        
        # --- Rejection/Claim Message Keywords ---
        rejection_keywords = {
            'formulary', 'non-formulary', 'nonformulary', 'authorization',
            'therapy', 'quantity', 'limit', 'refill', 'covered', 'coverage',
            'ndc', 'prescriber', 'cardholder', 'recognized', 'daw', 'compound',
            'dur', 'interaction', 'override', 'reversal', 'adjudication',
            'adjudicated', 'dispensed', 'dispensing', 'billing', 'processing',
            'eligibility', 'eligible', 'ineligible', 'active', 'inactive',
            'terminated', 'effective', 'verification', 'validated', 'invalid',
        }
        if text_lower in rejection_keywords:
            return True
        
        # --- Field Name Patterns ---
        # camelCase: groupNumber, memberId, claimStatus
        if re.match(r'^[a-z]+[A-Z][a-zA-Z]*$', text_stripped):
            return True
        
        # Field name with number at end: date8, message5, test4
        if re.match(r'^[a-z]+\d+$', text_stripped):
            return True
        
        # CONSTANT_CASE or contains underscore
        if '_' in text_stripped:
            return True
        
        # --- System Code Patterns ---
        # All-caps alphanumeric codes: GML001, APNCP321, KKPLANP001
        if text_stripped.isupper() and re.match(r'^[A-Z0-9]+$', text_stripped):
            has_letters = any(c.isalpha() for c in text_stripped)
            has_numbers = any(c.isdigit() for c in text_stripped)
            if has_letters and has_numbers:
                return True
        
        # All-caps single word (4+ chars) without space - likely code/term
        if text_stripped.isupper() and len(text_stripped) >= 4 and ' ' not in text_stripped:
            return True
        
        # Letters followed by numbers or vice versa (codes)
        if re.match(r'^[A-Z]{2,6}\d{2,6}$', text_stripped):
            return True
        if re.match(r'^\d{2,6}[A-Z]{2,6}$', text_stripped):
            return True
        
        # --- Numeric Mixed Patterns ---
        # Contains digits (real names don't have digits)
        if re.search(r'\d', text_stripped):
            return True
        
        # Masked token pattern without brackets
        if re.match(r'^[A-Z_]+_[A-F0-9]{8}$', text_stripped):
            return True
        
        # --- Truncated Terms ---
        if text_stripped.endswith('...'):
            base = text_lower.rstrip('.')
            # Drug/pharmacy terms truncated
            safe_bases = ['mg', 'tab', 'cap', 'drug', 'pharm', 'claim', 'member', 'patient']
            if any(sb in base for sb in safe_bases):
                return True
            # Dosage pattern truncated
            if re.search(r'\d+\s*(mg|mcg|ml)', base):
                return True
        
        # --- Common Placeholder Names ---
        # NOTE: Removed 'john doe', 'jane doe', 'john smith', 'jane smith'
        # because these could be REAL member names in production data.
        # Only include truly generic/template placeholders.
        placeholder_names = {
            'test user', 'test member', 'sample user', 'sample member',
            'example', 'example user', 'example member',
            'patient name', 'member name', 'your name', 'full name',
            'first name', 'last name', 'name here', 'enter name',
            '[name]', '<name>', '{name}', '(name)',
            'n/a', 'na', 'none', 'null', 'unknown', 'tbd',
        }
        if text_lower in placeholder_names:
            return True
    
    # ================================================================
    # LOCATION ENTITY
    # ================================================================
    elif entity_type == "LOCATION":
        # US state abbreviations (2 uppercase letters)
        if len(text_stripped) == 2 and text_stripped.isupper() and text_stripped.isalpha():
            return True
        
        # Generic location terms
        generic_locations = {
            'usa', 'us', 'united states', 'pharmacy', 'store', 'retail',
            'mail order', 'online', 'local', 'national', 'regional',
        }
        if text_lower in generic_locations:
            return True
        
        # camelCase field names (submittedProductId)
        if re.match(r'^[a-z]+[A-Z]', text_stripped):
            return True
        
        # System codes with numbers
        if text_stripped.isupper() and re.search(r'\d', text_stripped):
            return True
        
        # Contains underscore
        if '_' in text_stripped:
            return True
        
        # Truncated ZIP codes
        if re.match(r'^\d{5}\.{3}$', text_stripped):
            return True
        
        # All-caps city names from API (allow them in pharmacy context)
        if text_stripped.isupper() and text_stripped.isalpha() and len(text_stripped) >= 3:
            return True
    
    # ================================================================
    # US_DRIVER_LICENSE ENTITY
    # ================================================================
    elif entity_type == "US_DRIVER_LICENSE":
        # Short codes (≤6 chars)
        if len(text_stripped) <= 6:
            return True
        
        # API versions: v1, v2, V1
        if re.match(r'^v\d+$', text_lower):
            return True
        
        # YYYYMMDD dates
        if re.match(r'^(19|20)\d{6}$', text_stripped):
            return True
        
        # Long numeric IDs (claim IDs, transaction IDs)
        if re.match(r'^\d{8,18}$', text_stripped):
            return True
        
        # Short alphanumeric codes: AP711, t101, CA12345
        if re.match(r'^[A-Za-z]{1,4}\d{2,8}$', text_stripped):
            return True
        
        # Alphanumeric with mixed letters/numbers
        if re.match(r'^[A-Za-z0-9]+$', text_stripped):
            has_letters = any(c.isalpha() for c in text_stripped)
            has_numbers = any(c.isdigit() for c in text_stripped)
            if has_letters and has_numbers and len(text_stripped) <= 15:
                return True
    
    # ================================================================
    # US_PASSPORT ENTITY
    # ================================================================
    elif entity_type == "US_PASSPORT":
        # All-zero placeholders
        if re.match(r'^0+$', text_stripped):
            return True
        
        # Common test/sequential values
        test_values = {'123456789', '987654321', '111111111', '999999999', '000000000'}
        if text_stripped in test_values:
            return True
        
        # 9-digit numbers in pharmacy context likely system IDs
        if re.match(r'^\d{9}$', text_stripped):
            return True
    
    # ================================================================
    # PHONE_NUMBER ENTITY
    # ================================================================
    elif entity_type == "PHONE_NUMBER":
        # Decimal numbers (prices, measurements)
        if re.match(r'^\d+\.\d+$', text_stripped):
            return True
        
        # 10-11 digit solid numbers (NPI, RxNumber, PrescriberID)
        if re.match(r'^\d{10,11}$', text_stripped):
            return True
        
        # Very short numbers
        if re.match(r'^\d{1,6}$', text_stripped):
            return True
        
        # Date formats: DD.MM.YYYY, MM.DD.YYYY, DD/MM/YYYY, etc.
        if re.match(r'^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$', text_stripped):
            return True
        
        # Date formats: YYYY-MM-DD, YYYY.MM.DD
        if re.match(r'^\d{4}[./-]\d{1,2}[./-]\d{1,2}$', text_stripped):
            return True
    
    # ================================================================
    # EMAIL_ADDRESS ENTITY
    # ================================================================
    elif entity_type == "EMAIL_ADDRESS":
        # Generic/system emails
        generic_prefixes = ['support', 'help', 'info', 'contact', 'service', 
                           'noreply', 'no-reply', 'admin', 'system', 'test']
        if any(text_lower.startswith(p + '@') for p in generic_prefixes):
            return True
    
    # ================================================================
    # PHARMACY DOMAIN IDENTIFIERS - Always contextual (NOT personal info)
    # ================================================================
    # These are system/transaction identifiers, not personally identifiable:
    # - CLAIM_ID: Transaction identifier (user asks about their claim)
    # - RX_NUMBER: Prescription number (system ID)
    # - NDC: National Drug Code (drug identifier)
    # - MEDICAL_LICENSE: Provider license (public record)
    elif entity_type in ("NDC", "MEDICAL_LICENSE", "CLAIM_ID", "RX_NUMBER"):
        return True
    
    # ================================================================
    # CREDIT_CARD ENTITY
    # ================================================================
    elif entity_type == "CREDIT_CARD":
        # Long claim IDs could be misclassified
        if len(text_stripped) < 13 or len(text_stripped) > 19:
            return True
    
    # ================================================================
    # US_SSN ENTITY - Be careful but check format
    # ================================================================
    elif entity_type == "US_SSN":
        # If not in XXX-XX-XXXX format, likely false positive
        if not re.match(r'^\d{3}-\d{2}-\d{4}$', text_stripped):
            return True
    
    # ================================================================
    # Final catch-all: Short alphanumeric codes (≤4 chars uppercase)
    # ================================================================
    if len(text_stripped) <= 4 and text_stripped.isupper():
        return True
    
    # Not contextual - could be real PII
    return False


def _looks_like_system_data(entity_text: str, entity_type: str) -> bool:
    """
    Additional check for lenient mode - identify system-like data patterns.
    
    Used when token mapping is empty and we need to be more permissive.
    This is a secondary defense layer for edge cases.
    
    Args:
        entity_text: The detected PII text
        entity_type: The type of PII entity
        
    Returns:
        True if this looks like system/generated data rather than real PII
    """
    import re
    
    # PHARMACY DOMAIN IDENTIFIERS - Always system data (NOT personal info)
    # These should never be flagged as leakage
    if entity_type in ("CLAIM_ID", "RX_NUMBER", "NDC", "MEDICAL_LICENSE"):
        return True
    
    text = entity_text.strip()
    text_lower = text.lower()
    
    # ================================================================
    # SHORT SEQUENCE NUMBERS - Catch misclassifications (ANY entity type)
    # ================================================================
    # Handles cases where Presidio misidentifies short numbers (001, 997, 12345)
    # as US_DRIVER_LICENSE, PHONE_NUMBER, etc. instead of CLAIM_ID.
    # These are likely claim sequence numbers or system codes, not real PII.
    # (Enhanced from MVP-1 merge - generalized to work for ALL entity types)
    numbers = re.findall(r'\d+', text)
    if numbers:
        max_digits = max(len(n) for n in numbers)
        if max_digits <= 5:  # 1-5 digit numbers are likely sequences/codes, not real PII
            # Additional safety: ensure not part of a very long number (10+ digits)
            if not re.search(r'\d{10,}', text):
                return True
    
    # System code patterns (letters + numbers mixed)
    if re.match(r'^[A-Z]{2,6}\d{2,8}$', text) or re.match(r'^\d{2,8}[A-Z]{2,6}$', text):
        return True
    
    # All-caps codes without spaces
    if text.isupper() and len(text) >= 4 and ' ' not in text:
        return True
    
    # Contains digits (unlikely to be real person name)
    if entity_type == "PERSON" and re.search(r'\d', text):
        return True
    
    # Looks like drug dosage
    if re.search(r'\d+\s*(mg|mcg|ml|g|tab|cap)', text_lower):
        return True
    
    # Very long single word (>15 chars) - likely code/ID
    if len(text) > 15 and ' ' not in text:
        return True
    
    # Field names with numbers
    if re.match(r'^[a-z]+\d+$', text):
        return True
    
    # Alphanumeric codes
    if re.match(r'^[A-Za-z0-9]+$', text) and len(text) <= 12:
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        if has_letters and has_numbers:
            return True
    
    # Common drug name suffixes (lenient mode catch-all)
    drug_suffixes = ('statin', 'pril', 'olol', 'sartan', 'dipine', 'prazole',
                     'cillin', 'mycin', 'azole', 'formin', 'mab', 'nib', 'afil')
    if any(text_lower.endswith(suffix) for suffix in drug_suffixes):
        return True
    
    # Known pharmacy chain keywords
    chain_keywords = ('walgreens', 'cvs', 'walmart', 'rite aid', 'costco', 'kroger')
    if any(kw in text_lower for kw in chain_keywords):
        return True
    
    # Rejection message patterns (all caps words in rejection context)
    if text.isupper() and len(text) >= 3:
        rejection_words = {'FORMULARY', 'AUTHORIZATION', 'PRESCRIBER', 'CARDHOLDER',
                          'COVERED', 'DENIED', 'REJECTED', 'REFILL', 'LIMIT'}
        if text in rejection_words:
            return True
    
    return False


def _validate_and_convert_invalid_claim_ids(entities: Dict[str, Any], logger) -> Dict[str, Any]:
    """
    Validate claim IDs and convert invalid ones to potential_claim_ids.
    
    Ensures consistency with entity_extractor's potential_claim_id concept:
    - Valid claim ID: exactly 15 digits OR CLM prefix
    - Invalid claim ID: 4+ digits but NOT 15 → moves to potential_claim_ids
    
    Args:
        entities: Entity dictionary to validate
        logger: Logger instance
        
    Returns:
        Modified entities dict with invalid claim IDs converted
    """
    if not entities:
        return entities
    
    claim_num = entities.get('claim_number')
    claim_ids = entities.get('claim_ids', [])
    
    # Initialize potential_claim_ids from existing or empty
    potential_ids = list(entities.get('potential_claim_ids', []))
    modified = False
    
    # Validate claim_number (valid: 15 digits OR CLM prefix)
    if claim_num:
        claim_num_str = str(claim_num)
        if not (re.match(r'^\d{15}$', claim_num_str) or re.match(r'^CLM\d{3,10}$', claim_num_str)):
            potential_ids.append(claim_num_str)
            del entities['claim_number']
            modified = True
            logger.info(f"   📋 Converted invalid claim_number '{claim_num_str}' to potential_claim_ids")
    
    # Validate claim_ids list
    if claim_ids:
        valid_ids = []
        for cid in claim_ids:
            cid_str = str(cid)
            if re.match(r'^\d{15}$', cid_str) or re.match(r'^CLM\d{3,10}$', cid_str):
                valid_ids.append(cid_str)
            else:
                if cid_str not in potential_ids:
                    potential_ids.append(cid_str)
                modified = True
                logger.info(f"   📋 Converted invalid claim_id '{cid_str}' to potential_claim_ids")
        
        if valid_ids:
            entities['claim_ids'] = valid_ids
        elif 'claim_ids' in entities:
            del entities['claim_ids']
    
    # Set potential_claim_ids and flag if we found any invalid IDs
    if potential_ids:
        entities['potential_claim_ids'] = potential_ids
        entities['claim_id_format_invalid'] = True
        if modified:
            logger.info(f"   ✅ Claim ID validation: potential_claim_ids={potential_ids}")
    
    return entities


# ============================================================================
# NODE 1: UNIFIED SAFETY PRECHECK (All-in-One)
# ============================================================================

async def safety_precheck_node(state: AgentState) -> Dict[str, Any]:
    """
    Unified safety check node that handles everything:
    
    Internal Steps:
    1. Violence pattern check (fast, local)
    2. Mask PII/PHI (local, no external calls)
    3. Call Gemini safety filters (on masked data - no PII leakage)
    4. Unmask PII/PHI (restore original values)
    5. Return safe query with PII/PHI intact
    
    INPUT (from state):
        - text: User's query
        - session_id: For token storage
    
    OUTPUT (to state):
        - text: Safe, unmasked query with PII/PHI intact
        - safety_precheck_passed: bool
        - threat_detected: bool (for backward compatibility)
        - threat_reason: str (for backward compatibility)
        - safety_block_reason: str (if blocked)
        - response: error message (if blocked)
        - metadata.pii_metadata: PII detection metadata
        - metadata.safety_precheck_metadata: SafetyResult metadata
    
    FLOW:
        Blocked → END (safety violation)
        Passed → Continue with PII/PHI intact for downstream nodes
    """
    node_name = "safety_precheck"
    start_time = time.time()
    
    logger.info("\n" + "="*70)
    logger.info("🛡️  SAFETY PRECHECK NODE - Unified Safety Check")
    logger.info("="*70)
    
    # Extract logging context
    log_ctx = extract_logging_context(state)
    session_id = log_ctx["session_id"]
    request_id = log_ctx["request_id"]
    
    # Check if safety precheck is enabled
    if not settings.enable_safety_precheck:
        logger.info("⭐ Safety precheck disabled in config")
        
        # Create result for disabled case
        processing_time_ms = (time.time() - start_time) * 1000
        safety_obj = create_safety_result(
            check_type=SafetyCheckType.PRECHECK,
            passed=True,
            block_reason="Safety precheck disabled in configuration",
            processing_time_ms=processing_time_ms
        )
        
        result = {
            "safety_precheck_passed": True,
            "threat_detected": False,
            "threat_reason": None,
            "safety_block_reason": None,
            "metadata": {
                **state.get("metadata", {}),
                "safety_precheck_metadata": to_dict(safety_obj)
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result
    
    text = state.get("text", "")
    
    try:
        # Get safety checker instance
        safety_checker = get_safety_checker()
        
        # Run complete safety pipeline (Method 3)
        logger.info(f"🔍 Running safety checks for session: {session_id}")
        safety_result = await safety_checker.check_harmful_content(text, session_id)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        if not safety_result["is_safe"]:
            # Safety violation detected
            reason = safety_result.get("reason", "Content safety violation")
            violation_categories = safety_result.get("violation_categories", [])
            
            logger.warning(f"🚫 BLOCKED: {reason}")
            if violation_categories:
                logger.warning(f"   Categories: {', '.join(violation_categories)}")
            
            # Log to telemetry
            await log_event(
                event_type=EventType.SAFETY_BLOCKED,
                session_id=session_id,
                data={
                    "reason": reason,
                    "violation_categories": violation_categories,
                    "text_length": len(text),
                    "processing_time_ms": processing_time_ms
                }
            )
            
            # Create SafetyResult object
            safety_obj = create_safety_result(
                check_type=SafetyCheckType.PRECHECK,
                passed=False,
                violation_type=SafetyViolationType.INAPPROPRIATE_CONTENT,
                block_reason=reason,
                confidence_score=1.0,
                user_message=(
                    "I'm here to help with pharmacy claims and coverage questions. "
                    "I can't assist with that type of request."
                ),
                processing_time_ms=processing_time_ms
            )
            
            # Log to persistence store
            persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
            error = create_safety_error(
                reason=reason,
                is_precheck=True,
                session_id=session_id
            )
            await persistence_store.log_exception(
                error_code=error.error_code.value,
                category=error.category.value,
                severity=error.severity.value,
                message=error.message,
                user_message=error.user_message,
                session_id=session_id,
                request_id=request_id,
                node_name=node_name,
                stacktrace=error.stacktrace,
                user_id=log_ctx.get("user_id", "unknown")
            )
            
            # Preserve intent/confidence from metadata if they exist (for visibility even when blocked)
            metadata = state.get("metadata", {})
            intent_classification_metadata = metadata.get("intent_classification_metadata", {})
            top_intent = intent_classification_metadata.get("top_intent")
            top_confidence = intent_classification_metadata.get("top_confidence")
            
            result_dict = {
                "safety_precheck_passed": False,
                "threat_detected": True,  # Backward compatibility
                "threat_reason": reason,  # Backward compatibility
                "safety_block_reason": reason,
                "response": safety_obj.user_message,
                "intent": top_intent if top_intent else state.get("intent"),  # Preserve intent if available
                "confidence": top_confidence if top_confidence else state.get("confidence"),  # Preserve confidence if available
                "metadata": {
                    **metadata,
                    "safety_precheck_metadata": to_dict(safety_obj),
                    "violation_categories": violation_categories
                }
            }
            await log_state_snapshot(state, node_name, result_dict)
            return result_dict
        
        # Safety check passed - return unmasked text with PII/PHI
        logger.info("✅ Safety check passed - Query is safe")
        logger.info(f"   PII/PHI intact for downstream processing")
        logger.info(f"   Processing time: {processing_time_ms:.2f}ms")
        
        # Store PII metadata for later use
        pii_metadata = safety_result.get("pii_metadata", {})
        metadata = state.get("metadata", {})
        metadata["pii_metadata"] = pii_metadata
        
        # Create SafetyResult object for passed case
        detected_keywords = []
        if pii_metadata.get("has_pii"):
            detected_keywords = pii_metadata.get("entities_detected", [])
        
        safety_obj = create_safety_result(
            check_type=SafetyCheckType.PRECHECK,
            passed=True,
            violation_type=SafetyViolationType.NONE,
            block_reason=None,
            detected_keywords=detected_keywords,
            confidence_score=1.0,
            processing_time_ms=processing_time_ms
        )
        
        metadata["safety_precheck_metadata"] = to_dict(safety_obj)
        
        # Log successful check to telemetry
        await log_event(
            event_type=EventType.REQUEST_RECEIVED,
            session_id=session_id,
            data={
                "node": "safety_precheck",
                "passed": True,
                "text_length": len(text),
                "has_pii": pii_metadata.get("has_pii", False),
                "pii_count": pii_metadata.get("masked_count", 0),
                "processing_time_ms": processing_time_ms,
                "mock_mode": safety_result.get("mock_mode", False)
            }
        )
        
        result_dict = {
            "text": safety_result["text"],  # Unmasked text with PII/PHI intact
            "safety_precheck_passed": True,
            "threat_detected": False,  # Backward compatibility
            "threat_reason": None,  # Backward compatibility
            "safety_block_reason": None,
            "metadata": metadata
        }
        await log_state_snapshot(state, node_name, result_dict)
        return result_dict
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        tb = traceback.format_exc()
        logger.error(f"❌ Safety precheck failed: {e}\n{tb}")
        
        # Create error result
        safety_obj = create_safety_result(
            check_type=SafetyCheckType.PRECHECK,
            passed=False,
            violation_type=SafetyViolationType.MALICIOUS_INPUT,
            block_reason=f"Safety check error: {str(e)}",
            user_message=(
                "I'm unable to process your request at this time. "
                "Please try again later."
            ),
            processing_time_ms=processing_time_ms
        )
        
        # Log error to persistence store
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        error = create_internal_error(
            error_message=f"Safety precheck failed: {str(e)}",
            stacktrace=tb,
            session_id=session_id,
            node_name=node_name
        )
        await persistence_store.log_exception(
            error_code=error.error_code.value,
            category=error.category.value,
            severity=error.severity.value,
            message=error.message,
            user_message=error.user_message,
            session_id=session_id,
            request_id=request_id,
            node_name=node_name,
            stacktrace=tb,
            metadata=error.metadata,
            user_id=log_ctx.get("user_id", "unknown")
        )
        
        # Log to telemetry
        await log_event(
            event_type=EventType.ERROR_OCCURRED,
            session_id=session_id,
            data={
                "node": "safety_precheck",
                "error": str(e),
                "text_length": len(text),
                "processing_time_ms": processing_time_ms
            }
        )
        
        # Fail closed - block on error
        # Preserve intent/confidence from metadata if they exist (for visibility even when blocked)
        metadata = state.get("metadata", {})
        intent_classification_metadata = metadata.get("intent_classification_metadata", {})
        top_intent = intent_classification_metadata.get("top_intent")
        top_confidence = intent_classification_metadata.get("top_confidence")
        
        result = {
            "safety_precheck_passed": False,
            "threat_detected": True,  # Backward compatibility
            "threat_reason": f"Safety check error: {str(e)}",  # Backward compatibility
            "safety_block_reason": f"Safety check error: {str(e)}",
            "response": safety_obj.user_message,
            "intent": top_intent if top_intent else state.get("intent"),  # Preserve intent if available
            "confidence": top_confidence if top_confidence else state.get("confidence"),  # Preserve confidence if available
            "metadata": {
                **metadata,
                "safety_precheck_metadata": to_dict(safety_obj),
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result


# ============================================================================
# NODE 2: RESPONSE SAFETY PII PRECHECK (Mask before LLM)
# ============================================================================

async def response_safety_pii_precheck_node(state: AgentState) -> Dict[str, Any]:
    """
    Mask PII/PHI before sending to response generation LLM
    
    This prevents PII leakage to the LLM service
    
    INPUT (from state):
        - text: UNMASKED text (with PII/PHI from previous nodes)
        - tool_results: May contain PII/PHI
        - conversation_history: UNMASKED history (may contain PII/PHI from previous turns)
        - session_id: For token storage
    
    OUTPUT (to state):
        - text: MASKED text
        - tool_results: MASKED (if contains PII/PHI)
        - conversation_history: MASKED (previous conversation with PII/PHI protected)
        - metadata.response_pii_masking: Masking metadata
    
    FLOW:
        Always continues
        Response agent works with MASKED data (safe)
    """
    node_name = "response_safety_pii_precheck"
    log_ctx = extract_logging_context(state)
    
    logger.info("\n" + "="*70)
    logger.info("🔐 RESPONSE SAFETY PII PRECHECK - Masking before Response LLM")
    logger.info("="*70)
    
    text = state.get("text", "")
    tool_results = state.get("tool_results", {})
    conversation_history = state.get("conversation_history", [])
    session_id = state.get("session_id", "default")
    
    try:
        pii_service = get_pii_service()
        
        # Mask text (user input PII)
        masked_text, text_metadata = pii_service.mask_pii_phi(text, session_id)
        
        # Mask tool results using mask_api_response (handles structured data properly)
        # This will reuse tokens from text_metadata if same PII appears, and track API-sourced PII separately
        if tool_results and isinstance(tool_results, dict):
            logger.info(f"🔍 Masking tool results (API data)...")
            existing_token_mapping = text_metadata.get("token_mapping", {})
            masked_tool_results, combined_token_mapping = pii_service.mask_api_response(
                api_response=tool_results,
                session_id=session_id,
                existing_token_mapping=existing_token_mapping
            )
            
            # Calculate tool-specific masked count (new tokens created for API data)
            tool_masked_count = len(combined_token_mapping) - len(existing_token_mapping)
            
            # Build tool_metadata with proper structure
            tool_metadata = {
                "has_pii": tool_masked_count > 0,
                "masked_count": tool_masked_count,
                "entities_detected": [
                    data["entity_type"] 
                    for token, data in combined_token_mapping.items() 
                    if token not in existing_token_mapping
                ],
                "token_mapping": combined_token_mapping  # Full combined mapping
            }
            
            logger.info(f"🎭 Masked {tool_masked_count} NEW PII/PHI entities from API data")
        else:
            masked_tool_results = tool_results
            tool_metadata = {
                "has_pii": False,
                "masked_count": 0,
                "entities_detected": [],
                "token_mapping": text_metadata.get("token_mapping", {})
            }
        
        # CRITICAL: Mask conversation history (previous turns may contain PII/PHI)
        # FIX: Track history token mappings for multi-turn unmasking
        masked_history = []
        history_masked_count = 0
        history_token_mapping = {}  # Track tokens from history masking
        
        if conversation_history:
            logger.info(f"🔍 Masking conversation history ({len(conversation_history)} messages)...")
            
            for msg in conversation_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if content:
                    # Mask content and CAPTURE the metadata (FIX: don't discard with _)
                    masked_content, history_msg_metadata = pii_service.mask_pii_phi(content, session_id)
                    
                    # FIX: Collect history token mappings for unmasking
                    msg_token_mapping = history_msg_metadata.get("token_mapping", {})
                    if msg_token_mapping:
                        history_token_mapping.update(msg_token_mapping)
                        history_masked_count += len(msg_token_mapping)
                    
                    masked_history.append({
                        "role": role,
                        "content": masked_content
                    })
                else:
                    masked_history.append(msg)
            
            logger.info(f"🎭 Masked {history_masked_count} PII/PHI entities in conversation history")
            
            # FIX: Merge history token mappings into tool_metadata for postcheck
            if history_token_mapping:
                tool_metadata["token_mapping"].update(history_token_mapping)
                logger.info(f"📋 Merged {len(history_token_mapping)} history tokens into mapping")
        else:
            masked_history = conversation_history
        
        total_masked = text_metadata["masked_count"] + tool_metadata["masked_count"] + history_masked_count
        
        if total_masked > 0:
            logger.info(f"🎭 Total masked: {total_masked} PII/PHI entities before response LLM")
            logger.info(f"   - Text (user input): {text_metadata['masked_count']} entities")
            logger.info(f"   - Tool results (API data): {tool_metadata['masked_count']} entities")
            logger.info(f"   - Conversation history: {history_masked_count} entities")
            logger.debug(f"   Original text: {text[:100]}...")
            logger.debug(f"   Masked text: {masked_text[:100]}...")
        else:
            logger.info("ℹ️  No PII/PHI detected - data unchanged")
        
        # Store metadata for unmasking (with FULL token mappings)
        metadata = state.get("metadata", {})
        metadata["response_pii_masking"] = {
            "text_metadata": text_metadata,
            "tool_metadata": tool_metadata
        }
        
        result = {
            "text": masked_text,
            "tool_results": masked_tool_results,  # Pass masked tool results to response agent
            "conversation_history": masked_history,  # Pass masked conversation history to response agent
            "metadata": metadata
        }
        await log_state_snapshot(state, node_name, result)
        return result
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Response PII masking failed: {e}\n{tb}")
        
        error = create_internal_error(
            error_message=f"Response PII masking failed: {str(e)}",
            stacktrace=tb,
            session_id=log_ctx["session_id"],
            node_name=node_name
        )
        
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        await persistence_store.log_exception(
            error_code=error.error_code.value,
            category=error.category.value,
            severity=error.severity.value,
            message=error.message,
            user_message=error.user_message,
            session_id=log_ctx["session_id"],
            request_id=log_ctx["request_id"],
            node_name=node_name,
            stacktrace=error.stacktrace,
            metadata=error.metadata,
            user_id=log_ctx["user_id"]
        )
        
        # FIX: Try to preserve any token mappings we created before failure
        # This ensures postcheck can still unmask even if precheck partially failed
        preserved_text_mapping = {}
        preserved_tool_mapping = {}
        try:
            if 'text_metadata' in locals() and text_metadata:
                preserved_text_mapping = text_metadata.get("token_mapping", {})
            if 'tool_metadata' in locals() and tool_metadata:
                preserved_tool_mapping = tool_metadata.get("token_mapping", {})
            if 'history_token_mapping' in locals() and history_token_mapping:
                preserved_tool_mapping.update(history_token_mapping)
            if preserved_text_mapping or preserved_tool_mapping:
                logger.info(f"💾 Preserved {len(preserved_text_mapping)} text + {len(preserved_tool_mapping)} tool tokens despite error")
        except Exception:
            pass  # Best effort - don't fail the error handler
        
        # Fail-safe: continue with original data but PRESERVE any token mappings
        result = {
            "metadata": {
                **state.get("metadata", {}),
                "response_pii_masking": {
                    "text_metadata": {
                        "has_pii": bool(preserved_text_mapping),
                        "masked_count": len(preserved_text_mapping),
                        "token_mapping": preserved_text_mapping
                    },
                    "tool_metadata": {
                        "has_pii": bool(preserved_tool_mapping),
                        "masked_count": len(preserved_tool_mapping),
                        "token_mapping": preserved_tool_mapping
                    },
                    "error": str(e)
                },
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result


# ============================================================================
# NODE 3: RESPONSE SAFETY PII POSTCHECK (Unmask for user)
# ============================================================================

async def response_safety_pii_postcheck_node(state: AgentState) -> Dict[str, Any]:
    """
    Unmask PII/PHI in the generated response + Leakage detection
    
    Two steps:
    1. Leakage Detection: Check if LLM response contains unexpected PII
    2. Token Unmasking: Restore original PII/PHI values for user
    
    INPUT (from state):
        - response: LLM-generated response (with tokens)
        - session_id: For token lookup
        - metadata.response_pii_masking: Token mappings
    
    OUTPUT (to state):
        - response: UNMASKED response (real values restored)
        - text: UNMASKED text (for conversation history)
        - safety_postcheck_passed: bool
        - metadata.response_pii_unmasking: Unmasking stats
        - metadata.leakage_check: Leakage detection results
    
    FLOW:
        Leakage detected → Block response, return generic message
        No leakage → Unmask tokens, return to user
    """
    node_name = "response_safety_pii_postcheck"
    log_ctx = extract_logging_context(state)
    
    logger.info("\n" + "="*70)
    logger.info("🔍 RESPONSE SAFETY PII POSTCHECK - Leakage Check + Unmasking")
    logger.info("="*70)
    
    response = state.get("response", "")
    session_id = state.get("session_id", "default")
    
    if not response:
        logger.warning("⚠️  No response to postcheck (LLM judge path)")
        
        # FIX #9: Unmask entities even when no response (LLM judge path)
        # LLM judge extracts entities from MASKED text, so entities contain masked tokens
        # Without unmasking here, call_claims_tool receives masked tokens and fails API routing
        pii_service = get_pii_service()
        entities = state.get("entities")
        unmasked_entities = None
        
        if entities and session_id:
            logger.info("🔍 Unmasking entities (no-response path for LLM judge)...")
            unmasked_entities = {}
            
            for key, value in entities.items():
                if isinstance(value, str):
                    unmasked_value = value
                    # Handle full tokens [CLAIM_ID_XXXXXXXX]
                    full_token_match = re.match(r'^\[([A-Z_]+)_([A-Fa-f0-9]{8})\]$', value)
                    if full_token_match:
                        storage_key = f"{session_id}:{value}"
                        if storage_key in pii_service.token_storage:
                            original = pii_service.token_storage[storage_key]
                            # Strip "claim" prefix for claim-related entities
                            if key in ('claim_number', 'claim_id', 'claim_ids', 'claimNumber', 'claimId'):
                                numeric_match = re.search(r'\d+$', str(original))
                                unmasked_value = numeric_match.group(0) if numeric_match else original
                            else:
                                unmasked_value = original
                            logger.info(f"   🔓 Unmasked '{key}': {value} → {unmasked_value}")
                    unmasked_entities[key] = unmasked_value
                    
                elif isinstance(value, list):
                    unmasked_list = []
                    for item in value:
                        if isinstance(item, str):
                            unmasked_item = item
                            full_token_match = re.match(r'^\[([A-Z_]+)_([A-Fa-f0-9]{8})\]$', item)
                            if full_token_match:
                                storage_key = f"{session_id}:{item}"
                                if storage_key in pii_service.token_storage:
                                    original = pii_service.token_storage[storage_key]
                                    if key in ('claim_number', 'claim_id', 'claim_ids', 'claimNumber', 'claimId'):
                                        numeric_match = re.search(r'\d+$', str(original))
                                        unmasked_item = numeric_match.group(0) if numeric_match else original
                                    else:
                                        unmasked_item = original
                                    logger.info(f"   🔓 Unmasked '{key}' item: {item} → {unmasked_item}")
                            unmasked_list.append(unmasked_item)
                        else:
                            unmasked_list.append(item)
                    unmasked_entities[key] = unmasked_list
                else:
                    unmasked_entities[key] = value
            
            # Validate claim IDs (consistency with embedding classifier path)
            unmasked_entities = _validate_and_convert_invalid_claim_ids(unmasked_entities, logger)
        
        result = {
            "safety_postcheck_passed": True,
            "entities": unmasked_entities if unmasked_entities else entities
        }
        await log_state_snapshot(state, node_name, result)
        return result
    
    try:
        pii_service = get_pii_service()
        metadata = state.get("metadata", {})
        response_pii_masking = metadata.get("response_pii_masking", {})
        
        # ================================================================
        # DEFENSIVE CHECK 1: Handle previous errors in pipeline
        # ================================================================
        if state.get("error") or metadata.get("error_occurred"):
            logger.info("⚠️ Previous error detected - skipping strict leakage check")
            result = {
                "response": response,
                "safety_postcheck_passed": True,
                "metadata": {
                    **metadata,
                    "leakage_check": {
                        "has_leakage": False,
                        "skipped_reason": "Previous error in pipeline"
                    }
                }
            }
            await log_state_snapshot(state, node_name, result)
            return result
        
        # ================================================================
        # DEFENSIVE CHECK 2: Handle missing masking metadata
        # ================================================================
        if not response_pii_masking:
            logger.warning("⚠️ No response_pii_masking in metadata - using lenient mode")
            # Without token mapping, we can't compare; be lenient to avoid false positives
            result = {
                "response": response,
                "safety_postcheck_passed": True,
                "metadata": {
                    **metadata,
                    "leakage_check": {
                        "has_leakage": False,
                        "warning": "Missing masking metadata - leakage check lenient"
                    }
                }
            }
            await log_state_snapshot(state, node_name, result)
            return result
        
        # Get token mappings with defaults
        text_metadata = response_pii_masking.get("text_metadata", {})
        tool_metadata = response_pii_masking.get("tool_metadata", {})
        
        text_token_mapping = text_metadata.get("token_mapping", {})
        tool_token_mapping = tool_metadata.get("token_mapping", {})
        
        # Combine token mappings
        combined_token_mapping = {**text_token_mapping, **tool_token_mapping}
        
        # ================================================================
        # DEFENSIVE CHECK 3: Empty token mapping - use lenient mode
        # ================================================================
        lenient_mode = False
        if not combined_token_mapping:
            logger.info("ℹ️ No tokens in mapping - using lenient leakage detection")
            lenient_mode = True
        
        # ===== STEP 1: Check for PII Leakage =====
        logger.info(f"Step 1: Leakage detection{' (lenient mode)' if lenient_mode else ''}")
        
        # Detect any NEW PII in response (not from original input)
        detected_pii = pii_service.detect_pii_phi(response)
        
        leaked_entities = []
        for entity in detected_pii:
            entity_text = entity["text"]
            entity_type = entity["entity_type"]
            
            # STEP 0: Skip very short entities (≤3 chars) - preserves original behavior
            if len(entity_text.strip()) <= 3:
                logger.debug(f"   ✓ Skipping short entity: '{entity_text}'")
                continue
            
            # STEP 1: Skip if this is a masked token (e.g., [CLAIM_ID_ABC123])
            # These are legitimate masked tokens, not leaked PII
            if _is_masked_token(entity_text):
                logger.debug(f"   ✓ Skipping token: {entity_text} (legitimate masked token)")
                continue
            
            # STEP 2: Check if this PII value was in original input or API data
            # This checks against BOTH text_token_mapping (user input) AND tool_token_mapping (API data)
            is_expected = any(
                entity_text == data["original"] 
                for data in combined_token_mapping.values()
            )
            
            # STEP 2b: For PERSON entities, check name variations
            if not is_expected and entity_type == "PERSON":
                # Check reversed name order (first last <-> last first)
                if ' ' in entity_text:
                    parts = entity_text.split()
                    if len(parts) == 2:
                        reversed_name = f"{parts[1]} {parts[0]}"
                        is_expected = any(
                            reversed_name == data["original"]
                            for data in combined_token_mapping.values()
                            if data.get("entity_type") == "PERSON"
                        )
                
                # STEP 2c: Check partial matches (first name OR last name)
                if not is_expected:
                    for data in combined_token_mapping.values():
                        if data.get("entity_type") == "PERSON":
                            original = data["original"]
                            # Partial match: entity is part of original or vice versa
                            if entity_text in original or original in entity_text:
                                is_expected = True
                                logger.debug(f"   ✓ Partial name match: '{entity_text}' ~ '{original}'")
                                break
                            # Check individual name parts
                            original_parts = set(original.split())
                            entity_parts = set(entity_text.split())
                            if original_parts & entity_parts:  # Intersection exists
                                is_expected = True
                                logger.debug(f"   ✓ Name part match: '{entity_text}' shares parts with '{original}'")
                                break
            
            # STEP 3: Check if this is contextual data (comprehensive filter)
            is_contextual = _is_contextual_entity(entity_text, entity_type)
            
            # STEP 4: In lenient mode, apply additional permissive checks
            if lenient_mode and not is_expected and not is_contextual:
                # Secondary check: if entity looks like system data, allow it
                if _looks_like_system_data(entity_text, entity_type):
                    is_contextual = True
                    logger.debug(f"   ✓ Lenient mode: allowing system-like data: {entity_text[:20]}")
            
            if not is_expected and not is_contextual:
                # This could be a real leak
                logger.warning(f"   ⚠️  Potential leak: {entity_type} = {entity_text[:20]}...")
                leaked_entities.append(entity)
            else:
                if is_expected:
                    logger.debug(f"   ✓ Expected PII: {entity_type} (from input or API)")
                if is_contextual:
                    logger.debug(f"   ✓ Contextual data: {entity_type} = {entity_text[:20]}...")
        
        if leaked_entities:
            logger.error(
                f"🚨 PII LEAKAGE DETECTED! Found {len(leaked_entities)} "
                f"unexpected entities: {[e['entity_type'] for e in leaked_entities]}"
            )
            result = {
                "response": (
                    "I apologize, but I cannot display that information due to "
                    "privacy protection. Please try rephrasing your question."
                ),
                "safety_postcheck_passed": False,
                "metadata": {
                    **metadata,
                    "leakage_check": {
                        "has_leakage": True,
                        "leaked_entities": [
                            {
                                "type": e["entity_type"],
                                "text": e["text"][:20] + "..."
                            }
                            for e in leaked_entities
                        ]
                    }
                }
            }
            await log_state_snapshot(state, node_name, result)
            return result
        
        logger.info("✅ No PII leakage detected")
        
        # ===== STEP 2: Unmask PII/PHI Tokens =====
        logger.info("Step 2: Unmasking tokens")
        
        if not combined_token_mapping:
            logger.info("ℹ️  No tokens to unmask - using token_storage fallback")
            
            # Even without token mapping, apply bare bracket cleanup (LLM formatting)
            response_cleaned = pii_service.cleanup_remaining_tokens(response, None)
            text_cleaned = pii_service.cleanup_remaining_tokens(state.get("text", ""), None)
            
            # FIX: Clean up "claim claim" duplication even when no token mapping
            # This can happen when response_safety_pii_precheck runs twice and overwrites mapping
            response_cleaned = re.sub(r'\bclaim\s+claim\s+(\d{4,})', r'claim \1', response_cleaned, flags=re.IGNORECASE)
            
            # FIX: Unmask entities using token_storage fallback even when combined_token_mapping is empty
            # The second call to response_safety_pii_precheck clears the mapping, but tokens persist in storage
            entities = state.get("entities")
            unmasked_entities = None
            entities_tokens_unmasked = 0
            session_id = state.get("session_id", "")
            
            if entities and session_id:
                logger.info("🔍 Unmasking entities using token_storage fallback...")
                unmasked_entities = {}
                
                for key, value in entities.items():
                    if isinstance(value, str):
                        unmasked_value = value
                        
                        # Handle full tokens like [CLAIM_ID_BC3D5E2E] using token_storage lookup
                        full_token_match = re.match(r'^\[([A-Z_]+)_([A-Fa-f0-9]{8})\]$', value)
                        if full_token_match:
                            storage_key = f"{session_id}:{value}"
                            if storage_key in pii_service.token_storage:
                                original = pii_service.token_storage[storage_key]
                                # Strip "claim" prefix for claim-related entities
                                if key in ('claim_number', 'claim_id', 'claim_ids', 'claimNumber', 'claimId'):
                                    numeric_match = re.search(r'\d+$', str(original))
                                    if numeric_match:
                                        unmasked_value = numeric_match.group(0)
                                    else:
                                        unmasked_value = original
                                else:
                                    unmasked_value = original
                                entities_tokens_unmasked += 1
                                logger.info(f"   🔓 Unmasked full token '{key}': {value} → {unmasked_value}")
                        
                        # Handle bare hash like BC3D5E2E (LLM corrupted token)
                        elif re.match(r'^[A-Fa-f0-9]{8}$', value) and not value.isdigit():
                            hash_upper = value.upper()
                            for storage_key, original in pii_service.token_storage.items():
                                if storage_key.startswith(f"{session_id}:") and hash_upper in storage_key.upper():
                                    if key in ('claim_number', 'claim_id', 'claim_ids', 'claimNumber', 'claimId'):
                                        numeric_match = re.search(r'\d+$', str(original))
                                        if numeric_match:
                                            unmasked_value = numeric_match.group(0)
                                        else:
                                            unmasked_value = original
                                    else:
                                        unmasked_value = original
                                    entities_tokens_unmasked += 1
                                    logger.info(f"   🔓 Unmasked bare hash '{key}': {value} → {unmasked_value}")
                                    break
                        
                        unmasked_entities[key] = unmasked_value
                        
                    elif isinstance(value, list):
                        unmasked_list = []
                        for item in value:
                            if isinstance(item, str):
                                unmasked_item = item
                                
                                # Handle full tokens in list items
                                full_token_match = re.match(r'^\[([A-Z_]+)_([A-Fa-f0-9]{8})\]$', item)
                                if full_token_match:
                                    storage_key = f"{session_id}:{item}"
                                    if storage_key in pii_service.token_storage:
                                        original = pii_service.token_storage[storage_key]
                                        if key in ('claim_number', 'claim_id', 'claim_ids', 'claimNumber', 'claimId'):
                                            numeric_match = re.search(r'\d+$', str(original))
                                            if numeric_match:
                                                unmasked_item = numeric_match.group(0)
                                            else:
                                                unmasked_item = original
                                        else:
                                            unmasked_item = original
                                        entities_tokens_unmasked += 1
                                        logger.info(f"   🔓 Unmasked full token '{key}' item: {item} → {unmasked_item}")
                                
                                # Handle bare hash in list items
                                elif re.match(r'^[A-Fa-f0-9]{8}$', item) and not item.isdigit():
                                    hash_upper = item.upper()
                                    for storage_key, original in pii_service.token_storage.items():
                                        if storage_key.startswith(f"{session_id}:") and hash_upper in storage_key.upper():
                                            if key in ('claim_number', 'claim_id', 'claim_ids', 'claimNumber', 'claimId'):
                                                numeric_match = re.search(r'\d+$', str(original))
                                                if numeric_match:
                                                    unmasked_item = numeric_match.group(0)
                                                else:
                                                    unmasked_item = original
                                            else:
                                                unmasked_item = original
                                            entities_tokens_unmasked += 1
                                            logger.info(f"   🔓 Unmasked bare hash '{key}' item: {item} → {unmasked_item}")
                                            break
                                
                                unmasked_list.append(unmasked_item)
                            else:
                                unmasked_list.append(item)
                        unmasked_entities[key] = unmasked_list
                    else:
                        unmasked_entities[key] = value
                
                if entities_tokens_unmasked > 0:
                    logger.info(f"✅ Unmasked {entities_tokens_unmasked} token(s) in entities field (fallback path)")
                
                # FIX: Validate claim IDs and convert invalid ones to potential_claim_ids
                unmasked_entities = _validate_and_convert_invalid_claim_ids(unmasked_entities, logger)
            
            result = {
                "response": response_cleaned if response_cleaned != response else response,
                "text": text_cleaned if text_cleaned != state.get("text", "") else state.get("text", ""),
                "entities": unmasked_entities if unmasked_entities else state.get("entities"),
                "safety_postcheck_passed": True,
                "metadata": {
                    **metadata,
                    "leakage_check": {"has_leakage": False},
                    "response_pii_unmasking": {
                        "tokens_unmasked": 0,
                        "entities_tokens_unmasked": entities_tokens_unmasked,
                        "fallback_path": True
                    }
                }
            }
            await log_state_snapshot(state, node_name, result)
            return result
        
        # Unmask tokens in response
        unmasked_response = pii_service.unmask_pii_phi(response, combined_token_mapping)
        
        # FIX: Final cleanup to catch any remaining token patterns
        # Handles: fake tokens (LLM hallucinated), history tokens not in mapping, malformed tokens
        unmasked_response = pii_service.cleanup_remaining_tokens(
            unmasked_response, 
            combined_token_mapping
        )
        
        # CRITICAL: Also unmask text field so conversation history stores unmasked data
        text = state.get("text", "")
        unmasked_text = pii_service.unmask_pii_phi(text, text_token_mapping) if text_token_mapping else text
        # FIX: Apply cleanup to text as well
        unmasked_text = pii_service.cleanup_remaining_tokens(unmasked_text, text_token_mapping)
        
        # === FIX 2: Unmask entities field ===
        # Entities can contain masked tokens if extracted from conversation history by LLM Judge
        # Example: User provides only sequence "997", LLM Judge extracts claim_number from history
        # but it's masked like "[CLAIM_ID_F309B22F]", causing masked tokens in API response
        # Also handles bare hash fragments (e.g., "BD6783CC") by looking up in token_storage
        entities = state.get("entities")
        unmasked_entities = None
        entities_tokens_unmasked = 0
        
        # Process entities if they exist - even without combined_token_mapping, we may need
        # to look up bare hashes from token_storage (cross-turn token lookup)
        if entities:
            logger.info("🔍 Checking entities field for masked tokens...")
            unmasked_entities = {}
            
            # Get session_id for cross-turn token_storage lookup
            session_id = state.get("session_id", "")
            
            for key, value in entities.items():
                if isinstance(value, str):
                    unmasked_value = value  # Start with original value
                    
                    # Standard unmask if we have token mapping
                    if combined_token_mapping:
                        unmasked_value = pii_service.unmask_pii_phi(value, combined_token_mapping)
                        unmasked_value = pii_service.cleanup_remaining_tokens(unmasked_value, combined_token_mapping)
                    
                    # FIX: For claim-related entities, ALWAYS strip "claim" prefix
                    # Entities should contain clean numeric IDs (e.g., "233211748898001" not "claim 233211748898001")
                    # Context-aware unmasking in unmask_pii_phi won't work for entity values
                    # because entity value is just "[TOKEN]" with no surrounding "claim" keyword
                    if key in ('claim_number', 'claim_id', 'claim_ids', 'claimNumber', 'claimId'):
                        # Skip if value looks like a bare hash (8 hex chars) - handled by bare hash code below
                        is_bare_hash = re.match(r'^[A-Fa-f0-9]{8}$', unmasked_value) and not unmasked_value.isdigit()
                        if not is_bare_hash:
                            numeric_match = re.search(r'\d+$', unmasked_value)
                            if numeric_match:
                                unmasked_value = numeric_match.group(0)
                    
                    # FIX: Handle tokens not in combined_token_mapping using token_storage fallback
                    # This handles both full tokens and bare hash fragments
                    if unmasked_value == value and session_id:
                        # Pattern 1: Full token like [CLAIM_ID_BC3D5E2E] not in mapping
                        full_token_match = re.match(r'^\[([A-Z_]+)_([A-Fa-f0-9]{8})\]$', value)
                        if full_token_match:
                            storage_key = f"{session_id}:{value}"
                            if storage_key in pii_service.token_storage:
                                original_value = pii_service.token_storage[storage_key]
                                if key in ('claim_number', 'claim_id', 'claim_ids', 'claimNumber', 'claimId'):
                                    id_match = re.search(r'\d+$', str(original_value))
                                    if id_match:
                                        unmasked_value = id_match.group(0)
                                    else:
                                        unmasked_value = original_value
                                else:
                                    unmasked_value = original_value
                                logger.info(f"   🔓 Matched full token '{value}' to stored token → {unmasked_value}")
                        
                        # Pattern 2: Bare hash like BD6783CC (LLM corrupted token)
                        elif re.match(r'^[A-Fa-f0-9]{8}$', value) and not value.isdigit():
                            hash_upper = value.upper()
                            # Search singleton's token_storage for any token containing this hash
                            for storage_key, original_value in pii_service.token_storage.items():
                                # Match: session_id:[ENTITY_TYPE_HASH] where HASH matches our bare hash
                                if storage_key.startswith(f"{session_id}:") and hash_upper in storage_key.upper():
                                    # Extract just the numeric ID portion from original value
                                    # e.g., "claim 233211748898001" → "233211748898001"
                                    id_match = re.search(r'\d{8,20}', str(original_value))
                                    if id_match:
                                        unmasked_value = id_match.group(0)
                                    else:
                                        unmasked_value = original_value  # Fallback to full original
                                    logger.info(f"   🔓 Matched bare hash '{value}' to stored token → {unmasked_value}")
                                    break
                    
                    if unmasked_value != value:
                        entities_tokens_unmasked += 1
                        logger.info(f"   🔓 Unmasked entity '{key}': {value} → {unmasked_value}")
                    
                    unmasked_entities[key] = unmasked_value
                    
                elif isinstance(value, list):
                    # Handle list values (e.g., claim_ids)
                    unmasked_list = []
                    for item in value:
                        if isinstance(item, str):
                            unmasked_item = item  # Start with original value
                            
                            # Standard unmask if we have token mapping
                            if combined_token_mapping:
                                unmasked_item = pii_service.unmask_pii_phi(item, combined_token_mapping)
                                unmasked_item = pii_service.cleanup_remaining_tokens(unmasked_item, combined_token_mapping)
                            
                            # FIX: For claim-related entities, ALWAYS strip "claim" prefix
                            if key in ('claim_number', 'claim_id', 'claim_ids', 'claimNumber', 'claimId'):
                                # Skip if value looks like a bare hash - handled by bare hash code below
                                is_bare_hash = re.match(r'^[A-Fa-f0-9]{8}$', unmasked_item) and not unmasked_item.isdigit()
                                if not is_bare_hash:
                                    numeric_match = re.search(r'\d+$', unmasked_item)
                                    if numeric_match:
                                        unmasked_item = numeric_match.group(0)
                            
                            # FIX: Handle tokens not in mapping in list items using token_storage fallback
                            if unmasked_item == item and session_id:
                                # Pattern 1: Full token like [CLAIM_ID_BC3D5E2E] not in mapping
                                full_token_match = re.match(r'^\[([A-Z_]+)_([A-Fa-f0-9]{8})\]$', item)
                                if full_token_match:
                                    storage_key = f"{session_id}:{item}"
                                    if storage_key in pii_service.token_storage:
                                        original_value = pii_service.token_storage[storage_key]
                                        if key in ('claim_number', 'claim_id', 'claim_ids', 'claimNumber', 'claimId'):
                                            id_match = re.search(r'\d+$', str(original_value))
                                            if id_match:
                                                unmasked_item = id_match.group(0)
                                            else:
                                                unmasked_item = original_value
                                        else:
                                            unmasked_item = original_value
                                        logger.info(f"   🔓 Matched full token '{item}' in list to stored token → {unmasked_item}")
                                
                                # Pattern 2: Bare hash like BD6783CC (LLM corrupted)
                                elif re.match(r'^[A-Fa-f0-9]{8}$', item) and not item.isdigit():
                                    hash_upper = item.upper()
                                    for storage_key, original_value in pii_service.token_storage.items():
                                        if storage_key.startswith(f"{session_id}:") and hash_upper in storage_key.upper():
                                            id_match = re.search(r'\d{8,20}', str(original_value))
                                            if id_match:
                                                unmasked_item = id_match.group(0)
                                            else:
                                                unmasked_item = original_value
                                            logger.info(f"   🔓 Matched bare hash '{item}' in list to stored token → {unmasked_item}")
                                            break
                            
                            if unmasked_item != item:
                                entities_tokens_unmasked += 1
                                logger.info(f"   🔓 Unmasked entity '{key}' item: {item} → {unmasked_item}")
                            
                            unmasked_list.append(unmasked_item)
                        else:
                            unmasked_list.append(item)
                    unmasked_entities[key] = unmasked_list
                    
                else:
                    # Keep non-string values as-is (numbers, booleans, dicts, etc.)
                    unmasked_entities[key] = value
            
            if entities_tokens_unmasked > 0:
                logger.info(f"✅ Unmasked {entities_tokens_unmasked} token(s) in entities field")
            
            # FIX: Validate claim IDs and convert invalid ones to potential_claim_ids
            unmasked_entities = _validate_and_convert_invalid_claim_ids(unmasked_entities, logger)
        
        tokens_unmasked = sum(1 for token in combined_token_mapping.keys() if token in response)
        text_tokens_unmasked = sum(1 for token in text_token_mapping.keys() if token in text) if text_token_mapping else 0
        
        logger.info(f"🔓 Unmasked {tokens_unmasked} tokens in final response")
        if text_tokens_unmasked > 0:
            logger.info(f"🔓 Unmasked {text_tokens_unmasked} tokens in text field (for conversation history)")
        logger.debug(f"   Masked response: {response[:100]}...")
        logger.debug(f"   Unmasked response: {unmasked_response[:100]}...")
        
        # FIX: Clean up "claim claim" duplication in response
        # This happens when LLM writes "claim" + copies "claim 233..." from input = "claim claim 233..."
        # Pattern: 4+ digits catches all potential claim IDs (valid=15 digits, invalid=4+ but not 15)
        # Excludes 3-digit sequences which are not claim IDs
        cleaned_response = re.sub(r'\bclaim\s+claim\s+(\d{4,})', r'claim \1', unmasked_response, flags=re.IGNORECASE)
        if cleaned_response != unmasked_response:
            logger.info(f"🔧 Cleaned 'claim claim' duplication in response")
            unmasked_response = cleaned_response
        
        result = {
            "text": unmasked_text,  # ← CRITICAL: Unmask text so conversation history stores real values
            "response": unmasked_response,  # ← CRITICAL: Replace with unmasked
            "entities": unmasked_entities if unmasked_entities else state.get("entities"),  # ← NEW: Unmask entities
            "safety_postcheck_passed": True,
            "metadata": {
                **metadata,
                "leakage_check": {"has_leakage": False},
                "response_pii_unmasking": {
                    "tokens_unmasked": tokens_unmasked,
                    "text_tokens_unmasked": text_tokens_unmasked,
                    "entities_tokens_unmasked": entities_tokens_unmasked,  # NEW: Track entities unmasking
                    "token_types": list(set(
                        data["entity_type"] 
                        for data in combined_token_mapping.values()
                    ))
                }
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Response postcheck failed: {e}\n{tb}")
        
        error = create_internal_error(
            error_message=f"Response postcheck failed: {str(e)}",
            stacktrace=tb,
            session_id=log_ctx["session_id"],
            node_name=node_name
        )
        
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        await persistence_store.log_exception(
            error_code=error.error_code.value,
            category=error.category.value,
            severity=error.severity.value,
            message=error.message,
            user_message=error.user_message,
            session_id=log_ctx["session_id"],
            request_id=log_ctx["request_id"],
            node_name=node_name,
            stacktrace=error.stacktrace,
            metadata=error.metadata,
            user_id=log_ctx["user_id"]
        )
        
        # Fail-safe: Return response as-is (may contain tokens)
        result = {
            "response": response,
            "safety_postcheck_passed": True,
            "metadata": {
                **state.get("metadata", {}),
                "response_pii_unmasking": {"error": str(e)},
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result


# ============================================================================
# ROUTERS
# ============================================================================

def should_continue_after_precheck(state: AgentState) -> str:
    """
    After safety precheck, decide next step

    ROUTING:
        Blocked → END (return error)
        Passed → continue (to cache check)
    """
    if not state.get("safety_precheck_passed", False):
        logger.info("⛔ Flow blocked - threat detected")
        return END
    
    logger.info("✅ Flow continues")
    return "check_cache"