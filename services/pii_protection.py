"""
PII/PHI Protection Service with Masking/Unmasking
Integrates Microsoft Presidio for enterprise-grade PII detection
"""

import uuid
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from core.logger import get_logger
from services.llm_connection import client, MODEL_ID
from google.genai import types

logger = get_logger(__name__)

# ============================================================================
# PHARMACY-SPECIFIC PII/PHI RECOGNIZERS
# ============================================================================

def create_pharmacy_recognizers() -> List[PatternRecognizer]:
    """
    Create custom recognizers for pharmacy domain PII/PHI
    
    High scores (0.95) to prioritize over generic patterns like US_DRIVER_LICENSE
    
    Returns:
        List of PatternRecognizer objects for pharmacy-specific entities
    """
    
    recognizers = []
    
    # Member ID: M + 7-10 digits (HIGH PRIORITY)
    recognizers.append(PatternRecognizer(
        supported_entity="MEMBER_ID",
        patterns=[Pattern(
            name="member_id",
            regex=r"\b[Mm]\d{7,10}\b",
            score=0.95  # High score to override US_DRIVER_LICENSE
        )],
        context=["member", "patient", "subscriber", "ID"]
    ))
    
    # Claim ID: CLM-XXXXXX or CLAIM-XXXXXX or 15-digit numeric (HIGH PRIORITY)
    recognizers.append(PatternRecognizer(
        supported_entity="CLAIM_ID",
        patterns=[Pattern(
            name="claim_id",
            regex=r"\b((?:CLM|CLAIM)[-_]?\d{4,10}|\d{15})\b",
            score=0.95  # High score to override US_DRIVER_LICENSE
        )],
        context=["claim", "rejected", "approved", "status"]
    ))
    
    # RX Number: RX-XXXXXX or PRESCRIPTION-XXXXXX
    recognizers.append(PatternRecognizer(
        supported_entity="RX_NUMBER",
        patterns=[Pattern(
            name="rx_number",
            regex=r"\b(?:RX|PRESCRIPTION)[-_]?\d{4,10}\b",
            score=0.95
        )],
        context=["prescription", "medication", "refill"]
    ))
    
    # NDC: National Drug Code (XXXXX-XXXX-XX format)
    recognizers.append(PatternRecognizer(
        supported_entity="NDC",
        patterns=[Pattern(
            name="ndc",
            regex=r"\b\d{4,5}-\d{3,4}-\d{1,2}\b",
            score=0.95
        )],
        context=["drug", "ndc", "medication"]
    ))
    
    return recognizers


# ============================================================================
# PII/PHI PROTECTION SERVICE
# ============================================================================

class PIIProtectionService:
    """
    Complete PII/PHI protection service with detection, masking, and unmasking
    
    Features:
    - Detects standard PII (names, SSN, email, phone, etc.)
    - Detects pharmacy-specific PHI (member IDs, claim IDs, RX numbers)
    - Masks sensitive data with unique tokens
    - Checks for PII leakage in LLM responses
    - Unmasks tokens back to original values
    
    Usage:
        service = PIIProtectionService()
        
        # Mask input
        masked_text, metadata = service.mask_pii_phi(user_input, session_id)
        
        # Process with LLM (receives masked text)
        llm_response = call_llm(masked_text)
        
        # Check for leakage
        has_leak, leaked = service.check_leakage(llm_response, expected_tokens)
        
        # Unmask output
        final_response = service.unmask_pii_phi(llm_response, metadata["token_mapping"])
    """
    
    def __init__(self):
        """Initialize Presidio engines with custom pharmacy recognizers"""
        # Create registry with standard + custom recognizers
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        
        # Add pharmacy-specific recognizers
        for recognizer in create_pharmacy_recognizers():
            registry.add_recognizer(recognizer)
        
        self.analyzer = AnalyzerEngine(registry=registry)
        self.anonymizer = AnonymizerEngine()
        
        # Token storage for session cleanup
        self.token_storage: Dict[str, str] = {}
        
        logger.info("✅ PII Protection Service initialized with Presidio")
    
    def detect_pii_phi(self, text: str, language: str = "en") -> List[Dict]:
        """
        Detect all PII/PHI in text
        
        Filters overlapping detections by keeping only the highest-scoring one
        (e.g., MEMBER_ID score 0.95 beats US_DRIVER_LICENSE score 0.5)
        
        Args:
            text: Text to analyze
            language: Language code (default: "en")
            
        Returns:
            List of detected entities with metadata (highest-priority only)
        """
        results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=[
                # Standard PII
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                "US_PASSPORT", "US_DRIVER_LICENSE", "CREDIT_CARD",
                "DATE_TIME", "LOCATION", "MEDICAL_LICENSE",
                # Pharmacy-specific PHI (HIGH PRIORITY)
                "MEMBER_ID", "CLAIM_ID", "RX_NUMBER", "NDC"
            ]
        )
        
        # Convert to dict format
        detected = []
        for result in results:
            detected.append({
                "entity_type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "score": result.score,
                "text": text[result.start:result.end]
            })
        
        # Filter overlapping detections - keep highest score only
        filtered = []
        detected_sorted = sorted(detected, key=lambda x: x["score"], reverse=True)
        
        for entity in detected_sorted:
            # Check if this overlaps with any already-kept entity
            overlaps = False
            for kept in filtered:
                # Check if ranges overlap
                if not (entity["end"] <= kept["start"] or entity["start"] >= kept["end"]):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(entity)
        
        # Sort back by position
        filtered.sort(key=lambda x: x["start"])
        
        return filtered
    
    def mask_pii_phi(self, text: str, session_id: str) -> Tuple[str, Dict]:
        """
        Mask PII/PHI with unique tokens
        
        Args:
            text: Original text containing PII/PHI
            session_id: Session identifier for token storage
            
        Returns:
            (masked_text, metadata)
            - masked_text: Text with PII/PHI replaced by tokens
            - metadata: Dict with masking details and token mapping
        """
        detected = self.detect_pii_phi(text)
        
        if not detected:
            return text, {
                "has_pii": False,
                "masked_count": 0,
                "entities_detected": [],
                "token_mapping": {}
            }
        
        # Process entities in reverse order to maintain text indices
        masked_text = text
        token_mapping = {}
        
        detected_sorted = sorted(detected, key=lambda x: x["start"], reverse=True)
        
        for entity in detected_sorted:
            entity_type = entity["entity_type"]
            original_value = entity["text"]
            start = entity["start"]
            end = entity["end"]
            
            # Generate unique token: [ENTITY_TYPE_XXXXXXXX]
            token = f"[{entity_type}_{uuid.uuid4().hex[:8].upper()}]"
            
            # Store mapping for later unmasking
            storage_key = f"{session_id}:{token}"
            self.token_storage[storage_key] = original_value
            token_mapping[token] = {
                "original": original_value,
                "entity_type": entity_type,
                "position": (start, end)
            }
            
            # Replace in text
            masked_text = masked_text[:start] + token + masked_text[end:]
        
        logger.info(
            f"🎭 Masked {len(detected)} PII/PHI entities: "
            f"{[e['entity_type'] for e in detected]}"
        )
        
        metadata = {
            "has_pii": True,
            "masked_count": len(detected),
            "entities_detected": [e["entity_type"] for e in detected],
            "token_mapping": token_mapping,
            "original_length": len(text),
            "masked_length": len(masked_text)
        }
        
        return masked_text, metadata
    
    def check_leakage(
        self, 
        text: str, 
        expected_tokens: List[str],
        original_values: Optional[List[str]] = None,
        language: str = "en"
    ) -> Tuple[bool, List[Dict]]:
        """
        Check if text contains unexpected PII/PHI (leakage)
        
        Simple leak detection logic:
        1. Detect PII/PHI in LLM response
        2. Ignore masked tokens (expected - they came from input)
        3. Ignore user's own PII from input (expected - user's own data)
        4. If NEW PII found → It's a LEAK, BLOCK the response
        
        Args:
            text: Text to check (typically LLM response)
            expected_tokens: List of tokens that SHOULD be present (safe)
            original_values: List of original PII values from user input (for comparison)
            language: Language code (default: "en")
            
        Returns:
            (should_block, leaked_entities)
            - should_block: True if response should be blocked (new PII detected)
            - leaked_entities: List of leaked entity details
        """
        detected = self.detect_pii_phi(text, language)
        
        # Convert original_values to set for faster lookup
        original_values_set = set(original_values) if original_values else set()
        
        leaked = []
        for entity in detected:
            entity_text = entity["text"]
            
            # Check if it's an expected token (safe)
            # Check both ways: token in text OR text in token (handles cases where brackets are included/excluded)
            is_expected_token = any(
                token in entity_text or entity_text in token 
                for token in expected_tokens
            )
            
            if is_expected_token:
                # It's a token, not real PII - safe
                continue
            
            # It's real PII/PHI, not a token
            # Check if it matches the original input
            is_original_value = entity_text in original_values_set
            
            if not is_original_value:
                # This is NEW/DIFFERENT PII - LEAK!
                leaked.append({
                    "type": entity["entity_type"],
                    "value": entity_text,
                    "score": entity["score"],
                    "position": (entity["start"], entity["end"])
                })
            else:
                # This is the same PII from user's input - safe
                logger.info(
                    f"[SAFE] Found original PII in response: {entity['entity_type']} "
                    f"(user's own data - not a leak)"
                )
        
        if leaked:
            # NEW PII detected - this is a leak, BLOCK it
            logger.error(
                f"[LEAK DETECTED] LLM response contains NEW PII/PHI: "
                f"{[e['type'] + '=' + e['value'] for e in leaked]}"
            )
            return True, leaked  # Block the response
        else:
            logger.info("[OK] No PII/PHI leakage detected")
            return False, []  # All good
    
    def unmask_pii_phi(self, masked_text: str, token_mapping: Dict[str, Dict]) -> str:
        """
        Restore original PII/PHI values
        
        Args:
            masked_text: Text containing tokens
            token_mapping: Mapping from mask operation (token -> original value)
            
        Returns:
            Text with tokens replaced by original values
        """
        unmasked_text = masked_text
        
        # Replace each token with its original value
        for token, info in token_mapping.items():
            original_value = info["original"]
            unmasked_text = unmasked_text.replace(token, original_value)
        
        logger.info(f"🔓 Unmasked {len(token_mapping)} tokens")
        
        return unmasked_text
    
    def mask_api_response(
        self, 
        api_response: Dict[str, Any], 
        session_id: str,
        existing_token_mapping: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict]:
        """
        Mask PII/PHI in API response data
        
        This is used to mask data coming FROM APIs before sending to LLM.
        It preserves existing token mappings and creates new ones for new PII.
        
        Args:
            api_response: Dictionary with API data (may contain PII)
            session_id: Session identifier
            existing_token_mapping: Tokens from user input (to reuse if same PII)
            
        Returns:
            (masked_response, updated_token_mapping)
            - masked_response: Dict with PII replaced by tokens
            - updated_token_mapping: Combined old + new token mappings
        """
        import json
        
        # Convert dict to string for PII detection
        response_str = json.dumps(api_response, indent=2)
        
        # Detect all PII in the response
        detected = self.detect_pii_phi(response_str)
        
        if not detected:
            return api_response, existing_token_mapping or {}
        
        # Initialize mapping with existing tokens
        combined_mapping = existing_token_mapping.copy() if existing_token_mapping else {}
        
        # Build reverse lookup: original_value → token
        reverse_lookup = {}
        for token, info in combined_mapping.items():
            reverse_lookup[info["original"]] = token
        
        # Process detected PII
        for entity in detected:
            original_value = entity["text"]
            entity_type = entity["entity_type"]
            
            # Check if we already have a token for this value
            if original_value in reverse_lookup:
                # Reuse existing token (same PII from user input)
                token = reverse_lookup[original_value]
                logger.info(f"♻️  Reusing token {token} for {entity_type}")
            else:
                # Create new token for API response PII
                token = f"[{entity_type}_{uuid.uuid4().hex[:8].upper()}]"
                
                # Store mapping
                storage_key = f"{session_id}:{token}"
                self.token_storage[storage_key] = original_value
                combined_mapping[token] = {
                    "original": original_value,
                    "entity_type": entity_type,
                    "source": "api_response"  # Track where it came from
                }
                reverse_lookup[original_value] = token
                logger.info(f"🆕 Created token {token} for API PII {entity_type}")
        
        # Replace PII in the response dict (recursive)
        masked_response = self._replace_pii_in_dict(
            api_response, 
            reverse_lookup
        )
        
        logger.info(f"🎭 Masked API response: {len(detected)} PII entities, {len(combined_mapping)} total tokens")
        
        return masked_response, combined_mapping
    
    def _replace_pii_in_dict(
        self, 
        data: Any, 
        reverse_lookup: Dict[str, str]
    ) -> Any:
        """
        Recursively replace PII values in nested dict/list structures
        
        Args:
            data: Dict, list, or primitive value
            reverse_lookup: Map of original_value → token
            
        Returns:
            Same structure with PII replaced by tokens
        """
        if isinstance(data, dict):
            return {
                key: self._replace_pii_in_dict(value, reverse_lookup)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [
                self._replace_pii_in_dict(item, reverse_lookup)
                for item in data
            ]
        elif isinstance(data, str):
            # Replace any PII values found in this string
            result = data
            for original, token in reverse_lookup.items():
                if original in result:
                    result = result.replace(original, token)
            return result
        else:
            # Primitive types (int, float, bool, None)
            return data
    
    def cleanup_session(self, session_id: str):
        """
        Remove token mappings for a session (memory cleanup)
        
        Args:
            session_id: Session identifier to clean up
        """
        keys_to_remove = [
            k for k in self.token_storage.keys()
            if k.startswith(f"{session_id}:")
        ]
        
        for key in keys_to_remove:
            del self.token_storage[key]
        
        logger.info(f"🧹 Cleaned {len(keys_to_remove)} tokens for session {session_id}")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_pii_service: Optional[PIIProtectionService] = None

def get_pii_service() -> PIIProtectionService:
    """
    Get singleton instance of PII protection service
    
    Returns:
        PIIProtectionService instance
    """
    global _pii_service
    if _pii_service is None:
        _pii_service = PIIProtectionService()
    return _pii_service


# ============================================================================
# SAFETY CHECK CLASS - Unified Safety with PII Protection
# ============================================================================

class SafetyCheck:
    """
    Unified safety checking with PII protection
    
    Three-method structure:
    1. _check_violence_patterns() - Fast pattern-based threat detection (private)
    2. check_with_gemini_filters() - AI-based content safety with PII protection
    3. check_harmful_content() - Orchestrator that runs both checks
    
    Architecture:
        Input Query
            ↓
        [Method 1] Violence pattern check (fast, local)
            ↓
        [Method 2] Mask PII → Gemini filters → Unmask
            ↓
        [Method 3] Return safe query with PII intact
    """
    
    def __init__(self):
        self.pii_service = get_pii_service()
        logger.info("✅ SafetyCheck initialized")
    
    def _check_violence_patterns(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Private method: Fast pattern-based threat detection
        
        Comprehensive context-aware threat detection
        
        ROOT CAUSE SOLUTION based on TEST_QUERIES.txt analysis:
        - Covers ALL threat categories from test file
        - Uses PHRASE patterns (not single keywords)
        - Excludes medical/pharmacy contexts automatically
        - Handles obfuscation attempts (leetspeak, spacing, symbols)
        
        Threat Categories Covered:
        1. Direct Violence/Threats
        2. Weapon Threats
        3. Self-Harm Threats
        4. Harassment with Profanity
        5. Hate Speech
        6. Sexual Harassment
        7. Obfuscation Attempts
        
        Args:
            text: User input text
            
        Returns:
            (is_threat, threat_reason)
            - is_threat: True if threat detected
            - threat_reason: Explanation if threat detected
        """
        import re
        
        text_lower = text.lower()
        
        # Check for medical context FIRST (to exclude false positives)
        medical_indicators = [
            'bacteria', 'infection', 'virus', 'cancer', 'cells', 'tumor',
            'disease', 'medication', 'drug', 'treatment', 'therapy', 'antibiotic',
            'medicine', 'prescription', 'symptom', 'diagnosis', 'condition',
            'patient', 'dose', 'side effect', 'reaction', 'allergy'
        ]
        
        has_medical_context = any(indicator in text_lower for indicator in medical_indicators)
        
        # Check for common idioms/phrases that should NOT be blocked
        safe_idioms = [
            'kill time', 'kill two birds', 'dressed to kill', 'killer deal',
            'killing it', 'killer app', 'drop dead gorgeous', 'to die for',
            'die hard', 'die laughing', 'dying to', 'dying of boredom',
            'dying of laughter', 'killer instinct', 'lady killer',
            'born killer', 'natural born', 'license to kill', 'kill bill',
            'killing eve', 'kill the lights', 'kill switch'
        ]
        
        has_safe_idiom = any(idiom in text_lower for idiom in safe_idioms)
        
        # If medical context or safe idiom, skip threat detection
        if has_medical_context or has_safe_idiom:
            return False, None
        
        # Comprehensive threat patterns
        threat_patterns = [
            # CATEGORY 1: DIRECT VIOLENCE/MURDER THREATS
            (r"\bi\s*['']?\s*(will|ll|am\s+going\s+to|m\s+gonna|gonna)\s+(kill|murder|hurt|harm|attack)", 
             "direct violence threat"),
            
            # CATEGORY 2: SHOOTING THREATS
            (r"\b(i\s*['']?\s*(will|ll|m|am)|gonna|going\s+to)\s+(shoot|shot)\s+(up|at|into|you|them|someone|everyone)", 
             "shooting threat"),
            (r"\bshoot\s+up\s+(the\s+)?(place|pharmacy|store|building|office)", 
             "shooting threat"),
            
            # CATEGORY 3: BOMBING/EXPLOSION THREATS
            (r"\b(i\s*['']?\s*(will|ll|m)|gonna|going\s+to)\s+(bomb|blow\s+up|explode)", 
             "bombing threat"),
            (r"\b(bomb|blow\s+up|explode)\s+(the\s+)?(place|pharmacy|store|building|office|your)", 
             "bombing threat"),
            
            # CATEGORY 4: WEAPON THREATS
            (r"\bi\s+(have|got|brought|own)\s+(a\s+|my\s+)?(gun|knife|weapon|bomb|rifle|pistol)", 
             "weapon threat"),
            (r"\bi\s*['']?\s*m\s+armed", 
             "weapon threat"),
            (r"\b(use|bring)\s+(my|a)\s+(gun|knife|weapon)", 
             "weapon threat"),
            
            # CATEGORY 5: PHYSICAL ATTACK THREATS
            (r"\b(will|ll|gonna|going\s+to)\s+attack\s+(the\s+)?(pharmacy|store|place|building|office|your|you|them)", 
             "attack threat"),
            
            # CATEGORY 6: SELF-HARM THREATS
            (r"\bi\s*['']?\s*(will|ll|m\s+going\s+to|m\s+gonna)\s+(hurt|kill|harm)\s+myself", 
             "self-harm threat"),
            (r"\bi\s*['']?\s*(will|ll|m\s+going\s+to)\s+(commit\s+suicide|end\s+it\s+all)", 
             "self-harm threat"),
            
            # CATEGORY 7: THREATS TO FAMILY/OTHERS
            (r"\b(harm|hurt|kill|murder|attack)\s+your\s+(family|kids|children|staff|employees|office|people)", 
             "threat to others"),
            
            # CATEGORY 8: CONDITIONAL THREATS
            (r"\bor\s+i\s*['']?\s*(will|ll)\s+(kill|harm|hurt|shoot|bomb|attack|murder)", 
             "conditional threat"),
            (r"\belse\s+i\s*['']?\s*(will|ll)\s+(kill|harm|hurt|shoot|bomb|attack|murder)", 
             "conditional threat"),
            
            # CATEGORY 9: PROFANITY + AGGRESSION
            (r"\b(f\*+k|f\*+king|fuc?k|fuq|fuk|f\*+ing)\s+(you|idiots|morons|bastards|assholes)", 
             "harassment"),
            (r"\byou\s+(morons|idiots|stupid\s+bastards|dumbasses|assholes)", 
             "harassment"),
            
            # CATEGORY 10: HATE SPEECH
            (r"\b(all\s+)?(women|men|jews|muslims|christians|blacks|whites|asians|latinos|hispanics)\s+(should\s+die|are\s+worthless|deserve\s+to\s+die)", 
             "hate speech"),
            (r"\bi\s+hate\s+(all\s+)?(women|men|jews|muslims|christians|blacks|whites|asians|latinos|hispanics)", 
             "hate speech"),
            
            # CATEGORY 11: SEXUAL HARASSMENT
            (r"\b(hey|hi)\s+(sexy|hot|beautiful|gorgeous|babe|baby)", 
             "sexual harassment"),
            (r"\byou\s*['']?\s*re\s+(hot|sexy|beautiful)", 
             "sexual harassment"),
            (r"\bnice\s+(body|ass|tits|boobs|legs)", 
             "sexual harassment"),
            (r"\bsend\s+(nudes|pics|pictures)", 
             "sexual harassment"),
            
            # CATEGORY 12: OBFUSCATION - LEETSPEAK
            (r"\bi\s*['']?\s*(will|ll)\s+(k[1i!]ll|h[4a]rm|murd[3e]r|k[1i!]l{2})", 
             "obfuscated threat"),
            
            # CATEGORY 13: OBFUSCATION - SYMBOL SUBSTITUTION  
            (r"\bi\s*['']?\s*(will|ll)\s+(k\*+l{1,2}|h[@a]rm|k\*{2,}l|att[@a]ck)", 
             "obfuscated threat"),
            
            # CATEGORY 14: OBFUSCATION - SPACING
            (r"\bi\s+(w i l l|w\s+i\s+l{2})\s+(k\s+i\s+l{2}|h\s+a\s+r\s+m|m\s+u\s+r\s+d\s+e\s+r)", 
             "obfuscated threat"),
        ]
        
        # Check each pattern
        for pattern, threat_type in threat_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(f"🚫 Threat detected: {threat_type}")
                logger.warning(f"   Pattern matched: {pattern[:50]}...")
                logger.warning(f"   Query: {text[:100]}...")
                return True, f"Your message contains threatening language ({threat_type}). This violates our terms of service."
        
        return False, None
    
    def check_violence_patterns(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Public method: Fast pattern-based threat detection
        
        Wrapper around _check_violence_patterns() for external use
        
        Args:
            text: User input text
            
        Returns:
            (is_safe, block_reason)
            - is_safe: False if threat detected
            - block_reason: Explanation if blocked
        """
        logger.info("🔍 [Method 1] Running violence pattern check...")
        is_threat, threat_reason = self._check_violence_patterns(text)
        
        if is_threat:
            logger.warning(f"🚫 [Method 1] Threat detected: {threat_reason}")
            return False, threat_reason
        
        logger.info("✅ [Method 1] No threat patterns detected")
        return True, None
    
    async def check_with_gemini_filters(
        self, 
        text: str, 
        session_id: str
    ) -> Dict[str, Any]:
        """
        Method 2: AI-based content safety with PII protection
        
        Steps:
        1. Detect & Mask PII/PHI (local, no external calls)
        2. Call Gemini safety filters (with masked data - no PII leakage)
        3. Unmask PII/PHI (restore original values)
        4. Return result with original PII/PHI
        
        Args:
            text: User input text
            session_id: Session identifier for token storage
            
        Returns:
            {
                "is_safe": bool,
                "text": str (unmasked if safe),
                "reason": str (if blocked),
                "pii_metadata": dict,
                "violation_categories": list (if blocked)
            }
        """
        logger.info("🔍 [Method 2] Running Gemini filters with PII protection...")
        
        try:
            # Step 1: Detect & Mask PII/PHI
            logger.info("   Step 1: Masking PII/PHI before Gemini call")
            masked_text, pii_metadata = self.pii_service.mask_pii_phi(text, session_id)
            
            masked_count = pii_metadata.get("masked_count", 0)
            if masked_count > 0:
                logger.info(f"   🎭 Masked {masked_count} PII/PHI entities")
                logger.debug(f"   Original: {text[:100]}...")
                logger.debug(f"   Masked: {masked_text[:100]}...")
            else:
                logger.info("   ℹ️  No PII/PHI detected")
            
            # Step 2: Call Gemini safety filters (with masked data)
            logger.info("   Step 2: Calling Gemini safety filters with masked data")
            
            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_LOW_AND_ABOVE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_LOW_AND_ABOVE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_LOW_AND_ABOVE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_LOW_AND_ABOVE"
                ),
            ]
            
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part(text=masked_text)]  # Masked text - no PII exposure
                )],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    safety_settings=safety_settings,
                    max_output_tokens=10
                )
            )
            
            # Check if response was blocked
            violation_categories = []
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = str(candidate.finish_reason)
                    if 'SAFETY' in finish_reason:
                        logger.warning(f"   🚫 Content blocked by Gemini: {finish_reason}")
                        
                        # Get specific violation categories
                        if hasattr(candidate, 'safety_ratings'):
                            for rating in candidate.safety_ratings:
                                if hasattr(rating, 'probability'):
                                    prob = str(rating.probability)
                                    if prob in ['HIGH', 'MEDIUM']:
                                        category = str(rating.category) if hasattr(rating, 'category') else 'UNKNOWN'
                                        violation_categories.append(category)
                        
                        return {
                            "is_safe": False,
                            "text": text,  # Return original (unmasked) for context
                            "reason": "Content blocked by AI safety filters",
                            "violation_categories": violation_categories,
                            "pii_metadata": pii_metadata
                        }
            
            # Step 3: Unmask PII/PHI (restore original values)
            logger.info("   Step 3: Unmasking PII/PHI - restoring original values")
            token_mapping = pii_metadata.get("token_mapping", {})
            unmasked_text = self.pii_service.unmask_pii_phi(masked_text, token_mapping)
            
            logger.info("   ✅ Gemini filters passed - returning unmasked text")
            
            # Step 4: Return safe result with original PII/PHI
            return {
                "is_safe": True,
                "text": unmasked_text,  # Original text with PII/PHI intact
                "pii_metadata": pii_metadata
            }
            
        except Exception as e:
            logger.error(f"   ❌ Gemini safety check failed: {e}")
            # Fail closed - block on error
            return {
                "is_safe": False,
                "text": text,
                "reason": f"Safety check error: {str(e)}",
                "error": str(e)
            }
    
    async def check_harmful_content(
        self, 
        text: str, 
        session_id: str
    ) -> Dict[str, Any]:
        """
        Method 3: Complete safety pipeline (orchestrator)
        
        Runs both safety layers:
        1. Violence pattern check (fast)
        2. Gemini filters with PII protection (comprehensive)
        
        Args:
            text: User input text
            session_id: Session identifier
            
        Returns:
            {
                "is_safe": bool,
                "text": str (original with PII/PHI if safe),
                "reason": str (if blocked),
                "pii_metadata": dict
            }
        """
        logger.info("="*60)
        logger.info("🛡️  [Method 3] COMPLETE SAFETY CHECK PIPELINE")
        logger.info("="*60)
        
        # Layer 1: Violence pattern check (fast, local)
        logger.info("Layer 1: Violence pattern detection")
        is_safe_patterns, pattern_reason = self.check_violence_patterns(text)
        if not is_safe_patterns:
            logger.warning(f"❌ Safety check FAILED - Pattern violation")
            return {
                "is_safe": False,
                "text": text,
                "reason": pattern_reason
            }
        
        # Layer 2: Gemini filters with PII protection (AI-based, comprehensive)
        logger.info("Layer 2: Gemini filters with PII protection")
        gemini_result = await self.check_with_gemini_filters(text, session_id)
        
        if not gemini_result["is_safe"]:
            logger.warning(f"❌ Safety check FAILED - Gemini violation")
            logger.info("="*60)
            return gemini_result
        
        logger.info("✅ Safety check PASSED - Query is safe")
        logger.info("="*60)
        return gemini_result


# Singleton instance for SafetyCheck
_safety_checker: Optional[SafetyCheck] = None

def get_safety_checker() -> SafetyCheck:
    """
    Get singleton instance of SafetyCheck
    
    Returns:
        SafetyCheck instance
    """
    global _safety_checker
    if _safety_checker is None:
        _safety_checker = SafetyCheck()
    return _safety_checker

