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
    Determine if a detected PII entity is contextual data vs actual leakage
    
    Allows common contextual patterns that LLMs generate in normal responses:
    - Generic location terms (state abbreviations, generic terms)
    - API versions, error codes, status words
    - Pharmaceutical terms (NDC codes, drug names)
    - Truncated terms with "..." that are clearly non-personal
    - Short numeric codes and ZIP codes
    
    Args:
        entity_text: The detected PII text
        entity_type: The type of PII entity
        
    Returns:
        True if this is likely contextual data, False if it's potential leakage
    """
    import re
    
    # PERSON entities: Allow generic drug/medical terms that get misclassified
    if entity_type == "PERSON":
        generic_terms = [
            'pharmacy', 'pharmacist', 'prescriber', 'patient', 'member',
            'doctor', 'physician', 'nurse', 'provider', 'caregiver',
            'male', 'female', 'adult', 'child', 'senior',
            # Common drug/system terms misclassified as PERSON
            'generic', 'brand', 'retail', 'mail', 'specialty'
        ]
        if entity_text.lower() in generic_terms:
            return True
        
        # CRITICAL: Allow field names (camelCase, snake_case) misclassified as PERSON
        # E.g., "groupNumber", "member_id", "claim_id"
        field_name_patterns = [
            r'^[a-z]+[A-Z]',  # camelCase (groupNumber, memberId)
            r'^\w+_\w+$',      # snake_case (group_number, member_id)
            r'^[A-Z_]+$',      # CONSTANT_CASE (GROUP_NUMBER)
        ]
        for pattern in field_name_patterns:
            if re.match(pattern, entity_text):
                return True
        
        # CRITICAL: Allow masked token IDs misclassified as PERSON
        # E.g., "NDC_D0B47B08", "CLAIM_ID_ABC123" (without brackets)
        if re.match(r'^[A-Z_]+_[A-F0-9]{8}$', entity_text):
            return True
        
        # Allow truncated contextual terms (but NOT names with "...")
        if entity_text.endswith('...'):
            base_text = entity_text[:-3].lower()
            # Only allow clearly non-personal contextual terms
            safe_bases = ['pharmacy', 'drug', 'medication', 'prescription', 'claim']
            if any(base in base_text for base in safe_bases):
                return True
            # Allow token IDs with "..." (truncated in logs)
            if re.match(r'^[A-Z_]+_[A-F0-9]+$', base_text.replace('_', '_').upper()):
                return True
    
    # LOCATION entities: Allow generic terms and state codes
    elif entity_type == "LOCATION":
        # Allow US state abbreviations (2 letters)
        if len(entity_text) == 2 and entity_text.isupper():
            return True
        
        # Allow generic location terms
        generic_locations = ['usa', 'us', 'united states', 'pharmacy', 'store', 'retail']
        if entity_text.lower() in generic_locations:
            return True
        
        # Allow truncated ZIP codes (e.g., "35115...")
        if re.match(r'^\d{5}\.\.\.$', entity_text):
            return True
    
    # US_DRIVER_LICENSE: These are often false positives for system codes
    elif entity_type == "US_DRIVER_LICENSE":
        # Allow short alphanumeric codes (likely system codes, not real licenses)
        if len(entity_text) <= 4:
            return True
        
        # Allow API versions (v1, v2, etc.)
        if re.match(r'^v\d+$', entity_text.lower()):
            return True
    
    # NDC codes: These are drug codes, not personal information
    elif entity_type == "NDC":
        # Allow all NDC codes (they're pharmaceutical identifiers, not PII)
        return True
    
    # CLAIM_ID: Allow short numbers (likely sequence numbers, not real claim IDs)
    # Real claim IDs are typically 15+ digits, sequences are 3 digits
    elif entity_type == "CLAIM_ID":
        # Extract just the numeric part if prefixed with "claim"
        import re
        numbers = re.findall(r'\d+', entity_text)
        if numbers:
            # If the number is short (less than 6 digits), it's likely a sequence number
            max_digits = max(len(n) for n in numbers)
            if max_digits < 6:
                return True
        # Also allow if it matches sequence patterns (3 digits)
        if re.search(r'\b\d{1,5}\b', entity_text) and not re.search(r'\d{10,}', entity_text):
            return True
    
    # Short codes that are likely system codes
    if len(entity_text) <= 3:
        return True
    
    return False


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
        masked_history = []
        history_masked_count = 0
        if conversation_history:
            logger.info(f"🔍 Masking conversation history ({len(conversation_history)} messages)...")
            current_token_mapping = tool_metadata["token_mapping"].copy()
            
            for msg in conversation_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if content:
                    # Mask content using existing tokens to avoid duplicate masking
                    masked_content, _ = pii_service.mask_pii_phi(content, session_id)
                    
                    # Count how many NEW PII entities were found in history
                    # (Don't double-count if already in tool_metadata)
                    content_pii = pii_service.detect_pii_phi(content)
                    history_masked_count += len(content_pii)
                    
                    masked_history.append({
                        "role": role,
                        "content": masked_content
                    })
                else:
                    masked_history.append(msg)
            
            logger.info(f"🎭 Masked {history_masked_count} PII/PHI entities in conversation history")
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
        
        # Fail-safe: continue with original text (not ideal but prevents blocking)
        result = {
            "metadata": {
                **state.get("metadata", {}),
                "response_pii_masking": {
                    "has_pii": False,
                    "masked_count": 0,
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
        logger.warning("⚠️  No response to postcheck")
        result = {"safety_postcheck_passed": True}
        await log_state_snapshot(state, node_name, result)
        return result
    
    try:
        pii_service = get_pii_service()
        metadata = state.get("metadata", {})
        response_pii_masking = metadata.get("response_pii_masking", {})
        
        # Get token mappings
        text_metadata = response_pii_masking.get("text_metadata", {})
        tool_metadata = response_pii_masking.get("tool_metadata", {})
        
        text_token_mapping = text_metadata.get("token_mapping", {})
        tool_token_mapping = tool_metadata.get("token_mapping", {})
        
        # Combine token mappings
        combined_token_mapping = {**text_token_mapping, **tool_token_mapping}
        
        # ===== STEP 1: Check for PII Leakage =====
        logger.info("Step 1: Leakage detection")
        
        # Detect any NEW PII in response (not from original input)
        detected_pii = pii_service.detect_pii_phi(response)
        
        leaked_entities = []
        for entity in detected_pii:
            entity_text = entity["text"]
            entity_type = entity["entity_type"]
            
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
            
            # STEP 2b: For PERSON entities, also check name variations (first/last order)
            # E.g., "Chloe Roberts" in API but "Roberts Chloe" in response
            if not is_expected and entity_type == "PERSON" and ' ' in entity_text:
                parts = entity_text.split()
                if len(parts) == 2:
                    # Try reversed name order
                    reversed_name = f"{parts[1]} {parts[0]}"
                    is_expected = any(
                        reversed_name == data["original"]
                        for data in combined_token_mapping.values()
                        if data.get("entity_type") == "PERSON"
                    )
            
            # STEP 3: Check if this is contextual data (generic terms, system codes, etc.)
            is_contextual = _is_contextual_entity(entity_text, entity_type)
            
            if not is_expected and not is_contextual:
                # NEW PII detected that isn't contextual - this is a REAL leak!
                logger.warning(f"   ⚠️  Potential leak: {entity_type} = {entity_text[:20]}...")
                leaked_entities.append(entity)
            else:
                if is_expected:
                    logger.debug(f"   ✓ Expected PII: {entity_type} (from input or API data)")
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
            logger.info("ℹ️  No tokens to unmask")
            result = {
                "safety_postcheck_passed": True,
                "metadata": {
                    **metadata,
                    "leakage_check": {"has_leakage": False},
                    "response_pii_unmasking": {"tokens_unmasked": 0}
                }
            }
            await log_state_snapshot(state, node_name, result)
            return result
        
        # Unmask tokens in response
        unmasked_response = pii_service.unmask_pii_phi(response, combined_token_mapping)
        
        # CRITICAL: Also unmask text field so conversation history stores unmasked data
        text = state.get("text", "")
        unmasked_text = pii_service.unmask_pii_phi(text, text_token_mapping) if text_token_mapping else text
        
        tokens_unmasked = sum(1 for token in combined_token_mapping.keys() if token in response)
        text_tokens_unmasked = sum(1 for token in text_token_mapping.keys() if token in text) if text_token_mapping else 0
        
        logger.info(f"🔓 Unmasked {tokens_unmasked} tokens in final response")
        if text_tokens_unmasked > 0:
            logger.info(f"🔓 Unmasked {text_tokens_unmasked} tokens in text field (for conversation history)")
        logger.debug(f"   Masked response: {response[:100]}...")
        logger.debug(f"   Unmasked response: {unmasked_response[:100]}...")
        
        result = {
            "text": unmasked_text,  # ← CRITICAL: Unmask text so conversation history stores real values
            "response": unmasked_response,  # ← CRITICAL: Replace with unmasked
            "safety_postcheck_passed": True,
            "metadata": {
                **metadata,
                "leakage_check": {"has_leakage": False},
                "response_pii_unmasking": {
                    "tokens_unmasked": tokens_unmasked,
                    "text_tokens_unmasked": text_tokens_unmasked,
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