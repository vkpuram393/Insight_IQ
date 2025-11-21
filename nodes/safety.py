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

from typing import Dict, Any
from langgraph.graph import END
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger
from core.logging_context import extract_logging_context, log_state_snapshot
from core.pii_protection import (
    get_safety_checker,
    get_pii_service
)

logger = get_logger(__name__)

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
        - threat_detected: bool
        - threat_reason: str (if blocked)
        - response: error message (if blocked)
        - metadata.pii_metadata: PII detection metadata
    
    FLOW:
        Blocked → END (safety violation)
        Passed → Continue with PII/PHI intact for downstream nodes
    """
    node_name = "safety_precheck"
    
    logger.info("\n" + "="*70)
    logger.info("🛡️  SAFETY PRECHECK NODE - Unified Safety Check")
    logger.info("="*70)
    
    if not settings.enable_safety_precheck:
        logger.info("⭐ Safety precheck disabled in config")
        result = {"safety_precheck_passed": True}
        await log_state_snapshot(state, node_name, result)
        return result
    
    text = state.get("text", "")
    session_id = state.get("session_id", "default")
    
    try:
        # Get safety checker instance
        safety_checker = get_safety_checker()
        
        # Run complete safety pipeline (Method 3)
        result = await safety_checker.check_harmful_content(text, session_id)
        
        if not result["is_safe"]:
            # Safety violation detected
            reason = result.get("reason", "Content safety violation")
            violation_categories = result.get("violation_categories", [])
            
            logger.warning(f"🚫 BLOCKED: {reason}")
            if violation_categories:
                logger.warning(f"   Categories: {', '.join(violation_categories)}")
            
            result_dict = {
                "safety_precheck_passed": False,
                "threat_detected": True,
                "threat_reason": reason,
                "response": (
                    "I'm here to help with pharmacy claims and coverage questions. "
                    "I can't assist with that type of request."
                )
            }
            await log_state_snapshot(state, node_name, result_dict)
            return result_dict
        
        # Safety check passed - return unmasked text with PII/PHI
        logger.info("✅ Safety check passed - Query is safe")
        logger.info(f"   PII/PHI intact for downstream processing")
        
        # Store PII metadata for later use
        metadata = state.get("metadata", {})
        metadata["pii_metadata"] = result.get("pii_metadata", {})
        
        result_dict = {
            "text": result["text"],  # Unmasked text with PII/PHI intact
            "safety_precheck_passed": True,
            "threat_detected": False,
            "threat_reason": None,
            "metadata": metadata
        }
        await log_state_snapshot(state, node_name, result_dict)
        return result_dict
        
    except Exception as e:
        logger.error(f"❌ Safety precheck failed: {e}", exc_info=True)
        # Fail closed - block on error
        return {
            "safety_precheck_passed": False,
            "threat_detected": True,
            "threat_reason": f"Safety check error: {str(e)}",
            "response": (
                "I'm unable to process your request at this time. "
                "Please try again later."
            )
        }


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
        - session_id: For token storage
    
    OUTPUT (to state):
        - text: MASKED text
        - tool_results: MASKED (if contains PII/PHI)
        - metadata.response_pii_masking: Masking metadata
    
    FLOW:
        Always continues
        Response agent works with MASKED data (safe)
    """
    node_name = "response_safety_pii_precheck"
    
    logger.info("\n" + "="*70)
    logger.info("🔐 RESPONSE SAFETY PII PRECHECK - Masking before Response LLM")
    logger.info("="*70)
    
    text = state.get("text", "")
    tool_results = state.get("tool_results", {})
    session_id = state.get("session_id", "default")
    
    try:
        pii_service = get_pii_service()
        
        # Mask text
        masked_text, text_metadata = pii_service.mask_pii_phi(text, session_id)
        
        # Mask tool results (convert to string, mask, keep structure)
        tool_results_str = str(tool_results)
        masked_tool_results_str, tool_metadata = pii_service.mask_pii_phi(tool_results_str, session_id)
        
        total_masked = text_metadata["masked_count"] + tool_metadata["masked_count"]
        
        if total_masked > 0:
            logger.info(f"🎭 Masked {total_masked} PII/PHI entities before response LLM")
            logger.debug(f"   Original text: {text[:100]}...")
            logger.debug(f"   Masked text: {masked_text[:100]}...")
        else:
            logger.info("ℹ️  No PII/PHI detected - data unchanged")
        
        # Store metadata for unmasking
        metadata = state.get("metadata", {})
        metadata["response_pii_masking"] = {
            "text_metadata": text_metadata,
            "tool_metadata": tool_metadata
        }
        
        result = {
            "text": masked_text,
            "metadata": metadata
        }
        await log_state_snapshot(state, node_name, result)
        return result
        
    except Exception as e:
        logger.error(f"❌ Response PII masking failed: {e}", exc_info=True)
        # Fail-safe: continue with original text (not ideal but prevents blocking)
        result = {
            "metadata": {
                **state.get("metadata", {}),
                "response_pii_masking": {
                    "has_pii": False,
                    "masked_count": 0,
                    "error": str(e)
                }
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
        - safety_postcheck_passed: bool
        - metadata.response_pii_unmasking: Unmasking stats
        - metadata.leakage_check: Leakage detection results
    
    FLOW:
        Leakage detected → Block response, return generic message
        No leakage → Unmask tokens, return to user
    """
    node_name = "response_safety_pii_postcheck"
    
    logger.info("\n" + "="*70)
    logger.info("🔍 RESPONSE SAFETY PII POSTCHECK - Leakage Check + Unmasking")
    logger.info("="*70)
    
    response = state.get("response", "")
    
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
            # Check if this PII value was in original input
            is_expected = any(
                entity_text == data["original"] 
                for data in combined_token_mapping.values()
            )
            if not is_expected:
                # NEW PII detected - this is a leak!
                leaked_entities.append(entity)
        
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
        
        tokens_unmasked = sum(1 for token in combined_token_mapping.keys() if token in response)
        
        logger.info(f"🔓 Unmasked {tokens_unmasked} tokens in final response")
        logger.debug(f"   Masked: {response[:100]}...")
        logger.debug(f"   Unmasked: {unmasked_response[:100]}...")
        
        result = {
            "response": unmasked_response,  # ← CRITICAL: Replace with unmasked
            "safety_postcheck_passed": True,
            "metadata": {
                **metadata,
                "leakage_check": {"has_leakage": False},
                "response_pii_unmasking": {
                    "tokens_unmasked": tokens_unmasked,
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
        logger.error(f"❌ Response postcheck failed: {e}", exc_info=True)
        # Fail-safe: Return response as-is (may contain tokens)
        return {
            "response": response,
            "safety_postcheck_passed": True,
            "metadata": {
                **state.get("metadata", {}),
                "response_pii_unmasking": {"error": str(e)}
            }
        }


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
