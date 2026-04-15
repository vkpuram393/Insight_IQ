
import json
import traceback
from pathlib import Path
from typing import Dict, Any
from state.schema import AgentState
from config.config import settings
from core.logger import get_logger
from core.errors.models import create_internal_error
from core.logging_context import extract_logging_context, log_state_snapshot
from persistence import PersistenceStoreFactory

logger = get_logger(__name__)

def _load_config() -> Dict[str, Any]:
    """Load domain config from JSON file"""
    config_path = Path(__file__).parent.parent / "config" / "domain_config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Return defaults
        return {
            "clarification_messages": {
                "low_confidence": "I'm not quite sure what you're asking. Could you rephrase your question?",
                "missing_entity_template": "Could you provide your {missing_entity}?"
            }
        }

async def clarification_node(state: AgentState) -> Dict[str, Any]:
    """
    Set clarification context for LLM-based follow-up question generation
    
    🎓 CONCEPT:
    When confidence is low or entities missing, prepare context for response agent
    to generate an intelligent follow-up question using LLM.
    
    This node:
    - Determines WHY clarification is needed
    - Prepares clarification_context with relevant information
    - Sets needs_clarification flag for response agent
    
    INPUT (from state):
        - text: User's query
        - intent: Classified intent
        - confidence: Classification confidence
        - missing_slots: Required entities that are missing
        - metadata: Additional context
    
    OUTPUT (to state):
        - needs_clarification: True (tells response agent to generate question)
        - clarification_context: Dict with reason, missing info, etc.
        - metadata: Updated with clarification trigger info
    
    FLOW:
        After this → response_safety_pii_precheck → response_agent (generates question)
        → response_safety_pii_postcheck → update_memory → END
    """
    node_name = "clarification"
    log_ctx = extract_logging_context(state)
    
    try:
        logger.info("❓ Node: Clarification (preparing for LLM-based question generation)")
        
        # Extract state values
        text = state.get("text", "")
        intent = state.get("intent", "unknown")
        confidence = state.get("confidence", 0.0)
        missing_slots = state.get("missing_slots", [])
        entities = state.get("entities", {})
        metadata = state.get("metadata", {})
        
        # Load config for clarification templates
        config = _load_config()
        confidence_threshold = config.get("confidence_threshold", 0.7)
        
        # Determine clarification reason
        # Check if user provided an invalid format claim ID (potential_claim_id detected)
        claim_id_format_invalid = entities.get("claim_id_format_invalid", False)
        potential_claim_ids = entities.get("potential_claim_ids", [])
        
        if missing_slots:
            reason = "missing_entity"
            # Enhanced message if user provided invalid format claim ID
            # Check for any claim-related slot name (claim_ids, claim_number, claim_numbers)
            claim_related_slots = ['claim_ids', 'claim_number', 'claim_numbers','claim_id','claimId']
            has_missing_claim_slot = any(slot in missing_slots for slot in claim_related_slots)
            
            if claim_id_format_invalid and potential_claim_ids and has_missing_claim_slot:
                # User tried to provide claim ID but format is wrong
                reason = "invalid_claim_format"
                reason_detail = f"Invalid claim ID format: {potential_claim_ids}. Please provide a valid claim number."
                logger.info(f"   Detected invalid claim ID format: {potential_claim_ids}")
                logger.info(f"   Missing claim-related slots: {[s for s in missing_slots if s in claim_related_slots]}")
            else:
                reason_detail = f"Missing: {', '.join(missing_slots)}"
        elif confidence < confidence_threshold:
            reason = "low_confidence"
            reason_detail = f"Low confidence: {confidence:.2f} (threshold: {confidence_threshold})"
        else:
            reason = "ambiguous_intent"
            reason_detail = "Multiple possible interpretations"
        
        logger.info(f"   Reason: {reason}")
        logger.info(f"   Detail: {reason_detail}")
        logger.info(f"   Intent: {intent} (confidence: {confidence:.2f}, threshold: {confidence_threshold})")
        if missing_slots:
            logger.info(f"   Missing entities: {missing_slots}")
        
        # Build clarification context for response agent
        clarification_context = {
            "reason": reason,
            "confidence": confidence,
            "intent": intent,
            "user_query": text,
            "missing_entities": missing_slots,
            "provided_entities": list(entities.keys()) if entities else [],
            "intent_candidates": [],  # Could extract from metadata if needed
            # Additional context for invalid claim format
            "claim_id_format_invalid": claim_id_format_invalid,
            "potential_claim_ids": potential_claim_ids,
        }
        
        logger.info("   → Will generate intelligent follow-up question using response agent")

        result = {
            "needs_clarification": True,  # Flag for response agent to generate question
            "clarification_context": clarification_context,  # Context for question generation
            "metadata": {
                **metadata,
                "clarification_triggered": True,
                "clarification_reason": reason,
                "will_generate_followup": True
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result
        
    except Exception as e:
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Clarification failed: {str(e)}",
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
        
        logger.error(f"🚨 Exception in clarification: {e}\n{tb}")
        
        result = {
            "error": error.user_message,
            "needs_clarification": True,
            "response": error.user_message,  # Always use response field
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value,
                "clarification": True
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result
