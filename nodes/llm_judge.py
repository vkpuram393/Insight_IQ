"""
LLM Judge Node - Re-classifies intent using LLM when initial classifier has low confidence

This node acts as an "expert reviewer" that takes a second look at uncertain intent classifications.
It uses domain-aware LLM fallback with expert-level prompts for each domain (cap_api, benefits_api,
claim_history_search, member_domain, override_domain, general).
"""

import json
import re
import traceback
import asyncio
from pathlib import Path
from typing import Dict, Any
from state.schema import AgentState
from core.logger import get_logger
from core.errors.models import create_internal_error
from core.logging_context import extract_logging_context, log_state_snapshot
from persistence import PersistenceStoreFactory
from config.config import settings
from prompt_templates.domain_prompts.llm_fallback import llm_fallback_classify_async
from prompt_templates.domain_prompts.base_prompt import INTENT_TO_DOMAIN

logger = get_logger(__name__)

# Load config cache
_config_cache = None

def _load_config() -> Dict[str, Any]:
    """Load domain config from JSON file"""
    global _config_cache
    config_path = Path(__file__).parent.parent / "config" / "domain_config.json"
    try:
        with open(config_path, 'r') as f:
            _config_cache = json.load(f)
        return _config_cache
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Return defaults
        return {
            "confidence_threshold": 0.7
        }

async def llm_judge_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM Judge Node - Re-classifies intent using LLM when initial classifier has low confidence
    
    This node:
    1. Takes the original intent classification (low confidence)
    2. Uses LLM to re-classify with more context
    3. Updates confidence and potentially intent/entities
    4. Sets intent_reclassified = True (prevents infinite loops)
    
    INPUT (from state):
        - intent: Original intent from classifier
        - confidence: Original confidence (low)
        - entities: Original entities
        - text: User's query
        - intent_reclassified: Should be False (initial classification)
    
    OUTPUT (to state):
        - intent: Re-classified intent (or kept original)
        - confidence: Updated confidence from LLM
        - entities: Updated entities (or kept original)
        - intent_reclassified: True (KEY - prevents infinite loop)
    """
    node_name = "llm_judge"
    log_ctx = extract_logging_context(state)
    
    try:
        logger.info("\n" + "="*70)
        logger.info("LLM JUDGE NODE - Re-classifying Intent")
        logger.info("="*70)
        
        # Load config
        config = _load_config()
        threshold = config.get("confidence_threshold", 0.7)
        
        # Get current state values
        original_intent = state.get("intent", "unknown")
        original_confidence = state.get("confidence", 0.0)
        original_entities = state.get("entities") or {}
        text = state.get("text", "")
        conversation_history = state.get("conversation_history", [])
        intent_reclassified = state.get("intent_reclassified", False)
        
        logger.info(f"Input - Intent: {original_intent}, Confidence: {original_confidence:.2f}")
        logger.info(f"Entities: {original_entities}")
        logger.info(f"Text: {text[:100]}...")
        logger.info(f"Conversation history: {len(conversation_history)} messages")
        logger.info(f"intent_reclassified flag: {intent_reclassified}")
        
        # Use domain-aware LLM fallback for re-classification
        logger.info("Using domain-aware LLM fallback for intent re-classification")
        
        # Build top-5 candidates from state metadata or use original intent as primary candidate
        # The embedding classifier may have stored top candidates in metadata
        metadata = state.get("metadata", {})
        top5_from_state = metadata.get("top5_intents", None)
        
        if top5_from_state:
            # Use stored top-5 from embedding classifier
            top5_intents = [(name, prob) for name, prob in top5_from_state]
        else:
            # Construct approximate top-5 from original intent and its domain siblings
            original_domain = INTENT_TO_DOMAIN.get(original_intent, "general")
            domain_intents = [k for k, v in INTENT_TO_DOMAIN.items() if v == original_domain and k != original_intent]
            top5_intents = [(original_intent, original_confidence)]
            # Add domain siblings with decreasing confidence
            for i, sibling in enumerate(domain_intents[:4]):
                top5_intents.append((sibling, max(0.05, original_confidence * 0.3 - i * 0.05)))
        
        logger.info(f"Top-5 candidates: {[(n, f'{p:.2f}') for n, p in top5_intents]}")
        
        # Call the domain-aware LLM fallback (async version)
        try:
            llm_result = await llm_fallback_classify_async(
                query=text,
                top5_intents=top5_intents,
                ensemble_intent=original_intent,
                ensemble_confidence=original_confidence,
                conversation_history=conversation_history,
                model=getattr(settings, 'llm_model', 'gemini-2.0-flash'),
                temperature=getattr(settings, 'llm_temperature', 0.0),
            )
            
            new_intent = llm_result.get("intent", original_intent)
            new_confidence = float(llm_result.get("confidence", original_confidence))
            new_entities = llm_result.get("entities", {}) or original_entities
            
            # Normalize LLM entity keys to match Entity Extractor format
            # LLM uses singular keys; Confidence Checker expects plural list keys
            if new_entities:
                # Filter out None values
                new_entities = {k: v for k, v in new_entities.items() if v is not None}
                
                # Handle claim_number → claim_ids
                if "claim_number" in new_entities and new_entities["claim_number"] and "claim_ids" not in new_entities:
                    claim_num = new_entities["claim_number"]
                    if isinstance(claim_num, str) and ("," in claim_num or " and " in claim_num.lower()):
                        claims = re.split(r',|\s+and\s+', claim_num, flags=re.IGNORECASE)
                        new_entities["claim_ids"] = [c.strip() for c in claims if c.strip()]
                    else:
                        new_entities["claim_ids"] = [claim_num]
                
                # Handle sequence_number → claim_sequences
                if "sequence_number" in new_entities and new_entities["sequence_number"] and "claim_sequences" not in new_entities:
                    seq_num = new_entities["sequence_number"]
                    if isinstance(seq_num, str) and ("," in seq_num or " and " in seq_num.lower()):
                        seqs = re.split(r',|\s+and\s+', seq_num, flags=re.IGNORECASE)
                        new_entities["claim_sequences"] = [s.strip() for s in seqs if s.strip()]
                    else:
                        new_entities["claim_sequences"] = [seq_num]
                
                # Filter out lists containing only None
                if "claim_ids" in new_entities and new_entities["claim_ids"] == [None]:
                    del new_entities["claim_ids"]
                if "claim_sequences" in new_entities and new_entities["claim_sequences"] == [None]:
                    del new_entities["claim_sequences"]
            
            logger.info(f"LLM Fallback Response - Intent: {new_intent}, Confidence: {new_confidence:.2f}")
            logger.info(f"   Domain: {llm_result.get('domain', 'unknown')}")
            logger.info(f"   Entities: {new_entities}")
            logger.info(f"   Reasoning: {llm_result.get('reasoning', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Domain-aware LLM fallback failed: {e}")
            # Fallback to original values
            new_intent = original_intent
            new_confidence = original_confidence
            new_entities = original_entities
        
        # IMPORTANT: Clear needs_clarification flag if confidence is now high
        # This allows confidence_checker to re-evaluate and proceed normally
        needs_clarification = new_confidence < threshold
        
        logger.info(f"   Output - Intent: {new_intent}, Confidence: {new_confidence:.2f}")
        logger.info(f"   Entities: {new_entities}")
        logger.info(f"   needs_clarification: {needs_clarification} (based on new confidence)")
        logger.info(f"   Setting: intent_reclassified = True (prevents infinite loop)")
        
        # Construct result
        # Preserve original embedding classifier confidence in metadata for batch testing
        result = {
            "intent": new_intent,
            "confidence": new_confidence,  # LLM judge confidence
            "entities": new_entities,
            "intent_reclassified": True,  # KEY: Set flag to True to prevent infinite loops
            "needs_clarification": needs_clarification,  # Clear if high confidence
            "metadata": {
                **state.get("metadata", {}),
                "embedding_classifier_confidence": original_confidence,  # Preserve original for batch testing
                "llm_judge_confidence": new_confidence,  # Store LLM judge confidence
                "llm_fallback_domain": INTENT_TO_DOMAIN.get(new_intent, "unknown"),
            }
        }
        
        # Log state snapshot
        await log_state_snapshot(state, node_name, result)
        
        logger.info(" LLM Judge completed - returning to confidence_checker for re-evaluation")
        return result
        
    except Exception as e:
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"LLM Judge failed: {str(e)}",
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
        
        logger.error(f" Exception in LLM Judge: {e}\n{tb}")
        
        # Return error state - but still set flag to prevent loops
        result = {
            "intent_reclassified": True,  # Still set flag even on error
            "error": error.user_message,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value,
                "llm_judge_failed": True
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result

