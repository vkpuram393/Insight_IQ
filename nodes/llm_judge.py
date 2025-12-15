"""
LLM Judge Node - Re-classifies intent using LLM when initial classifier has low confidence

This node acts as an "expert reviewer" that takes a second look at uncertain intent classifications.
It uses an LLM to re-classify the intent with higher accuracy.
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
from services.llm_connection import GenerateRequest, _generate_core
from prompt_templates.prompt_template import claim_prompt_template

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
        
        # Use LLM to re-classify intent
        logger.info("Using Gemini LLM for intent re-classification")
        
        # Build user prompt with conversation history if available
        user_prompt = ""
        
        # Include conversation history if available
        if conversation_history and len(conversation_history) > 0:
            user_prompt += "Conversation History:\n"
            for msg in conversation_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    user_prompt += f"{role.capitalize()}: {content}\n"
            user_prompt += "\n"
        
        # Add current user message
        user_prompt += f"Current User message: {text}\n\n"
        
        # Call Gemini with prompt template as system instruction
        req = GenerateRequest(
            prompt=user_prompt,
            system_instruction=claim_prompt_template,
            temperature=settings.llm_temperature,
            model=settings.llm_model
        )
        
        # Run in executor to avoid blocking (Gemini client is sync)
        loop = asyncio.get_event_loop()
        gemini_response = await loop.run_in_executor(None, _generate_core, req)
        
        # Parse JSON response
        try:
            response_text = gemini_response.text.strip()
            
            # Remove markdown code blocks if present
            response_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
            response_text = re.sub(r'^```\s*', '', response_text, flags=re.MULTILINE)
            response_text = re.sub(r'```\s*$', '', response_text, flags=re.MULTILINE)
            response_text = response_text.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*"intent"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                llm_result = json.loads(json_match.group(0))
            else:
                llm_result = json.loads(response_text)
            
            # Extract new intent, confidence, and entities from LLM response
            new_intent = llm_result.get("intent", original_intent)
            new_confidence = float(llm_result.get("confidence", original_confidence))
            new_entities = llm_result.get("entities") or original_entities
            
            # FIX: Normalize LLM entity keys to match Entity Extractor format
            # LLM uses singular keys; Confidence Checker expects plural list keys
            # ALSO: Filter out None values to prevent "claim None" in responses
            if new_entities:
                # Filter out None values first
                new_entities = {k: v for k, v in new_entities.items() if v is not None}
                
                # Only create list keys if the value exists and is not None
                if "claim_number" in new_entities and new_entities["claim_number"] and "claim_ids" not in new_entities:
                    new_entities["claim_ids"] = [new_entities["claim_number"]]
                if "sequence_number" in new_entities and new_entities["sequence_number"] and "claim_sequences" not in new_entities:
                    new_entities["claim_sequences"] = [new_entities["sequence_number"]]
                
                # Also filter out lists containing only None
                if "claim_ids" in new_entities and new_entities["claim_ids"] == [None]:
                    del new_entities["claim_ids"]
                if "claim_sequences" in new_entities and new_entities["claim_sequences"] == [None]:
                    del new_entities["claim_sequences"]
            
            logger.info(f"LLM Response - Intent: {new_intent}, Confidence: {new_confidence:.2f}")
            logger.info(f"   Entities: {new_entities}")
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            logger.warning(f"   Response was: {gemini_response.text[:500]}")
            # Fallback to original values
            new_intent = original_intent
            new_confidence = original_confidence
            new_entities = original_entities
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
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
                "llm_judge_confidence": new_confidence  # Store LLM judge confidence
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

