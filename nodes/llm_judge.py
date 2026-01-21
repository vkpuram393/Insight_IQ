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

def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON from LLM response text.
    
    Handles various edge cases:
    - Markdown code blocks (```json ... ```)
    - Extra text before/after JSON
    - Nested JSON structures
    - Incomplete or malformed JSON
    - Multiple JSON objects (returns first valid one)
    
    Args:
        text: Raw response text from LLM
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted
    """
    if not text or not text.strip():
        raise json.JSONDecodeError("Empty response text", text, 0)
    
    # Strategy 1: Remove markdown code blocks
    cleaned = text.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^```\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    
    # Strategy 2: Try parsing the cleaned text directly
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Find JSON object boundaries by counting braces
    # This handles nested structures properly
    # Prefer objects that contain "intent" field
    brace_count = 0
    start_idx = -1
    candidates = []  # Store all valid JSON objects found
    
    for i, char in enumerate(cleaned):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                # Found a complete JSON object
                json_str = cleaned[start_idx:i+1]
                try:
                    parsed = json.loads(json_str)
                    # Prefer objects with "intent" field
                    if "intent" in parsed:
                        return parsed
                    candidates.append(parsed)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    # If we found any valid JSON objects, return the first one
    if candidates:
        return candidates[0]
    
    # Strategy 4: Try to find JSON using regex with balanced braces
    # Look for pattern: { ... "intent" ... } with proper nesting
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"intent"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(pattern, cleaned, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    
    # Strategy 5: Try to extract JSON by finding the first { and last }
    # This is a last resort and may fail with nested structures
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = cleaned[first_brace:last_brace+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 6: Try to fix common JSON issues
    # Remove trailing commas, fix quotes, etc.
    fixed = cleaned
    # Remove trailing commas before closing braces/brackets
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    # Try parsing fixed version
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # All strategies failed
    raise json.JSONDecodeError(
        f"Could not extract valid JSON from response. Response length: {len(text)} chars",
        text,
        0
    )

def _load_config() -> Dict[str, Any]:
    """Load domain config from JSON file"""
    global _config_cache
    config_path = Path(__file__).parent.parent / "config" / "domain_config.json"
    try:
        with open(config_path, 'r') as f:
            _config_cache = json.load(f)
        return _config_cache
    except Exception as e:
        logger.critical(f"🚨 CRITICAL: Failed to load domain config from {config_path}: {e}")
        logger.critical(f"   This is a fatal configuration error. Using default threshold (0.7)")
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
        
        # Parse JSON response using robust extraction
        try:
            response_text = gemini_response.text
            llm_result = _extract_json_from_text(response_text)
            
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
                
                # Handle claim_number - may be comma-separated for multiple claims
                if "claim_number" in new_entities and new_entities["claim_number"] and "claim_ids" not in new_entities:
                    claim_num = new_entities["claim_number"]
                    if isinstance(claim_num, str) and ("," in claim_num or " and " in claim_num.lower()):
                        # Split comma or "and" separated claims into list
                        claims = re.split(r',|\s+and\s+', claim_num, flags=re.IGNORECASE)
                        new_entities["claim_ids"] = [c.strip() for c in claims if c.strip()]
                    else:
                        new_entities["claim_ids"] = [claim_num]
                
                # Handle sequence_number - may be comma-separated for multiple sequences
                if "sequence_number" in new_entities and new_entities["sequence_number"] and "claim_sequences" not in new_entities:
                    seq_num = new_entities["sequence_number"]
                    if isinstance(seq_num, str) and ("," in seq_num or " and " in seq_num.lower()):
                        # Split comma or "and" separated sequences into list
                        seqs = re.split(r',|\s+and\s+', seq_num, flags=re.IGNORECASE)
                        new_entities["claim_sequences"] = [s.strip() for s in seqs if s.strip()]
                    else:
                        new_entities["claim_sequences"] = [seq_num]
                
                # Also filter out lists containing only None
                if "claim_ids" in new_entities and new_entities["claim_ids"] == [None]:
                    del new_entities["claim_ids"]
                if "claim_sequences" in new_entities and new_entities["claim_sequences"] == [None]:
                    del new_entities["claim_sequences"]
            
            logger.info(f"LLM Response - Intent: {new_intent}, Confidence: {new_confidence:.2f}")
            logger.info(f"   Entities: {new_entities}")
            
        except json.JSONDecodeError as e:
            if hasattr(gemini_response, 'text'):
                full_response = gemini_response.text
                response_length = len(full_response)
                logger.critical(f"🚨 CRITICAL: Failed to parse JSON from LLM response: {e}")
                logger.critical(f"   Response length: {response_length} chars")
                logger.critical(f"   Complete JSON response that failed to parse:")
                logger.critical(f"   {'='*70}")
                # Log complete response - split into chunks if too long to avoid truncation
                if response_length > 10000:
                    # For very long responses, log in chunks
                    chunk_size = 5000
                    for i in range(0, response_length, chunk_size):
                        chunk = full_response[i:i+chunk_size]
                        logger.critical(f"   [Chunk {i//chunk_size + 1}]: {chunk}")
                else:
                    # Log complete response for shorter ones
                    logger.critical(f"   {full_response}")
                logger.critical(f"   {'='*70}")
            else:
                response_preview = str(gemini_response)[:1000]
                logger.critical(f"🚨 CRITICAL: Failed to parse JSON from LLM response: {e}")
                logger.critical(f"   Response preview (first 1000 chars): {response_preview}")
                logger.critical(f"   Full response: {str(gemini_response)}")
            # Fallback to original values
            new_intent = original_intent
            new_confidence = original_confidence
            new_entities = original_entities
        except Exception as e:
            if hasattr(gemini_response, 'text'):
                full_response = gemini_response.text
                response_length = len(full_response)
                logger.critical(f"🚨 CRITICAL: Error parsing LLM response: {e}")
                logger.critical(f"   Exception type: {type(e).__name__}")
                logger.critical(f"   Response length: {response_length} chars")
                logger.critical(f"   Complete JSON response that failed to parse:")
                logger.critical(f"   {'='*70}")
                # Log complete response - split into chunks if too long to avoid truncation
                if response_length > 10000:
                    # For very long responses, log in chunks
                    chunk_size = 5000
                    for i in range(0, response_length, chunk_size):
                        chunk = full_response[i:i+chunk_size]
                        logger.critical(f"   [Chunk {i//chunk_size + 1}]: {chunk}")
                else:
                    # Log complete response for shorter ones
                    logger.critical(f"   {full_response}")
                logger.critical(f"   {'='*70}")
                logger.critical(f"   Traceback: {traceback.format_exc()}")
            else:
                response_preview = str(gemini_response)[:1000]
                logger.critical(f"🚨 CRITICAL: Error parsing LLM response: {e}")
                logger.critical(f"   Exception type: {type(e).__name__}")
                logger.critical(f"   Response preview (first 1000 chars): {response_preview}")
                logger.critical(f"   Full response: {str(gemini_response)}")
                logger.critical(f"   Traceback: {traceback.format_exc()}")
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
        
        logger.critical(f"🚨 CRITICAL: Exception in LLM Judge: {e}\n{tb}")
        
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

