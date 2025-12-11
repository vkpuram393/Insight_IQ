"""
Dynamic Claims API Orchestrator (final version).
Single entry node: call_claims_tool_node(state)
Uses repository-based API registry and returns ToolResult (with AgentError)
"""

import time
import uuid
import requests
import traceback
from typing import Dict, Any, Optional

from tools.api_repository import get_api_repository  # your repository builder
from core.errors.error_handler import to_agent_error
from utils.retry import retry
from core.errors.exceptions import ExternalAPIError, ToolTimeoutError
from tools.api_fallbacks import get_fallback_details, get_fallback_list  # dynamic fallback data

from core.node_models import ToolResult, ToolExecutionStatus
from core.errors.models import (
    AgentError,
    create_validation_error,
    create_api_error,
    create_internal_error
)

from core.node_models import IntentResult
from state.schema import AgentState
from core.logger import get_logger
from core.logging_context import extract_logging_context, log_state_snapshot
from config.config import settings
from persistence import PersistenceStoreFactory
# Note: PII masking is handled by safety layer, not in tools layer

# ============================================================================
# LOGGER
# ============================================================================
logger = get_logger(__name__)

# ============================================================================
# MAIN CLAIMS API TOOL NODE
# ============================================================================
# Main Claims API Tool Node
async def call_claims_tool_node(state) -> Dict[str, Any]:
    """
    Primary orchestrator node called by agent.
    Returns ToolResult as dict.
    
    Accepts either:
    INPUT (from state):
        - intent: What data to fetch
        - entities: Parameters (e.g., claim_number)

    OUTPUT (to state):
        - tool_results: API response data
    """
    logger.info("🔧 Node: Call Claims Tool (dynamic router)")
    
    # Extract logging context if called from LangGraph
    log_ctx = extract_logging_context(state) if isinstance(state, dict) else {}
    persistence_store = None
    
    try:
        # Log node entry for audit trail
        if log_ctx and settings.enable_telemetry:
            persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
            await persistence_store.log_audit(
                session_id=log_ctx.get("session_id"),
                request_id=log_ctx.get("request_id"),
                user_id=log_ctx.get("user_id"),
                node_name="call_claims_tool",
                event_type="node_entry",
                data={"node": "call_claims_tool"}
            )
        
        # Handle both AgentState (dict) and IntentResult (Pydantic model)
        if isinstance(state, dict):
            # Called from LangGraph with AgentState
            intent = state.get("intent")
            
            # CRITICAL: Merge extracted_slots (from conversation history) with current entities
            # This enables follow-up questions without re-asking for claim_id
            extracted_slots = state.get("extracted_slots", {})
            current_entities = state.get("entities", {})
            # Current entities take precedence over extracted ones
            entities = {**(extracted_slots or {}), **(current_entities or {})}
            
            if extracted_slots:
                logger.info(f"🔗 Context-aware: merged {len(extracted_slots)} slots from history + {len(current_entities)} current entities")
                logger.debug(f"   From history: {extracted_slots}")
                logger.debug(f"   From current: {current_entities}")
                logger.debug(f"   Final merged: {entities}")
            else:
                logger.debug(f"Input type: AgentState (dict), entities: {entities}")
        else:
            # Called directly with IntentResult model (for testing)
            intent = state.intent
            entities = state.entities if state.entities else {}
            logger.debug(f"Input type: IntentResult (Pydantic model)")
    
        # Validate intent exists
        if not intent:
            logger.warning("⚠️ Validation failed: No intent provided")
            ae = create_validation_error(
                message="No intent provided in state",
                field="intent",
                value=None
            )
            tool_result = ToolResult(
                tool_name="claims_api",
                status=ToolExecutionStatus.FAILURE,
                data={},
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                api_endpoint=None,
                http_status_code=None,
                is_retryable=False
            )
            return {"tool_results": tool_result.dict()}
            
        # Validate entities exist
        if not entities:
            logger.warning("⚠️ Validation failed: No entities provided")
            ae = create_validation_error(
                message="No entities provided in state",
                field="entities",
                value=None
            )
            tool_result = ToolResult(
                tool_name="claims_api",
                status=ToolExecutionStatus.FAILURE,
                data={},
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                api_endpoint=None,
                http_status_code=None,
                is_retryable=False
            )
            return {"tool_results": tool_result.dict()}
        
        entities = normalize_entities(entities) 

        logger.info(f"📋 Intent: {intent}")
        logger.info(f"📋 Entities: {entities}")

        # ========================================================================
        # SPECIAL BRANCH: Enriched claim_details flow (2-step API calls)
        # Hybrid Routing: Entities + Intent
        # - If (claimNumber OR claimId) + claimSequence → ALWAYS use enriched details flow
        # - If only claimId/claimNumber → use intent to decide between list vs details
        # ========================================================================
        if ("claimNumber" in entities or "claimId" in entities) and "claimSequence" in entities:
            # Get the claim number (might be under claimNumber or claimId key)
            claim_num = entities.get("claimNumber") or entities.get("claimId")
            claim_seq = entities.get("claimSequence")
            
            logger.info("🔄 HYBRID ROUTING: Both claimNumber/claimId + claimSequence present → forcing enriched details flow")
            logger.info(f"   📍 ClaimNumber: {claim_num} (type: {type(claim_num).__name__}, repr: {repr(claim_num)})")
            logger.info(f"   📍 ClaimSequence: {claim_seq} (type: {type(claim_seq).__name__}, repr: {repr(claim_seq)})")
            logger.info(f"   📍 Intent: {intent} (not used - entities are decisive)")
            
            # Both entities present, proceed with enriched flow
            # (No validation needed - already confirmed above)
            
            # Log enriched flow attempt
            if log_ctx and persistence_store:
                await persistence_store.log_audit(
                    session_id=log_ctx.get("session_id"),
                    request_id=log_ctx.get("request_id"),
                    user_id=log_ctx.get("user_id"),
                    node_name="call_claims_tool",
                    event_type="enriched_flow_attempt",
                    data={
                        "intent": intent,
                        "flow_type": "claim_details_enriched",
                        "entities": entities
                    }
                )
            
            # Execute enriched claim details flow
            start = time.time()
            try:
                logger.info(f"🌐 Calling enriched claim details flow")
                result = combine_claim_details_and_list(
                    claimNumber=claim_num,
                    claimSequence=claim_seq
                )
                elapsed_ms = (time.time() - start) * 1000.0
                
                logger.info(f"✅ Enriched claim details flow succeeded in {elapsed_ms:.2f}ms")
                
                # Log success
                if log_ctx and persistence_store:
                    await persistence_store.log_audit(
                        session_id=log_ctx.get("session_id"),
                        request_id=log_ctx.get("request_id"),
                        user_id=log_ctx.get("user_id"),
                        node_name="call_claims_tool",
                        event_type="enriched_flow_success",
                        data={
                            "flow_type": "claim_details_enriched",
                            "execution_time_ms": elapsed_ms,
                            "status": "success"
                        }
                    )
                
                # Return raw unmasked data - PII masking will be handled by safety layer
                # This follows separation of concerns: tools return data, safety layer handles security
                logger.info(f"✅ Returning raw enriched claim details (PII masking will be done by safety layer)")
                
                tool_result = ToolResult(
                    tool_name="claim_details_enriched",
                    status=ToolExecutionStatus.SUCCESS,
                    data=result,  # Raw unmasked data
                    error_message=None,
                    error_code=None,
                    agent_error=None,
                    execution_time_ms=elapsed_ms,
                    api_endpoint="/myclaims/claims/v1/details",
                    http_status_code=200,
                    is_retryable=False,
                    metadata={}
                )
                return {"tool_results": tool_result.dict()}
                
            except Exception as exc:
                elapsed_ms = (time.time() - start) * 1000.0
                logger.error(f"❌ Enriched claim details flow failed after {elapsed_ms:.2f}ms: {str(exc)}")
                
                # Map exception to AgentError
                ae = to_agent_error(exc, node="claim_details_enriched")
                
                # Log failure
                if log_ctx and persistence_store:
                    await persistence_store.log_audit(
                        session_id=log_ctx.get("session_id"),
                        request_id=log_ctx.get("request_id"),
                        user_id=log_ctx.get("user_id"),
                        node_name="call_claims_tool",
                        event_type="enriched_flow_failure",
                        data={
                            "flow_type": "claim_details_enriched",
                            "execution_time_ms": elapsed_ms,
                            "error": str(exc),
                            "status": "failure"
                        }
                    )
                
            # Extract full API error response details from exception if available
            api_error_details = {}
            if hasattr(exc, 'details') and isinstance(exc.details, dict):
                api_error_details = {
                    "error_type": "api_error",
                    "status_code": exc.details.get("status"),
                    "response_body": exc.details.get("response_body"),
                    "response_json": exc.details.get("response_json"),
                    "api_url": exc.details.get("url"),
                    "api_name": exc.details.get("api")
                }
            
            tool_result = ToolResult(
                tool_name="claim_details_enriched",
                status=ToolExecutionStatus.FAILURE,
                data=api_error_details,  # Put API error details in data field
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                api_endpoint="/myclaims/claims/v1/details",
                http_status_code=ae.metadata.get("status_code") if isinstance(ae.metadata, dict) else None,
                is_retryable=ae.is_retryable
            )
            return {"tool_results": tool_result.dict()}
        
        # ========================================================================
        # STANDARD FLOW: All other intents use normal matching logic
        # ========================================================================
        # 1) match API (Option A: treat no match as validation error)
        logger.debug(f"🔍 Matching API for intent: {intent}")
        api = match_api(intent, entities)

        if not api:
            logger.warning(f"❌ No matching API found for intent '{intent}' with entities: {list(entities.keys())}")
            ae = create_validation_error(
                message="No matching API found for given intent/entities",
                field="entities",
                value=entities
            )

            tool_result = ToolResult(
                tool_name="claims_api",
                status=ToolExecutionStatus.FAILURE,
                data={},
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                api_endpoint=None,
                http_status_code=None,
                is_retryable=False
            )
            # return tool_result.dict()
            return {"tool_results": tool_result.dict()}

        logger.info(f"✅ [MATCH] Selected API → {getattr(api, 'name', '<unknown>')}")

        # 2) build request body
        logger.debug(f"🔨 Building request body for API: {api.name}")
        try:
            body = api.body_template(entities)
            logger.debug(f"✅ Request body built successfully")
        except Exception as e:
            logger.error(f"❌ Failed to build request body: {e}")
            ae = create_internal_error(
                error_message=f"Failed building request body: {e}",
                node_name=getattr(api, "name", None)
            )
            tool_result = ToolResult(
                tool_name=getattr(api, "name", "claims_api"),
                status=ToolExecutionStatus.FAILURE,
                data={},
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                api_endpoint=getattr(api, "full_url", None),
                is_retryable=False
            )
            return {"tool_results": tool_result.dict()}

        # Log API call attempt for audit trail
        if log_ctx and persistence_store:
            await persistence_store.log_audit(
                session_id=log_ctx.get("session_id"),
                request_id=log_ctx.get("request_id"),
                user_id=log_ctx.get("user_id"),
                node_name="call_claims_tool",
                event_type="api_call_attempt",
                data={
                    "intent": intent,
                    "api_name": getattr(api, "name", "unknown"),
                    "api_endpoint": getattr(api, "full_url", None)
                }
            )

        # 3) call external API (sync call wrapped with retry decorator)
        logger.info(f"🌐 Calling external API: {api.name} → {getattr(api, 'full_url', 'N/A')}")
        start = time.time()
        try:
            result = call_external_api(api, body)
            elapsed_ms = (time.time() - start) * 1000.0
            
            logger.info(f"✅ API call succeeded in {elapsed_ms:.2f}ms")

            # Log successful API call
            if log_ctx and persistence_store:
                await persistence_store.log_audit(
                    session_id=log_ctx.get("session_id"),
                    request_id=log_ctx.get("request_id"),
                    user_id=log_ctx.get("user_id"),
                    node_name="call_claims_tool",
                    event_type="api_call_success",
                    data={
                        "api_name": getattr(api, "name", "unknown"),
                        "execution_time_ms": elapsed_ms,
                        "status": "success"
                    }
                )

            # Return raw unmasked data - PII masking will be handled by safety layer
            # This follows separation of concerns: tools return data, safety layer handles security
            logger.info(f"✅ Returning raw API response (PII masking will be done by safety layer)")

            tool_result = ToolResult(
                tool_name=getattr(api, "name", "claims_api"),
                status=ToolExecutionStatus.SUCCESS,
                data=result,  # Raw unmasked data
                error_message=None,
                error_code=None,
                agent_error=None,
                execution_time_ms=elapsed_ms,
                api_endpoint=getattr(api, "full_url", None),
                http_status_code=200,
                is_retryable=False,
                metadata={}
            )
            result_dict = {"tool_results": tool_result.dict()}
            if isinstance(state, dict):
                await log_state_snapshot(state, "call_claims_tool", result_dict)
            return result_dict

        except Exception as exc:
            elapsed_ms = (time.time() - start) * 1000.0
            logger.error(f"❌ API call failed after {elapsed_ms:.2f}ms: {str(exc)}")
            
            # map exception -> AgentError and fill ToolResult
            ae = to_agent_error(exc, node=getattr(api, "name", None))

            # Log failed API call
            if log_ctx and persistence_store:
                await persistence_store.log_audit(
                    session_id=log_ctx.get("session_id"),
                    request_id=log_ctx.get("request_id"),
                    user_id=log_ctx.get("user_id"),
                    node_name="call_claims_tool",
                    event_type="api_call_failure",
                    data={
                        "api_name": getattr(api, "name", "unknown"),
                        "execution_time_ms": elapsed_ms,
                        "error": str(exc),
                        "status": "failure"
                    }
                )

            # Extract full API error response details from exception if available
            api_error_details = {}
            if hasattr(exc, 'details') and isinstance(exc.details, dict):
                api_error_details = {
                    "error_type": "api_error",
                    "status_code": exc.details.get("status"),
                    "response_body": exc.details.get("response_body"),
                    "response_json": exc.details.get("response_json"),
                    "api_url": exc.details.get("url"),
                    "api_name": exc.details.get("api")
                }
            
            tool_result = ToolResult(
                tool_name=getattr(api, "name", "claims_api"),
                status=ToolExecutionStatus.FAILURE,
                data=api_error_details,  # Put API error details in data field
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                api_endpoint=getattr(api, "full_url", None),
                http_status_code=ae.metadata.get("status_code") if isinstance(ae.metadata, dict) else None,
                is_retryable=ae.is_retryable
            )
            result_dict = {"tool_results": tool_result.dict()}
            if isinstance(state, dict):
                await log_state_snapshot(state, "call_claims_tool", result_dict)
            return result_dict
    except Exception as e:
        # Outer exception handler for unexpected errors
        tb = traceback.format_exc()
        logger.error(f"🚨 Unexpected exception in call_claims_tool_node: {e}\n{tb}")
        
        # Log exception to database
        if log_ctx and persistence_store:
            try:
                await persistence_store.log_exception(
                    session_id=log_ctx.get("session_id"),
                    request_id=log_ctx.get("request_id"),
                    user_id=log_ctx.get("user_id"),
                    node_name="call_claims_tool",
                    error_code="E9001",
                    category="system",
                    severity="critical",
                    message=f"Unexpected error in claims API tool: {str(e)}",
                    stacktrace=tb
                )
            except Exception as log_error:
                logger.error(f"Failed to log exception: {log_error}")
        
        # Return error result
        ae = create_internal_error(
            error_message=str(e),
            stacktrace=tb,
            node_name="call_claims_tool"
        )
        tool_result = ToolResult(
            tool_name="claims_api",
            status=ToolExecutionStatus.FAILURE,
            data={},
            error_message=ae.message,
            error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
            agent_error=ae,
            api_endpoint=None,
            http_status_code=None,
            is_retryable=False
        )
        return {"tool_results": tool_result.dict()}



# ============================================================================
# ENTITY NORMALIZATION
# ============================================================================
ENTITY_MAP ={
    "claim_number":"claimNumber",
    "member_id":"memberId",
    "prescription_number":"prescriptionNumber",
    "medication_name":"medicationName",
    "date_from":"dateFrom",
    "date_to":"dateTo",
    "claim_sequence":"claimSequence",
    "sequence_number":"claimSequence",  # LLM judge uses this name
    "claim_id":"claimId",
    "claim_ids":"claimId",  # Map plural to singular for API
    "claim_sequences":"claimSequence",  # Map plural sequences
}

def _is_masked_token(value: Any) -> bool:
    """
    Check if a value is a masked PII token like [CLAIM_ID_ABC123] or [CLAIM_ID_253152631273000].
    
    These tokens are created by the PII protection layer and should not be sent to the API.
    In multi-turn conversations, the LLM may extract masked tokens from conversation history
    instead of the actual values.
    """
    import re
    if isinstance(value, str):
        # Match patterns like:
        # - [CLAIM_ID_ABC123] (hex hash)
        # - [CLAIM_ID_253152631273000] (actual claim number embedded)
        # - [PERSON_DEF456]
        # - Any token starting with [ and containing ENTITY_TYPE_ pattern
        return bool(re.match(r'^\[[A-Z_]+_[A-Za-z0-9]+\]$', value))
    return False

# --- Helper Function to handle Pydantic model extraction ---
def normalize_entities(entities_obj) -> Dict[str, Any]:
    """
    1. Merges top-level entities with raw_entities from the Pydantic model or dict.
    2. Normalizes all keys from snake_case to target API's camelCase format.
    3. Also accepts already-normalized camelCase keys (passthrough).
    4. Filters out masked PII tokens (e.g., [CLAIM_ID_ABC123]) that should not be sent to API.
    """
    
    # Handle both Pydantic models and plain dicts
    if isinstance(entities_obj, dict):
        # Already a dict from state
        all_entities = {k: v for k, v in entities_obj.items() if v is not None and k != 'raw_entities'}
        # Merge raw_entities if present
        if 'raw_entities' in entities_obj and isinstance(entities_obj['raw_entities'], dict):
            all_entities.update(entities_obj['raw_entities'])
    else:
        # Pydantic model - use .model_dump(exclude_none=True) for Pydantic v2 (or .dict() for v1)
        # Extract the main fields from the Pydantic model, excluding raw_entities itself initially
        all_entities = entities_obj.model_dump(exclude_none=True, exclude={'raw_entities'})
        
        # Merge the contents of raw_entities dictionary into the main dictionary
        all_entities.update(entities_obj.raw_entities)
    
    # CRITICAL: Filter out masked PII tokens that LLM may have extracted from conversation history
    # These look like [CLAIM_ID_ABC123] and should not be sent to the API
    filtered_entities = {}
    for k, v in all_entities.items():
        if _is_masked_token(v):
            logger.warning(f"⚠️ Filtering out masked token from entities: {k}={v}")
            continue
        filtered_entities[k] = v
    all_entities = filtered_entities
    
    # Create reverse mapping for camelCase -> camelCase (already normalized)
    # This allows function to accept both snake_case AND camelCase inputs
    REVERSE_ENTITY_MAP = {v: v for v in ENTITY_MAP.values()}
    COMBINED_MAP = {**ENTITY_MAP, **REVERSE_ENTITY_MAP}
    
    # Normalize keys to the target API format (camelCase)
    # Only include fields that are in ENTITY_MAP (filter out extra fields like person_names)
    normalized_entities = {}
    for k, v in all_entities.items():
        # Skip fields not in COMBINED_MAP (handles both snake_case and camelCase)
        if k not in COMBINED_MAP:
            continue
            
        # Use the mapped name
        target_key = COMBINED_MAP.get(k)
        
        # Handle list values - take first element for singular API parameters
        if isinstance(v, list) and len(v) > 0:
            # For plural entity names mapping to singular API params, use first value
            if k in ['claim_ids', 'member_ids', 'claim_sequences'] and target_key in ['claimId', 'memberId', 'claimSequence']:
                v = v[0]  # Take first claim/member ID or sequence
        
        normalized_entities[target_key] = v
        
    return normalized_entities

# ============================================================================
# API MATCHING LOGIC (Hybrid: Entities + Intent)
# ============================================================================
def match_api(intent: str, entities: Dict[str, Any]):
    """
    Hybrid API matcher using ENTITIES (primary) + INTENT (tiebreaker):
    
    Step 1: Filter APIs by required entities (must all be present)
    Step 2: Score remaining APIs by intent keyword matches
    Step 3: Add specificity bonus (more required entities = higher score)
    
    This ensures:
    - Entities determine WHICH APIs are eligible
    - Intent determines WHICH eligible API to use (if multiple match)
    
    Example:
      Query: "claim status 242563401456001"
      - Entities: claimId="242563401456001"
      - Intent: "claim_status" (keywords: status, check, track...)
      - Both APIs require claimId, but list API has "status" keyword
      - Result: list API wins due to intent match
    """
    registry = get_api_repository()
    intent_lower = (intent or "").lower()

    matched_api = None
    matched_api_score = -1
    eligible_apis = []

    logger.debug(f"🔍 API Matching started - Intent: '{intent}', Entities: {list(entities.keys())}")
    
    for api in registry:
        api_name = getattr(api, 'name', 'unknown')
        required_entities = getattr(api, "required_entities", [])
        
        # STEP 1: Entity filter - required entities must all be present
        missing_entities = [req for req in required_entities if req not in entities]
        if missing_entities:
            logger.debug(f"   ❌ {api_name}: Missing required entities {missing_entities}")
            continue
        
        logger.debug(f"   ✅ {api_name}: Has all required entities {required_entities}")
        eligible_apis.append(api)

        # STEP 2: Intent scoring - count keyword matches
        intent_keywords = getattr(api, "intent_keywords", [])
        matched_keywords = [kw for kw in intent_keywords if kw in intent_lower]
        intent_score = len(matched_keywords)
        
        # STEP 3: Specificity bonus - more required entities = more specific
        specificity_score = len(required_entities)
        
        total_score = intent_score + specificity_score
        
        logger.debug(f"      📊 Scoring: intent_score={intent_score} (matched: {matched_keywords}), specificity={specificity_score}, total={total_score}")

        if total_score > matched_api_score:
            matched_api_score = total_score
            matched_api = api
            logger.debug(f"      🏆 New leader: {api_name} (score: {total_score})")
    
    # Log final decision
    if matched_api:
        logger.info(f"✅ [MATCH] Selected API: {getattr(matched_api, 'name', 'None')} (score: {matched_api_score}, eligible: {len(eligible_apis)})")
    else:
        logger.warning(f"❌ [NO MATCH] No eligible APIs found for intent '{intent}' with entities {list(entities.keys())}")

    return matched_api


# ============================================================================
# EXTERNAL API CALL WITH RETRY
# ============================================================================
@retry(attempts=3, retry_on=(ExternalAPIError, ToolTimeoutError))
def call_external_api(api, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous HTTP call wrapper. Raises ExternalAPIError / ToolTimeoutError on issues.
    Returned value is parsed JSON.
    """
    url = getattr(api, "full_url", getattr(api, "endpoint", None))
    method = getattr(api, "method", "POST").upper()
    
    # Generate proper correlation ID in CVS format
    correlation_id = f"CVS-{uuid.uuid4()}"
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "X-Trace-Id": f"claim-search-{uuid.uuid4()}",
        "X-Loading-Mode": "detail",
        "x-correlation-id": correlation_id,
        "x-consumerAppName": "myclaims",
        "x-clientRefId": correlation_id
    }

    # Log request details for debugging
    logger.info(f"🌐 Making API request:")
    logger.info(f"   URL: {url}")
    logger.info(f"   Method: {method}")
    logger.info(f"   Headers: {headers}")
    logger.info(f"   Body: {body}")
    
    try:
        resp = requests.request(method, url, headers=headers, json=body, timeout=30, verify=True)
        logger.info(f"   Response Status: {resp.status_code}")
        if resp.status_code != 200:
            logger.error(f"   Response Body: {resp.text}")
        resp.raise_for_status()
    except requests.exceptions.Timeout as e:
        # timeout -> retriable
        raise ToolTimeoutError(str(e), details={"api": getattr(api, "name", url)}, retriable=True)
    except requests.RequestException as e:
        # non-timeout HTTP error or connection error
        status = getattr(getattr(e, "response", None), "status_code", None)
        
        # Capture full error response body for detailed error reporting
        error_response_body = None
        error_response_json = None
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"❌ API Error Response:")
            logger.error(f"   Status: {e.response.status_code}")
            logger.error(f"   Response Text: {e.response.text[:500]}")
            
            # Store the full response body
            error_response_body = e.response.text
            
            # Try to parse as JSON for structured error details
            try:
                error_response_json = e.response.json()
            except:
                pass  # Not JSON, use raw text
        
        # Build comprehensive error details
        error_details = {
            "status": status,
            "api": getattr(api, "name", url),
            "response_body": error_response_body,
            "response_json": error_response_json,
            "url": url
        }
        
        retriable = status is None or (500 <= status < 600)
        raise ExternalAPIError(str(e), details=error_details, retriable=retriable)

    # parse JSON or raise
    try:
        return resp.json()
    except ValueError:
        # non-json response
        raise ExternalAPIError("Non-JSON response", details={"status": resp.status_code, "raw": resp.text, "api": getattr(api, "name", url)}, retriable=False)


# ============================================================================
# HELPER METHODS FOR ENRICHED CLAIM DETAILS
# ============================================================================
def get_first_non_none(dicts_and_keys):
    """
    Safely get first non-None value from list of (dict, key) tuples.
    This handles zero values correctly (unlike 'or' chain which treats 0 as falsy).
    
    Args:
        dicts_and_keys: List of (dictionary, key) tuples to check
        
    Returns:
        First non-None value found, or None if all are None/missing
    """
    for d, key in dicts_and_keys:
        if d and key in d and d[key] is not None:
            return d[key]
    return None


def get_claim_details(claimNumber: str, claimSequence: str) -> Dict[str, Any]:
    """
    Fetch claim details by claim number and sequence.
    Uses existing call_external_api() with retry logic.
    Falls back to JSON file on failure.
    
    Args:
        claimNumber: The claim number
        claimSequence: The claim sequence
        
    Returns:
        Single claim details object as returned by API or fallback
    """
    logger.debug(f"📞 Fetching claim details for claimNumber={claimNumber}, claimSequence={claimSequence}")
    
    try:
        # Get the API definition from repository
        registry = get_api_repository()
        api = next((a for a in registry if getattr(a, 'name', '') == 'get_claim_details'), None)
        
        if not api:
            raise ValueError("get_claim_details API not found in repository")
        
        # Build request body using the API's body template
        entities = {
            "claimNumber": claimNumber,
            "claimSequence": claimSequence
        }
        body = api.body_template(entities)
        
        # Call external API with retry logic
        result = call_external_api(api, body)
        logger.debug(f"✅ Successfully fetched claim details")
        return result
    
    except ExternalAPIError as e:
        # Check if error is retriable (server down: 5xx errors, network issues)
        if e.retriable and settings.enable_api_fallback:
            # Server is down AND fallback is enabled - use mock data for testing
            logger.warning(f"⚠️ Fallback: Claim Details API server error (retriable) - Using mock data for testing")
            logger.warning(f"   Error: {str(e)}")
            logger.warning(f"   💡 Fallback is ENABLED (enable_api_fallback=True) - Set to False in production")
            fallback_data = get_fallback_details(claimNumber, claimSequence)
            return fallback_data
        elif e.retriable:
            # Server is down but fallback is disabled - return error
            logger.error(f"❌ Claim Details API server error - Fallback disabled, returning error")
            logger.error(f"   Error: {str(e)}")
            raise
        else:
            # Client error (400, 401, 404, etc.) - NEVER use fallback
            # These indicate actual issues: invalid claim number, unauthorized, not found, etc.
            logger.error(f"❌ Claim Details API client error (non-retriable) - NO FALLBACK")
            logger.error(f"   Error: {str(e)}")
            logger.error(f"   📌 Client errors (4xx) indicate real issues like:")
            logger.error(f"      - 400: Invalid request/claim number format")
            logger.error(f"      - 401: Unauthorized access")
            logger.error(f"      - 404: Claim not found")
            logger.error(f"   ⚠️ Returning actual error to user (NO FAKE DATA)")
            raise
    
    except ToolTimeoutError as e:
        # Network timeout
        if settings.enable_api_fallback:
            # Fallback enabled - use mock data
            logger.warning(f"⚠️ Fallback: Claim Details API timeout - Using mock data for testing")
            logger.warning(f"   Error: {str(e)}")
            fallback_data = get_fallback_details(claimNumber, claimSequence)
            return fallback_data
        else:
            # Fallback disabled - return error
            logger.error(f"❌ Claim Details API timeout - Fallback disabled, returning error")
            raise
    
    except Exception as e:
        # Unexpected errors - always re-raise
        logger.error(f"❌ Unexpected error in get_claim_details: {str(e)}")
        raise


def get_claim_list(claimId: str, claimSequence: str = "1") -> Dict[str, Any]:
    """
    Fetch claim list by claim ID.
    Uses existing call_external_api() with retry logic.
    Falls back to dynamic generated data on failure.
    
    Args:
        claimId: The claim ID to search for
        claimSequence: The claim sequence (optional, used for fallback data generation)
        
    Returns:
        List of claims as returned by API or fallback
    """
    logger.debug(f"📞 Fetching claim list for claimId={claimId} (type: {type(claimId).__name__}, repr: {repr(claimId)})")
    
    # CRITICAL: Ensure claimId is a string, not a list
    if isinstance(claimId, list):
        logger.warning(f"⚠️ claimId is a list {claimId}, converting to string")
        claimId = str(claimId[0]) if claimId else ""
        logger.info(f"   ✅ Converted to: claimId={claimId}")
    
    try:
        # Get the API definition from repository
        registry = get_api_repository()
        api = next((a for a in registry if getattr(a, 'name', '') == 'get_claim_list'), None)
        
        if not api:
            raise ValueError("get_claim_list API not found in repository")
        
        # Build request body using the API's body template
        entities = {
            "claimId": claimId
        }
        body = api.body_template(entities)
        
        # Call external API with retry logic
        result = call_external_api(api, body)
        logger.debug(f"✅ Successfully fetched claim list")
        return result
    
    except ExternalAPIError as e:
        # Check if error is retriable (server down: 5xx errors, network issues)
        if e.retriable and settings.enable_api_fallback:
            # Server is down AND fallback is enabled - use mock data for testing
            logger.warning(f"⚠️ Fallback: Claim List API server error (retriable) - Using mock data for testing")
            logger.warning(f"   Error: {str(e)}")
            logger.warning(f"   💡 Fallback is ENABLED (enable_api_fallback=True) - Set to False in production")
            fallback_data = get_fallback_list(claimId, claimSequence)
            return fallback_data
        elif e.retriable:
            # Server is down but fallback is disabled - return error
            logger.error(f"❌ Claim List API server error - Fallback disabled, returning error")
            logger.error(f"   Error: {str(e)}")
            raise
        else:
            # Client error (400, 401, 404, etc.) - NEVER use fallback
            # These indicate actual issues: invalid claim ID, unauthorized, not found, etc.
            logger.error(f"❌ Claim List API client error (non-retriable) - NO FALLBACK")
            logger.error(f"   Error: {str(e)}")
            logger.error(f"   📌 Client errors (4xx) indicate real issues like:")
            logger.error(f"      - 400: Invalid request/claim ID format")
            logger.error(f"      - 401: Unauthorized access")
            logger.error(f"      - 404: Claim not found")
            logger.error(f"   ⚠️ Returning actual error to user (NO FAKE DATA)")
            raise
    
    except ToolTimeoutError as e:
        # Network timeout
        if settings.enable_api_fallback:
            # Fallback enabled - use mock data
            logger.warning(f"⚠️ Fallback: Claim List API timeout - Using mock data for testing")
            logger.warning(f"   Error: {str(e)}")
            fallback_data = get_fallback_list(claimId, claimSequence)
            return fallback_data
        else:
            # Fallback disabled - return error
            logger.error(f"❌ Claim List API timeout - Fallback disabled, returning error")
            raise
    
    except Exception as e:
        # Unexpected errors - always re-raise
        logger.error(f"❌ Unexpected error in get_claim_list: {str(e)}")
        raise


def combine_claim_details_and_list(claimNumber: str, claimSequence: str) -> Dict[str, Any]:
    """
    Enriched claim details: combines claim details with filtered claim list.
    
    Flow:
        Step 1: Fetch claim details
        Step 2: Fetch claim list using claimId from details
        Step 3: Filter claim list by matching claimNumber and claimSequence
        Step 4: Merge filtered list record with claim details
        Step 5: Return enriched result
        
    Args:
        claimNumber: The claim number
        claimSequence: The claim sequence
        
    Returns:
        Merged enriched claim details object
    """
    logger.info(f"🔄 Starting enriched claim details flow")
    logger.info(f"   📌 Input: claimNumber={claimNumber} (type: {type(claimNumber).__name__}, repr: {repr(claimNumber)})")
    logger.info(f"   📌 Input: claimSequence={claimSequence} (type: {type(claimSequence).__name__}, repr: {repr(claimSequence)})")
    
    # CRITICAL: Ensure parameters are strings, not lists (defensive programming)
    if isinstance(claimNumber, list):
        logger.warning(f"⚠️ claimNumber is a list {claimNumber}, converting to string")
        claimNumber = str(claimNumber[0]) if claimNumber else ""
        logger.info(f"   ✅ Converted to: claimNumber={claimNumber}")
    
    if isinstance(claimSequence, list):
        logger.warning(f"⚠️ claimSequence is a list {claimSequence}, converting to string")
        claimSequence = str(claimSequence[0]) if claimSequence else "1"
        logger.info(f"   ✅ Converted to: claimSequence={claimSequence}")
    
    # Step 1: Get claim details
    logger.debug("Step 1: Fetching claim details")
    claim_details = get_claim_details(claimNumber, claimSequence)
    
    # Step 2: Get claim list (claimId is the claimNumber)
    logger.info(f"Step 2: Fetching claim list for claimId={claimNumber}, claimSequence={claimSequence}")
    try:
        claim_list_response = get_claim_list(claimNumber, claimSequence)
        logger.info(f"   ✅ Claim list fetched successfully")
        logger.debug(f"   Response type: {type(claim_list_response).__name__}")
        if isinstance(claim_list_response, dict):
            logger.debug(f"   Response keys: {list(claim_list_response.keys())}")
    except Exception as e:
        logger.error(f"   ❌ Failed to fetch claim list: {e}")
        logger.error(f"   Returning details only (no enrichment)")
        return claim_details  # Return just the details without list_data
    
    # Extract the actual list from response
    # Try multiple possible keys where the claim list might be located
    logger.info("   Extracting claim list from response...")
    logger.debug(f"   claim_list_response type: {type(claim_list_response).__name__}")
    
    claim_list = []
    if isinstance(claim_list_response, dict):
        logger.debug(f"   claim_list_response keys: {list(claim_list_response.keys())}")
        
        # Common response structures: {"claims": [...]} or {"data": [...]} or {"claimsList": [...]} or {"claimList": [...]}
        # Use explicit None check to handle empty lists vs missing keys
        for key in ["claims", "data", "claimsList", "claimList"]:
            if key in claim_list_response and claim_list_response[key] is not None:
                claim_list = claim_list_response[key]
                list_len = len(claim_list) if isinstance(claim_list, list) else "not a list"
                logger.info(f"   ✅ Found claim list in key '{key}' with {list_len} items")
                break
        
        # If no valid key found, default to empty list
        if not claim_list:
            logger.error(f"   ❌ No standard claim list key found in response!")
            logger.error(f"      Available keys: {list(claim_list_response.keys())}")
            logger.error(f"      Expected keys: ['claims', 'data', 'claimsList', 'claimList']")
            claim_list = []
    elif isinstance(claim_list_response, list):
        claim_list = claim_list_response
        logger.info(f"   ✅ Claim list response is a direct list with {len(claim_list)} claims")
    else:
        logger.error(f"   ❌ Unexpected claim_list_response type: {type(claim_list_response)}, using empty list")
        claim_list = []
    
    # Step 3: Filter claim list by matching claimNumber and claimSequence
    logger.info(f"Step 3: Filtering claim list (total: {len(claim_list)} claims)")
    logger.debug(f"   Looking for: claimNumber={claimNumber}, claimSequence={claimSequence}")
    
    # DEBUG: Log first claim structure to understand field names
    if claim_list and len(claim_list) > 0 and isinstance(claim_list[0], dict):
        logger.info(f"📋 First claim keys: {list(claim_list[0].keys())}")
        logger.debug(f"📋 First claim sample: {str(claim_list[0])[:500]}...")
    
    matched_claim = None
    for claim in claim_list:
        if isinstance(claim, dict):
            # Handle nested structure: check top level, claimInformation, and primary objects
            claim_info = claim.get("claimInformation", {})
            claim_primary = claim.get("primary", {})
            
            # Extract claim number from various possible locations
            # NOTE: API returns structure like {'primary': {'number': 123, 'sequence': 456}}
            claim_num = get_first_non_none([
                (claim_primary, "number"),      # ACTUAL API FIELD - check first!
                (claim, "claimNumber"),
                (claim, "claimId"),
                (claim, "claim_number"),
                (claim, "claim_id"),
                (claim_info, "claimNumber"),
                (claim_info, "claimId")
            ])
            
            # Extract claim sequence from various possible locations
            # NOTE: API returns structure like {'primary': {'number': 123, 'sequence': 456}}
            claim_seq = get_first_non_none([
                (claim_primary, "sequence"),    # ACTUAL API FIELD - check first!
                (claim, "claimSequence"),
                (claim, "sequenceNumber"),
                (claim, "claimSequenceNumber"),
                (claim, "sequence"),
                (claim_info, "claimSequence"),
                (claim_info, "claimSequenceNumber"),  # Add this field!
                (claim_info, "sequenceNumber")
            ])
            
            # Debug: Log what we're comparing
            logger.debug(f"   Comparing: API claim_num={claim_num}, claim_seq={claim_seq} vs Target claimNumber={claimNumber}, claimSequence={claimSequence}")
            
            if str(claim_num) == str(claimNumber) and str(claim_seq) == str(claimSequence):
                matched_claim = claim
                logger.info(f"✅ Found matching claim in list: claimNumber={claim_num}, claimSequence={claim_seq}")
                break
            else:
                logger.debug(f"   ❌ No match: {str(claim_num) == str(claimNumber)} (num) and {str(claim_seq) == str(claimSequence)} (seq)")
    
    # Step 4: Merge filtered list record with claim details
    logger.info("Step 4: Merging claim details with list data")
    logger.debug(f"   claim_details type: {type(claim_details).__name__}")
    logger.debug(f"   claim_details keys: {list(claim_details.keys()) if isinstance(claim_details, dict) else 'N/A'}")
    
    enriched_details = claim_details.copy() if isinstance(claim_details, dict) else {}
    
    if matched_claim:
        # Add list data as nested field (no common fields, so no duplicates)
        enriched_details["list_data"] = matched_claim
        logger.info("✅ Successfully enriched claim details with list_data")
        logger.info(f"   📦 list_data keys: {list(matched_claim.keys()) if isinstance(matched_claim, dict) else 'N/A'}")
        logger.info(f"   📤 Final enriched_details keys: {list(enriched_details.keys())}")
    else:
        logger.error(f"❌ CRITICAL: No matching claim found in list - enrichment failed!")
        logger.error(f"   🔍 Search criteria:")
        logger.error(f"      Target claimNumber: {claimNumber} (type: {type(claimNumber).__name__}, repr: {repr(claimNumber)})")
        logger.error(f"      Target claimSequence: {claimSequence} (type: {type(claimSequence).__name__}, repr: {repr(claimSequence)})")
        logger.error(f"   📋 Claim list info:")
        logger.error(f"      Total claims: {len(claim_list)}")
        logger.error(f"      Claim list type: {type(claim_list).__name__}")
        
        # Log all claims in the list for debugging
        if claim_list and isinstance(claim_list, list):
            logger.error(f"   📝 Claims in list:")
            for idx, claim in enumerate(claim_list):
                if isinstance(claim, dict):
                    claim_info = claim.get("claimInformation", {})
                    claim_primary = claim.get("primary", {})
                    # Check all possible field locations using helper function
                    c_num = get_first_non_none([
                        (claim_primary, "number"),
                        (claim, "claimNumber"),
                        (claim, "claimId"),
                        (claim_info, "claimNumber")
                    ])
                    c_seq = get_first_non_none([
                        (claim_primary, "sequence"),
                        (claim, "claimSequence"),
                        (claim, "sequenceNumber"),
                        (claim_info, "claimSequence")
                    ])
                    logger.error(f"      [{idx}] claimNumber={c_num} (type: {type(c_num).__name__}), claimSequence={c_seq} (type: {type(c_seq).__name__})")
                    # Log first claim's actual keys to help debug
                    if idx == 0:
                        logger.error(f"      [0] Available keys: {list(claim.keys())}")
                        if claim_primary:
                            logger.error(f"      [0] primary keys: {list(claim_primary.keys())}")
                        if claim_info:
                            logger.error(f"      [0] claimInformation keys: {list(claim_info.keys())}")
        
        logger.error(f"   ⚠️ Returning details only (NO list_data field)")
    
    logger.info(f"Step 5: Returning enriched result")
    logger.info(f"   📤 Final keys being returned: {list(enriched_details.keys()) if isinstance(enriched_details, dict) else 'N/A'}")
    logger.info(f"   🔍 Has 'list_data' key: {'list_data' in enriched_details if isinstance(enriched_details, dict) else 'N/A'}")
    
    # FINAL VERIFICATION: Ensure list_data is present
    if isinstance(enriched_details, dict):
        if "list_data" in enriched_details:
            logger.info(f"   ✅ SUCCESS: Returning enriched details WITH list_data")
        else:
            logger.error(f"   ❌ WARNING: Returning details WITHOUT list_data - enrichment incomplete!")
    
    # Step 5: Return merged result
    return enriched_details
