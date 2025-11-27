"""
Dynamic Claims API Orchestrator (final version).
Single entry node: call_claims_tool_node(state)
Uses repository-based API registry and returns ToolResult (with AgentError)
"""

import asyncio
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
            entities = state.get("entities", {})
            logger.debug(f"Input type: AgentState (dict)")
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
                agent_error=ae,
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
                agent_error=ae,
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
        # ========================================================================
        if intent == "claim_details":
            logger.info("🔄 Special flow detected: claim_details intent - using enriched 2-step flow")
            
            # Validate required entities for claim_details
            if "claimNumber" not in entities or "claimSequence" not in entities:
                logger.warning(f"❌ Missing required entities for claim_details: {entities}")
                ae = create_validation_error(
                    message="claim_details intent requires both claimNumber and claimSequence",
                    field="entities",
                    value=entities
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
                    claimNumber=entities["claimNumber"],
                    claimSequence=entities["claimSequence"]
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
                
                tool_result = ToolResult(
                    tool_name="claim_details_enriched",
                    status=ToolExecutionStatus.FAILURE,
                    data={},
                    error_message=ae.message,
                    error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                    agent_error=ae,
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
                agent_error=ae,
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
            body["requester"] = {
                "xCorrelationId":str(uuid.uuid4()),
                "xConsumerAppName":"PSS-MYCLAIMSPOC-CLAIM-MFE"
            }
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
                agent_error=ae,
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

            tool_result = ToolResult(
                tool_name=getattr(api, "name", "claims_api"),
                status=ToolExecutionStatus.FAILURE,
                data={},
                error_message=ae.message,
                error_code=ae.error_code.value if hasattr(ae.error_code, "value") else str(ae.error_code),
                agent_error=ae,
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
    "claim_id":"claimId",
    "claim_ids":"claimId",  # Map plural to singular for API
    "claim_sequences":"claimSequence",  # Map plural sequences
}

# --- Helper Function to handle Pydantic model extraction ---
def normalize_entities(entities_obj) -> Dict[str, Any]:
    """
    1. Merges top-level entities with raw_entities from the Pydantic model or dict.
    2. Normalizes all keys from snake_case to target API's camelCase format.
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
    
    # Normalize keys to the target API format (camelCase)
    normalized_entities = {}
    for k, v in all_entities.items():
        # Use the mapped name if available, otherwise use the original key
        target_key = ENTITY_MAP.get(k, k)
        
        # Handle list values - take first element for singular API parameters
        if isinstance(v, list) and len(v) > 0:
            # For plural entity names mapping to singular API params, use first value
            if k in ['claim_ids', 'member_ids'] and target_key in ['claimId', 'memberId']:
                v = v[0]  # Take first claim/member ID
        
        normalized_entities[target_key] = v
        
    return normalized_entities

# ============================================================================
# API MATCHING LOGIC
# ============================================================================
def match_api(intent: str, entities: Dict[str, Any]):
    """
    Simple scoring-based matcher:
      - required_entities must all be present
      - +1 per matched intent keyword (in intent lowercase)
      - +len(required_entities) bonus so more specific endpoints win ties
    """
    registry = get_api_repository()
    intent_lower = (intent or "").lower()

    matched_api = None
    matched_api_score = -1

    for api in registry:
        # ensure required entities present
        if not all(req in entities for req in getattr(api, "required_entities", [])):
            continue

        score = sum(1 for kw in getattr(api, "intent_keywords", []) if kw in intent_lower)
        score += len(getattr(api, "required_entities", []))

        if score > matched_api_score:
            matched_api_score = score
            matched_api = api

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
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-correlation-id":"test",
        "x-consumerAppName": "LOCAL-TEST",
        "x-clientRefId":"test"
    }

    try:
        resp = requests.request(method, url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.Timeout as e:
        # timeout -> retriable
        raise ToolTimeoutError(str(e), details={"api": getattr(api, "name", url)}, retriable=True)
    except requests.RequestException as e:
        # non-timeout HTTP error or connection error
        status = getattr(getattr(e, "response", None), "status_code", None)
        retriable = status is None or (500 <= status < 600)
        raise ExternalAPIError(str(e), details={"status": status, "api": getattr(api, "name", url)}, retriable=retriable)

    # parse JSON or raise
    try:
        return resp.json()
    except ValueError:
        # non-json response
        raise ExternalAPIError("Non-JSON response", details={"status": resp.status_code, "raw": resp.text, "api": getattr(api, "name", url)}, retriable=False)


# ============================================================================
# HELPER METHODS FOR ENRICHED CLAIM DETAILS
# ============================================================================
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
        body["requester"] = {
            "xCorrelationId": str(uuid.uuid4()),
            "xConsumerAppName": "PSS-MYCLAIMSPOC-CLAIM-MFE"
        }
        
        # Call external API with retry logic
        result = call_external_api(api, body)
        logger.debug(f"✅ Successfully fetched claim details")
        return result
    
    except Exception as e:
        # Fallback to dynamic generated data on any exception
        logger.warning(f"⚠️ Fallback: Claim Details API failed - Error: {str(e)}")
        fallback_data = get_fallback_details(claimNumber, claimSequence)
        return fallback_data


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
    logger.debug(f"📞 Fetching claim list for claimId={claimId}")
    
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
        body["requester"] = {
            "xCorrelationId": str(uuid.uuid4()),
            "xConsumerAppName": "PSS-MYCLAIMSPOC-CLAIM-MFE"
        }
        
        # Call external API with retry logic
        result = call_external_api(api, body)
        logger.debug(f"✅ Successfully fetched claim list")
        return result
    
    except Exception as e:
        # Fallback to dynamic generated data on any exception
        logger.warning(f"⚠️ Fallback: Claim List API failed - Error: {str(e)}")
        fallback_data = get_fallback_list(claimId, claimSequence)
        return fallback_data


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
    logger.info(f"🔄 Starting enriched claim details flow for claimNumber={claimNumber}, claimSequence={claimSequence}")
    
    # Step 1: Get claim details
    logger.debug("Step 1: Fetching claim details")
    claim_details = get_claim_details(claimNumber, claimSequence)
    
    # Step 2: Get claim list (claimId is the claimNumber)
    logger.debug(f"Step 2: Fetching claim list for claimId={claimNumber}, claimSequence={claimSequence}")
    claim_list_response = get_claim_list(claimNumber, claimSequence)
    
    # Extract the actual list from response
    claim_list = []
    if isinstance(claim_list_response, dict):
        # Common response structures: {"claims": [...]} or {"data": [...]} or direct list
        claim_list = (claim_list_response.get("claims") or 
                     claim_list_response.get("data") or 
                     claim_list_response.get("claimsList") or
                     [])
    elif isinstance(claim_list_response, list):
        claim_list = claim_list_response
    
    # Step 3: Filter claim list by matching claimNumber and claimSequence
    logger.debug(f"Step 3: Filtering claim list (total: {len(claim_list)} claims)")
    matched_claim = None
    for claim in claim_list:
        if isinstance(claim, dict):
            # Handle nested structure: check top level and claimInformation object
            claim_info = claim.get("claimInformation", {})
            
            # Try multiple field names and locations
            claim_num = (claim.get("claimNumber") or 
                        claim.get("claim_number") or
                        claim_info.get("claimNumber") or
                        claim_info.get("claim_number"))
            
            claim_seq = (claim.get("claimSequence") or 
                        claim.get("claim_sequence") or
                        claim.get("claimSequenceNumber") or
                        claim_info.get("claimSequence") or
                        claim_info.get("claim_sequence") or
                        claim_info.get("claimSequenceNumber"))
            
            if str(claim_num) == str(claimNumber) and str(claim_seq) == str(claimSequence):
                matched_claim = claim
                logger.debug(f"✅ Found matching claim in list: claimNumber={claim_num}, claimSequence={claim_seq}")
                break
    
    # Step 4: Merge filtered list record with claim details
    logger.debug("Step 4: Merging claim details with list data")
    enriched_details = claim_details.copy() if isinstance(claim_details, dict) else {}
    
    if matched_claim:
        # Add list data as nested field (no common fields, so no duplicates)
        enriched_details["list_data"] = matched_claim
        logger.info("✅ Successfully enriched claim details with list data")
    else:
        logger.warning(f"⚠️ No matching claim found in list for claimNumber={claimNumber}, claimSequence={claimSequence}, returning details only")
    
    # Step 5: Return merged result
    return enriched_details
