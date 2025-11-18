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
from tools.error_handler import to_agent_error
from tools.retry import retry
from tools.exceptions import ExternalAPIError, ToolTimeoutError

from core.node_models import ToolResult, ToolExecutionStatus
from core.error_models import (
    AgentError,
    create_validation_error,
    create_api_error,
    create_internal_error
)

from core.node_models import IntentResult
from state.schema import AgentState
from core.logger import get_logger
from core.logging_context import extract_logging_context
from core.config import settings
from persistence import PersistenceStoreFactory

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

            tool_result = ToolResult(
                tool_name=getattr(api, "name", "claims_api"),
                status=ToolExecutionStatus.SUCCESS,
                data=result if isinstance(result, dict) else {"result": result},
                error_message=None,
                error_code=None,
                agent_error=None,
                execution_time_ms=elapsed_ms,
                api_endpoint=getattr(api, "full_url", None),
                http_status_code=200,
                is_retryable=False
            )
            return {"tool_results": tool_result.dict()}

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
            return {"tool_results": tool_result.dict()}
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