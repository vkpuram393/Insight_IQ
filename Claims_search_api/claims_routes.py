"""
Claims_search_api.claims_routes

FastAPI router for the claims-search domain.
Provides a /claims-search/chat endpoint that runs the full pipeline:
  user query → API fetch → filter → LLM → response.

This is a standalone router — it does NOT modify the existing api/routes.py.
It is registered in main.py alongside the existing router.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import uuid

from Claims_search_api.claims_search_orchestrator import run_claims_search_pipeline
from core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ClaimsSearchRequest(BaseModel):
    """Request body for the claims-search chat endpoint."""
    text: str                                  # User's natural language query
    claim_id: str                              # Claim ID to look up (triggers member history fetch)
    session_id: Optional[str] = None           # Optional session tracking
    user_info: Optional[Dict[str, Any]] = None # Optional user context


class ClaimsSearchResponse(BaseModel):
    """Response body mirroring the existing ChatResponse shape."""
    response: str                              # LLM-generated answer
    session_id: str
    filtered_count: int                        # Claims that matched the filter
    total_count: int                           # Total claims from API
    success: bool
    error: Optional[str] = None
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ClaimsSearchResponse)
async def claims_search_chat(request: ClaimsSearchRequest, http_request: Request):
    """
    Claims-search chat endpoint.

    Flow:
      1. Extract auth headers from the HTTP request
      2. Run the claims search pipeline (fetch → filter → LLM)
      3. Return the LLM-generated response

    Auth headers expected (same as existing claims API):
      - Authorization: Bearer <token>
      - x-api-key: <key>
      - x-clientrefid: <ref-id>
    """
    session_id = request.session_id or str(uuid.uuid4())

    # Extract auth headers
    bearer_token = http_request.headers.get("Authorization", "")
    x_api_key = http_request.headers.get("x-api-key", "")
    x_clientrefid = http_request.headers.get("x-clientrefid", str(uuid.uuid4()))

    if not bearer_token:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
        )
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing x-api-key header",
        )

    logger.info(
        f"[ClaimsSearchRoute] Query: {request.text!r}, "
        f"claim_id={request.claim_id}, session={session_id}"
    )

    # Run the full pipeline
    result = await run_claims_search_pipeline(
        user_query=request.text,
        claim_id=request.claim_id,
        bearer_token=bearer_token,
        x_api_key=x_api_key,
        x_clientrefid=x_clientrefid,
        session_id=session_id,
    )

    return ClaimsSearchResponse(
        response=result["response"],
        session_id=session_id,
        filtered_count=result["filtered_count"],
        total_count=result["total_count"],
        success=result["success"],
        error=result.get("error"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
