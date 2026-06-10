"""
History API Routes - Conversation history endpoints

Endpoints:
  GET /session/{session_uuid} - Full history with pagination, feedback enrichment, and existence check
                        Always returns HTTP 200. Use the 'found' field to determine if
                        history exists (found: false = no history for this user_session).
"""

import uuid as uuid_mod
from datetime import datetime, timezone

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from config.config import settings
from persistence import PersistenceStoreFactory
from core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models  (match APIGEE doc schemas exactly)
# ---------------------------------------------------------------------------

class ConversationMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    response_id: Optional[str] = None
    feedback: Optional[str] = None


class ConversationHistoryResponse(BaseModel):
    session_uuid: str
    found: bool = True
    message_count: int
    turn_count: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    conversation_history: List[ConversationMessage]
    retrieved_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_datetime(val) -> Optional[str]:
    if val is None:
        return None
    return val.isoformat() if hasattr(val, "isoformat") else str(val)


def _error_response(
    status_code: int, error_type: str, title: str,
    detail: str, message: str = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "correlationId": str(uuid_mod.uuid4()),
            "type": error_type,
            "title": title,
            "status": status_code,
            "detail": detail,
            "message": message or detail,
        },
    )


def _validate_session_uuid(session_uuid: str) -> Optional[JSONResponse]:
    if not session_uuid or not session_uuid.strip():
        return _error_response(
            400, "validation_error", "Bad Request",
            "user_session cannot be empty",
        )
    if len(session_uuid) > 512:
        return _error_response(
            400, "validation_error", "Bad Request",
            f"user_session exceeds maximum length (got {len(session_uuid)}, max 512)",
        )
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/session/{session_uuid}",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history for a user session",
    description=(
        "Returns the full conversation history for the given session UUID. "
        "Always returns HTTP 200. When no history exists, returns found: false with "
        "an empty conversation_history array — this is the normal condition for "
        "first-time users. Use the 'found' boolean to determine whether to render "
        "prior messages or start a fresh chatbot. "
        "Supports pagination via limit/offset. "
        "Assistant messages include a 'feedback' field populated from Response_Feedback collection."
    ),
)
async def get_conversation_history(
    session_uuid: str,
    limit: Optional[int] = Query(
        default=None,
        description="Maximum messages to return (1-1000). Omit for full history.",
    ),
    offset: int = Query(default=0, description="Number of messages to skip from the start"),
):
    validation_error = _validate_session_uuid(session_uuid)
    if validation_error:
        return validation_error

    if limit is not None and (limit < 1 or limit > 1000):
        return _error_response(
            400, "validation_error", "Bad Request",
            "limit must be between 1 and 1000",
        )
    if offset < 0:
        return _error_response(
            400, "validation_error", "Bad Request",
            "offset must be >= 0",
        )

    try:
        store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)

        conv_doc: Optional[Dict[str, Any]] = await store.get_conversation_history(
            session_id=session_uuid,
            user_session=session_uuid,
        )

        if not conv_doc:
            logger.debug("No history found: session_uuid=%s — returning found:false", session_uuid)
            return ConversationHistoryResponse(
                session_uuid=session_uuid,
                found=False,
                message_count=0,
                turn_count=0,
                created_at=None,
                updated_at=None,
                conversation_history=[],
                retrieved_at=datetime.now(timezone.utc).isoformat(),
            )

        all_messages: List[Dict[str, Any]] = conv_doc.get("conversation_history", [])

        if limit is not None:
            paginated = all_messages[offset: offset + limit]
        else:
            paginated = all_messages[offset:]

        response_ids = [
            msg["response_id"]
            for msg in paginated
            if msg.get("role") == "assistant" and msg.get("response_id")
        ]

        feedback_map: Dict[str, str] = {}
        if response_ids:
            feedback_map = await store.get_feedback_for_responses(response_ids)

        messages: List[ConversationMessage] = []
        for msg in paginated:
            role = msg.get("role", "user")
            rid = msg.get("response_id")
            messages.append(
                ConversationMessage(
                    role=role,
                    content=msg.get("content", ""),
                    timestamp=msg.get("timestamp"),
                    response_id=rid,
                    feedback=feedback_map.get(rid) if rid else None,
                )
            )

        logger.info(
            "History returned: session_uuid=%s total=%d offset=%d limit=%s returned=%d",
            session_uuid, len(all_messages), offset, limit, len(messages),
        )

        return ConversationHistoryResponse(
            session_uuid=session_uuid,
            found=True,
            message_count=len(messages),
            turn_count=len(messages) // 2,
            created_at=_format_datetime(conv_doc.get("created_at")),
            updated_at=_format_datetime(conv_doc.get("updated_at")),
            conversation_history=messages,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error("History API error for session_uuid=%s: %s", session_uuid, e, exc_info=True)
        return _error_response(
            503, "service_unavailable", "Service Unavailable",
            "Conversation history service temporarily unavailable",
        )
