"""
Confidence Check - Conditional router
"""

from typing import Literal
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

def confidence_check_router(state: AgentState) -> Literal["clarification", "tool_call"]:
    """Route based on confidence and entity completeness.

    Rules:
      - If intent needs a claim_number and it's missing -> clarification
      - Else if confidence < threshold -> clarification
      - Else -> tool_call
    """
    intent = state.get("intent")
    entities = state.get("entities") or {}
    confidence = state.get("confidence", 0.0)
    threshold = settings.confidence_threshold

    # Entity completeness rule for claim rejection intent
    if intent == "claim_rejection_reason" and not entities.get("claim_number"):
        logger.info("⚠️ Missing claim_number for rejection intent -> Clarification")
        return "clarification"

    if confidence < threshold:
        logger.info(f"⚠️ Low confidence ({confidence:.2f}) < {threshold:.2f} -> Clarification")
        return "clarification"

    logger.info(f"✅ Confidence OK ({confidence:.2f}) -> Tool Call")
    return "tool_call"
