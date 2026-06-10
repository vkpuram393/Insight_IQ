"""
Override domain intent router.

Determines whether an AgentState should be routed to the override
(Prior Authorization) pipeline instead of the standard single-claim flow.

Routing strategy:
  1. state["domain"] == "override_domain"  (multidomain classifier signal)
  2. state["intent"] belongs to OVERRIDE_DOMAIN_INTENTS  (fallback check)
"""

from typing import Dict, Any, FrozenSet
from multidomain_intent_detection.config import INTENT_TO_DOMAIN
from core.logger import get_logger

logger = get_logger(__name__)

# All intent labels that belong to the override domain
OVERRIDE_DOMAIN_INTENTS: FrozenSet[str] = frozenset(
    intent for intent, domain in INTENT_TO_DOMAIN.items()
    if domain == "override_domain"
)


def is_override_query(state: Dict[str, Any]) -> bool:
    """
    Return True when the state should be routed to the override (PA) pipeline.

    Checks domain first (preferred — multidomain classifier output),
    then falls back to intent-level check.
    """
    domain = (state.get("domain") or "").strip()
    if domain == "override_domain":
        logger.info("🔀 Domain == 'override_domain' → override pipeline")
        return True

    intent = (state.get("intent") or "").strip()
    if intent and intent in OVERRIDE_DOMAIN_INTENTS:
        logger.info(f"🔀 Intent '{intent}' belongs to override_domain → override pipeline")
        return True

    return False
