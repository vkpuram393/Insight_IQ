"""
Overrides_api.intent_router — predicate used by the LangGraph
_route_after_build_context to detect override-domain queries.

Three priority tiers:
  1. state["domain_mapping"]["domain"] == "override_domain"   (richest signal)
  2. state["domain"] == "override_domain"                      (multidomain classifier)
  3. intent in OVERRIDE_INTENTS set                            (api_routing_config)
  4. regex fallback over user text                             (legacy / classifier-less)

If any tier returns True, the query is routed to call_overrides_tool.
"""

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


# 16 Prior Authorization intents — same set registered in
# config.api_routing_config.INTENT_API_ROUTING under domain="override_domain".
OVERRIDE_INTENTS = frozenset({
    "pa_summary",
    "pa_override_reject",
    "pa_field_help",
    "pa_copay_pricing",
    "pa_drug_coverage",
    "pa_claim_usage",
    "pa_reason_code",
    "pa_effective_dates",
    "pa_agent_code",
    "pa_ignore_status",
    "pa_specialty_rx_override",
    "pa_clinical_admin_code",
    "pa_transform_care",
    "pa_follow_me_logic",
    "pa_drug_type_indicator",
    "pa_modification_history",
    # New operational how-to intents
    "pa_contingent_therapy_override",
    "pa_smart_pa_override",
    "pa_part_b_override",
    "pa_esrd_override",
    "pa_skip_deductible",
    "pa_send_expiration",
    "pa_tf_letter_setup",
    "pa_copay_setup",
    "pa_suggest_override",
    "pa_reason_code_fields",
})


# Regex fallback — kept tight to avoid hijacking CHS queries that incidentally
# contain the word "override". Requires a strong PA marker.
_OVERRIDE_PATTERNS = [
    re.compile(r"\bprior[-\s]?auth(orization)?\b", re.IGNORECASE),
    re.compile(r"\bPA\s+\w{4,}\b"),                      # "PA JW012726LC"
    re.compile(r"\boverride\b.*\breject\b", re.IGNORECASE),
    re.compile(r"\bfollow[-\s]?me\s+logic\b", re.IGNORECASE),
    re.compile(r"\bspecialty\s+rx\s+override\b", re.IGNORECASE),
]


def is_override_domain_intent(intent: str) -> bool:
    """Return True if the given intent name belongs to the override domain."""
    return bool(intent) and intent.lower().strip() in OVERRIDE_INTENTS


def is_overrides_query(state: Dict[str, Any]) -> bool:
    """
    Determine whether the current LangGraph state should route to call_overrides_tool.

    Tier 1 — explicit domain_mapping (set by extended_intent_agent_node).
    Tier 2 — domain field from multidomain classifier.
    Tier 3 — intent in the override-domain intent set.
    Tier 4 — regex fallback over user text.
    """
    # Tier 1: domain_mapping (richest signal — set deterministically by intent agent)
    dm = state.get("domain_mapping")
    if isinstance(dm, dict) and dm.get("domain") == "override_domain":
        logger.info("🔍 domain_mapping.domain == 'override_domain' → overrides pipeline")
        return True

    # Tier 2: classifier domain field
    domain = (state.get("domain") or "").strip()
    if domain == "override_domain":
        logger.info("🔍 state['domain'] == 'override_domain' → overrides pipeline")
        return True

    # Tier 3: intent membership
    intent = (state.get("intent") or "").strip()
    if is_override_domain_intent(intent):
        logger.info("🔍 intent %r ∈ OVERRIDE_INTENTS → overrides pipeline", intent)
        return True

    # Tier 4: regex fallback (last resort)
    user_input = (state.get("user_input") or state.get("text") or "").strip()
    if not user_input:
        return False
    for pat in _OVERRIDE_PATTERNS:
        if pat.search(user_input):
            logger.info("🔍 Override regex %r matched: %r", pat.pattern, user_input)
            return True

    return False
