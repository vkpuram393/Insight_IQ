"""
Multidomain Intent Detection — LLM Fallback
=============================================

When the ensemble classifier confidence is below threshold, this module
delegates to the domain-aware LLM fallback system in
prompt_templates/domain_prompts/llm_fallback.py which has:

  - Full domain-specific expert prompts with decision trees
  - Confusion-pair disambiguation tables
  - Cross-domain routing rules
  - Entity extraction rules and PBM acronym glossary

This module is a thin adapter that:
  1. Converts the simple (query, candidates) API into the domain-aware format
  2. Falls back to a direct Gemini call if the domain-aware system is unavailable
"""

import os
import sys
import re
import json
import time
import logging
import threading
from typing import List, Optional, Tuple

from multidomain_intent_detection.config import INTENT_DESCRIPTIONS, INTENT_TO_DOMAIN

logger = logging.getLogger(__name__)

# ── Singleton Gemini client (thread-safe, lazy-init) ─────────────────────────
_llm_client = None
_llm_lock = threading.Lock()


def _get_llm_client():
    """Return a shared genai.Client instance (one gRPC channel per process)."""
    global _llm_client
    if _llm_client is None:
        with _llm_lock:
            if _llm_client is None:
                from google import genai
                _llm_client = genai.Client(
                    vertexai=True,
                    project=os.getenv("PROJECT_ID", "pbm-poc-coderev-genai-poc"),
                    location=os.getenv("LOCATION", "us-central1"),
                )
                logger.info("LLM fallback client initialized")
    return _llm_client


# ── Retry config ─────────────────────────────────────────────────────────────
_LLM_MAX_RETRIES = 3
_LLM_INITIAL_BACKOFF = 1.0

# ── Domain-aware fallback availability ───────────────────────────────────────
_domain_aware_available = None


def _check_domain_aware():
    """Check if the domain-aware fallback system is importable."""
    global _domain_aware_available
    if _domain_aware_available is not None:
        return _domain_aware_available
    try:
        # Ensure prompt_templates is on the path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from prompt_templates.domain_prompts.llm_fallback import llm_fallback_classify  # noqa: F401
        _domain_aware_available = True
        logger.info("Domain-aware LLM fallback available")
    except ImportError as e:
        _domain_aware_available = False
        logger.warning(f"Domain-aware fallback unavailable, using simple fallback: {e}")
    return _domain_aware_available


def llm_classify(
    query: str,
    candidates: List[str],
    ensemble_intent: Optional[str] = None,
    ensemble_confidence: float = 0.0,
) -> str:
    """LLM fallback — routes to domain-aware system for high accuracy.

    Tries the domain-aware fallback first (with full decision trees and
    confusion-pair disambiguation), and falls back to a direct Gemini
    call with a simpler prompt if the domain-aware system is unavailable.

    Args:
        query:               Raw user query (with claim numbers).
        candidates:          Top-5 intent names from the ensemble.
        ensemble_intent:     The ensemble's top prediction (for context).
        ensemble_confidence: The ensemble's confidence (for context).

    Returns:
        Predicted intent name (guaranteed to be a valid intent).
    """
    if not candidates:
        return ensemble_intent or "unknown"

    # ── Try domain-aware fallback first (98%+ accuracy) ──────────────
    if _check_domain_aware():
        try:
            return _domain_aware_classify(
                query, candidates, ensemble_intent, ensemble_confidence
            )
        except Exception as e:
            logger.warning(f"Domain-aware fallback failed, trying simple: {e}")

    # ── Fallback: direct Gemini call with simple prompt ──────────────
    return _simple_classify(query, candidates, ensemble_intent, ensemble_confidence)


def _domain_aware_classify(
    query: str,
    candidates: List[str],
    ensemble_intent: Optional[str],
    ensemble_confidence: float,
) -> str:
    """Use the full domain-aware fallback with decision trees and expert prompts."""
    from prompt_templates.domain_prompts.llm_fallback import llm_fallback_classify

    # Build top5_intents as (name, probability) tuples
    # Assign decreasing dummy probs since we only have names
    top5_intents: List[Tuple[str, float]] = [
        (c, max(0.05, ensemble_confidence - i * 0.08))
        for i, c in enumerate(candidates)
    ]

    result = llm_fallback_classify(
        query=query,
        top5_intents=top5_intents,
        ensemble_intent=ensemble_intent,
        ensemble_confidence=ensemble_confidence,
    )

    predicted = result.get("intent", "")

    # Validate against candidates or known intents
    for c in candidates:
        if c.lower() == predicted.lower():
            return c
    if predicted in INTENT_TO_DOMAIN:
        return predicted

    logger.warning(f"Domain-aware LLM returned '{predicted}', not in candidates")
    return candidates[0]


def _simple_classify(
    query: str,
    candidates: List[str],
    ensemble_intent: Optional[str],
    ensemble_confidence: float,
) -> str:
    """Direct Gemini call — used when domain-aware prompts are unavailable."""
    from google.genai import types

    client = _get_llm_client()

    candidate_desc = "\n".join(
        f"- {name}: {INTENT_DESCRIPTIONS.get(name, name)}" for name in candidates
    )

    system_instruction = f"""You are an expert intent classification system for a Pharmacy Benefit Manager (PBM).
Classify the user query into exactly ONE of these candidate intents:

{candidate_desc}

KEY RULES:
- cap_api intents = details about ONE specific claim (references "this claim" or a claim number)
- claim_history_search intents = SEARCH/FILTER across MULTIPLE claims (no specific claim number)
- member_domain intents = member demographics, eligibility, coverage (about the MEMBER, not a claim)
- override_domain intents = PA management/configuration (about the PA RECORD itself)
- benefits_api intents = benefit plan, approval logic, audit trail
- general intents = greetings, help, unrelated queries

CRITICAL DISAMBIGUATION (read carefully):

CLAIM NUMBER lookups:
- A bare claim number like "260302639954275" or "claim number 260302639954275" → ClaimNum (claim_history_search)
- "status of claim X" or "details for claim X" → claim_status (cap_api)
- "RX number / fill number / quantity for claim X" → rx_details (cap_api)

PHARMACY/PRESCRIPTION TERMINOLOGY:
- "script" = prescription in PBM context → prescriber_info (NOT out_of_scope)
- "Who wrote this script?" = Who is the prescribing doctor → prescriber_info
- "script writer" = prescribing physician → prescriber_info

SINGLE-CLAIM vs SEARCH confusion pairs:
- settlement_info = settlement codes for ONE specific claim | Settlement = SEARCH claims BY settlement code
- pricing_info = pricing for ONE claim | Pricing = cost of a DRUG across MANY claims
- pharmacy_info = which pharmacy filled ONE claim | Pharmacy = SEARCH claims FROM a pharmacy
- prescriber_info = who prescribed ONE claim | Prescriber = SEARCH claims BY prescriber
- mail_order_info = was THIS specific claim filled via mail order | PharmType = FILTER claims by pharmacy TYPE
- rx_details = RX number, fill number, quantity for ONE claim | Refills = SEARCH claims by refill count across history
- fill_date_info = date a specific claim was filled | DrugLast = when was a DRUG LAST dispensed for a member
- reversal_info = was THIS specific claim reversed | Status = FILTER/LIST claims by status across history

FILL NUMBER vs REFILL SEARCH:
- "fill number", "is this original or a refill", "new fill vs continuation" FOR THIS CLAIM → rx_details (cap_api)
- "Refills" = search claim history for claims by refill count (claim_history_search)
- Key: if the query references ONE specific claim's fill sequence → rx_details

MAIL ORDER vs PHARM TYPE vs DAYS SUPPLY:
- "fulfillment channel" or "retail or mail" for ONE specific claim → mail_order_info (cap_api)
- "Is this a 90-day mail order fill?" → mail_order_info (delivery channel, NOT days supply count)
- PharmType = filter MANY claims by pharmacy type | DaysSupply = filter MANY claims by days count
- mail_order_info = was THIS ONE SPECIFIC claim filled via mail order

COB vs ALTERNATE INSURANCE:
- cob_info = other insurance PAYMENT on THIS specific claim — coordination of benefits (cap_api)
- alternate_insurance = other/secondary insurance ON FILE for the MEMBER (member_domain)
- "other insurance payment on this claim" → cob_info
- "other insurance on file for this member" → alternate_insurance

PART D PRICING / CATASTROPHIC PHASE / LICS:
- medicare_part_d = Part D PRICING fields (MEDD, PDE, LIS amounts) ON A SPECIFIC CLAIM (cap_api)
- lics_status = MEMBER's LIS/subsidy enrollment status (member_domain, not claim-specific)
- beneficiary_info = benefit PHASE accumulations and YTD spend for the member (benefits_api)
- "low income subsidy applied to the pricing on this claim" → medicare_part_d (Part D claim pricing)
- "catastrophic coverage phase reached on this claim" → medicare_part_d (Part D benefit phases)
- "LICS status for this member" → lics_status (member record)

COVERAGE GAP / ACCUMULATIONS / YTD SPEND:
- beneficiary_info = accumulations, deductible progress, coverage gap closeness, YTD spend (benefits_api)
- member_coverage = eligibility dates and enrollment windows (member_domain)
- "coverage gap", "year-to-date spend", "YTD accumulations", "catastrophic threshold" → beneficiary_info
- "how close to coverage gap" → beneficiary_info

MEMBER PA vs PLAN PA on a CLAIM:
- "Was a member PA or plan PA used on this fill?" → prior_auth_info (cap_api) — PA status for one claim
- "PA summary/overview" → pa_summary (override_domain) — about the PA record itself

PA FOLLOW ME:
- "PA follow me?" or "follow me logic" → pa_follow_me_logic (override_domain)
- Short forms: "PA follow me?", "Follow me logic?" → pa_follow_me_logic
- This is about whether a PA configuration follows the member, NOT prior_auth_info

TF STATUS — CLAIM vs MEMBER ELIGIBILITY:
- "TF status on this member's eligibility record" → member_transition_status (member_domain)
- "TF applied to claim X", "was this a transition fill adjudication" → approval_info (benefits_api)
- Key: "eligibility record" or "member's TF status" → member_transition_status

PLAN SUMMARY vs MEMBER COVERAGE vs DRUG INFO:
- plan_summary = WHAT THE PLAN COVERS, plan name, formulary, benefit plan overview (benefits_api)
- member_coverage = WHEN IS THE MEMBER COVERED, eligibility dates (member_domain)
- drug_info = drug name, NDC, GPI, formulary status for a DRUG (claim_history_search)
- "formulary associated with this member's plan" → plan_summary (the plan's formulary)
- "formulary tier for this drug" → drug_info (a drug's formulary status)
- "plan name and effective date for this member" → plan_summary

HELP vs MEMBER QUERIES:
- "guidance on how to query X", "how do I look up X", "I need some help with a claim" → help
- If asking HOW TO USE the system → help
- If asking for actual member/claim data → use the relevant intent

BENEFITS vs CAP confusion:
- approval_info = WHY was claim APPROVED (overrides, TF, BPG) — approval LOGIC
- claim_status = WHAT is the current status (paid/rejected/pending) — claim OUTCOME
- beneficiary_info = benefit PHASE and ACCUMULATIONS for a claim
- member_coverage = coverage ELIGIBILITY windows for a MEMBER (member_domain)
- audit_info = WHEN was claim created/modified — timestamps

PRICING confusion:
- pricing_info = copay, ingredient cost, patient pay for ONE claim
- medicare_part_d = MEDD pricing, PDE, Part D specific pricing for ONE claim
- compound_info = COMPOUND drug ingredients, MIC breakdown
- cob_info = coordination pricing, dual coverage pricing for ONE claim

MEMBER confusion:
- member_coverage = coverage ELIGIBILITY windows, active status, enrollment dates (member_domain)
- plan_summary = benefit PLAN overview, what the plan covers, formulary (benefits_api)
- beneficiary_info = benefit PHASE, ACCUMULATIONS, deductible progress for a claim (benefits_api)
- family_type = INDIVIDUAL vs FAMILY plan classification, coverage tier TYPE (member_domain)
- family_members = LIST of dependents/subscribers on the plan (member_domain)

PA (OVERRIDE) confusion:
- pa_field_help = EXPLAIN what a PA field MEANS or DOES — field documentation
- pa_ignore_status = WHAT IS the ignore status VALUE on this PA
- pa_follow_me_logic = does this PA FOLLOW the member across PLAN CHANGES — PA configuration
- member_transition_status = the MEMBER's transition fill status from ELIGIBILITY — member record
- pa_summary = high-level OVERVIEW of a PA RECORD (override_domain)
- prior_auth_info = PA status/requirements for ONE SPECIFIC CLAIM (cap_api)

GREETING vs OUT_OF_SCOPE:
- greeting = hello, hi, welcome, good morning — SALUTATION (even with follow-up text like "I need help")
- out_of_scope = unrelated questions (weather, sports, recipes, etc.)
- "Hi — I need some help with a claim." = greeting (starts with salutation, help is secondary)
- "What's up" = greeting (casual salutation)

OUTPUT: Return ONLY valid JSON: {{"intent": "<name>", "confidence": <0-1>, "reasoning": "<brief>"}}
"""

    user_prompt = f"Query: {query}\n"
    if ensemble_intent:
        user_prompt += (
            f"Note: Primary classifier suggested '{ensemble_intent}' "
            f"({ensemble_confidence:.0%} confidence) but was uncertain.\n"
        )

    backoff = _LLM_INITIAL_BACKOFF
    for attempt in range(_LLM_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=150,
                    system_instruction=system_instruction,
                ),
            )
            text = response.text.strip()
            break
        except Exception as e:
            error_str = str(e).lower()
            is_retriable = any(
                k in error_str
                for k in ("429", "503", "resource exhausted", "unavailable", "deadline")
            )
            if is_retriable and attempt < _LLM_MAX_RETRIES - 1:
                logger.warning(f"LLM retry {attempt+1}/{_LLM_MAX_RETRIES} in {backoff:.0f}s: {e}")
                time.sleep(backoff)
                backoff *= 2
                continue
            logger.error(f"LLM fallback failed after {attempt+1} attempts: {e}")
            return candidates[0]

    try:
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        json_match = re.search(r'\{[^{}]*"intent"[^{}]*\}', text, re.DOTALL)
        if json_match:
            llm_result = json.loads(json_match.group(0))
        else:
            llm_result = json.loads(text)

        predicted = llm_result.get("intent", "")

        for c in candidates:
            if c.lower() == predicted.lower():
                return c
        if predicted in INTENT_TO_DOMAIN:
            return predicted

        logger.warning(f"LLM returned unknown intent '{predicted}', using ensemble pick")
        return candidates[0]

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"LLM response parse failed: {e}, text='{text[:100]}'")
        return candidates[0]
