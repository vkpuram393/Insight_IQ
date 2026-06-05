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
    return_details: bool = False,
    capture_thinking: Optional[bool] = None,
):
    """LLM fallback — routes to domain-aware system for high accuracy.

    Tries the domain-aware fallback first (with full decision trees and
    confusion-pair disambiguation), and falls back to a direct Gemini
    call with a simpler prompt if the domain-aware system is unavailable.

    Args:
        query:               Raw user query (with claim numbers).
        candidates:          Top-5 intent names from the ensemble.
        ensemble_intent:     The ensemble's top prediction (for context).
        ensemble_confidence: The ensemble's confidence (for context).
        return_details:      When True, return a dict with keys
                             {intent, thinking, reasoning, source}.
                             When False (default, for backward compat),
                             return the bare predicted intent string.

    Returns:
        Predicted intent name (str), or details dict (see ``return_details``).
    """
    if not candidates:
        bare = ensemble_intent or "unknown"
        return {"intent": bare, "thinking": None, "reasoning": None, "source": "noop"} if return_details else bare

    # ── Try domain-aware fallback first (98%+ accuracy) ──────────────
    if _check_domain_aware():
        try:
            return _domain_aware_classify(
                query, candidates, ensemble_intent, ensemble_confidence,
                return_details=return_details,
            )
        except Exception as e:
            logger.exception(f"Domain-aware fallback failed, trying simple: {e}")

    # ── Fallback: direct Gemini call with simple prompt ──────────────
    return _simple_classify(
        query, candidates, ensemble_intent, ensemble_confidence,
        return_details=return_details,
    )


def _domain_aware_classify(
    query: str,
    candidates: List[str],
    ensemble_intent: Optional[str],
    ensemble_confidence: float,
    return_details: bool = False,
):
    """Use the full domain-aware fallback with decision trees and expert prompts."""
    from prompt_templates.domain_prompts.llm_fallback import llm_fallback_classify

    # Build top5_intents as (name, probability) tuples
    # Assign decreasing dummy probs since we only have names
    top5_intents: List[Tuple[str, float]] = [
        (c, max(0.05, ensemble_confidence - i * 0.08))
        for i, c in enumerate(candidates)
    ]

    fallback_model = "gemini-2.5-flash"
    try:
        from config.config import settings as _app_settings
        fallback_model = getattr(_app_settings, "llm_fallback_model", "gemini-2.5-flash")
    except Exception:
        pass

    result = llm_fallback_classify(
        query=query,
        top5_intents=top5_intents,
        ensemble_intent=ensemble_intent,
        ensemble_confidence=ensemble_confidence,
        model=fallback_model,
    )

    predicted = result.get("intent", "")
    thinking = result.get("thinking")
    reasoning = result.get("reasoning")
    llm_confidence = float(result.get("confidence", 0.5))

    # Validate against candidates or known intents
    final = None
    for c in candidates:
        if c.lower() == predicted.lower():
            final = c
            break
    if final is None and predicted in INTENT_TO_DOMAIN:
        final = predicted
    if final is None:
        logger.warning(f"Domain-aware LLM returned '{predicted}', not in candidates")
        final = candidates[0]

    if return_details:
        return {"intent": final, "thinking": thinking, "reasoning": reasoning, "source": "domain_aware", "confidence": llm_confidence}
    return final


def _simple_classify(
    query: str,
    candidates: List[str],
    ensemble_intent: Optional[str],
    ensemble_confidence: float,
    return_details: bool = False,
):
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

SINGLE-CLAIM vs SEARCH confusion pairs:
- settlement_info = settlement codes for ONE specific claim | Settlement = SEARCH claims BY settlement code
- pricing_info = pricing for ONE claim | Pricing = cost of a DRUG across MANY claims  
- pharmacy_info = which pharmacy filled ONE claim | Pharmacy = SEARCH claims FROM a pharmacy
- prescriber_info = who prescribed ONE claim | Prescriber = SEARCH claims BY prescriber
- mail_order_info = was THIS specific claim filled via mail order | PharmType = FILTER claims by pharmacy TYPE (retail, mail-order, specialty)
- rx_details = RX number, fill number, quantity for ONE claim | Refills = SEARCH claims by refill count across history
- fill_date_info = date a specific claim was filled | DrugLast = when was a DRUG LAST dispensed for a member

BENEFITS vs CAP confusion:
- approval_info = WHY was claim APPROVED (overrides, TF, BPG) — approval LOGIC
- claim_status = WHAT is the current status (paid/rejected/pending) — claim OUTCOME
- beneficiary_info = benefit PHASE and ACCUMULATIONS for a claim
- member_coverage = coverage ELIGIBILITY windows for a MEMBER (member_domain)
- audit_info = WHEN was claim created/modified — timestamps

PRICING confusion:
- pricing_info = copay, ingredient cost, patient pay for ONE claim
- medicare_part_d = MEDD pricing, PDE, Part D specific pricing
- compound_info = COMPOUND drug ingredients, MIC breakdown
- cob_info = coordination pricing, dual coverage pricing

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

GREETING vs OUT_OF_SCOPE:
- greeting = hello, hi, welcome, good morning — SALUTATION
- out_of_scope = unrelated questions (weather, sports, recipes, etc.)
- "What's up" = greeting (casual salutation)

OUTPUT: Return ONLY valid JSON: {{"intent": "<name>", "confidence": <0-1>, "reasoning": "<brief>"}}

CONFIDENCE RULE: Use >= 0.88 for clear single-intent matches. The routing threshold is 0.85 —
only return < 0.85 when genuinely uncertain between multiple candidates.
CLAIM-AS-MEMBER-CONTEXT: A claim number in a SEARCH query identifies the member, not a specific
claim. In this pattern the claim_history_search intent is correct — assign confidence >= 0.90.
"""

    user_prompt = (
        f"Query: {query}\n"
        "Classify step by step: (1) core action, (2) one claim or many, "
        "(3) filter criterion, (4) best matching intent, (5) confidence.\n"
    )
    if ensemble_intent:
        user_prompt += (
            f"Advisory (do not anchor on this): classifier suggested '{ensemble_intent}' "
            f"({ensemble_confidence:.0%}) but was uncertain — classify from query text only.\n"
        )

    backoff = _LLM_INITIAL_BACKOFF
    for attempt in range(_LLM_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=400,
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
            if return_details:
                return {"intent": candidates[0], "thinking": None, "reasoning": f"LLM error: {e}", "source": "simple_error", "confidence": 0.0}
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
        reasoning = llm_result.get("reasoning")
        llm_confidence = float(llm_result.get("confidence", 0.5))

        final = None
        for c in candidates:
            if c.lower() == predicted.lower():
                final = c
                break
        if final is None and predicted in INTENT_TO_DOMAIN:
            final = predicted
        if final is None:
            logger.warning(f"LLM returned unknown intent '{predicted}', using ensemble pick")
            final = candidates[0]

        if return_details:
            return {"intent": final, "thinking": None, "reasoning": reasoning, "source": "simple", "confidence": llm_confidence}
        return final

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"LLM response parse failed: {e}, text='{text[:100]}'")
        if return_details:
            return {"intent": candidates[0], "thinking": None, "reasoning": f"parse error: {e}", "source": "simple_error", "confidence": 0.0}
        return candidates[0]
