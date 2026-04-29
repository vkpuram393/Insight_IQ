"""
Multidomain Intent Detection — LLM Fallback
=============================================

When the ensemble classifier confidence is below threshold, this module
sends the top-5 candidates to Gemini Flash for disambiguation.

Used by classifier.py when:
  confidence < confidence_threshold  OR  margin < margin_threshold
"""

import os
import re
import json
import time
import logging
import threading
from typing import List, Optional

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


def llm_classify(
    query: str,
    candidates: List[str],
    ensemble_intent: Optional[str] = None,
    ensemble_confidence: float = 0.0,
) -> str:
    """LLM fallback — sends top-5 candidates to Gemini for disambiguation.

    Args:
        query:               Raw user query (with claim numbers).
        candidates:          Top-5 intent names from the ensemble.
        ensemble_intent:     The ensemble's top prediction (for context).
        ensemble_confidence: The ensemble's confidence (for context).

    Returns:
        Predicted intent name (guaranteed to be a valid intent).
    """
    # Guard against empty candidate list
    if not candidates:
        return ensemble_intent or "unknown"

    from google.genai import types

    client = _get_llm_client()

    candidate_desc = "\n".join(
        f"- {name}: {INTENT_DESCRIPTIONS.get(name, name)}" for name in candidates
    )

    system_instruction = f"""You are an intent classification system for a Pharmacy Benefit Manager (PBM).
Classify the user query into exactly ONE of these candidate intents:

{candidate_desc}

KEY RULES:
- cap_api intents = details about ONE specific claim (references "this claim" or a claim number)
- claim_history_search intents = SEARCH/FILTER across MULTIPLE claims
- member_domain intents = member demographics, eligibility, coverage, contact info, DUR config
- override_domain intents = PA management, override analysis, PA fields/dates/codes
- benefits_api intents = benefit plan summary, history, finder; claim approval/audit details

CRITICAL DISAMBIGUATION RULES:
- "approval status/messages/details for claim" → approval_info (benefits_api), NOT claim_status
- "PA approval/type/authorization details for claim" → approval_info (benefits_api), NOT prior_auth_info
- "steps to avoid rejection / guidelines / instructions to prevent" → rejection_reasons (cap_api), NOT help
- "is claim approved or denied / approval or rejection" → rejection_reasons (cap_api), NOT claim_status
- "adjudication pathway / comprehensive summary" → claim_status (cap_api)
- "ingredient cost / approved ingredient cost for claim" → pricing_info (cap_api), NOT compound_info
- "MEDD pricing / Part D pricing for claim" → medicare_part_d (cap_api), NOT pricing_info
- "coordination pricing / COB pricing for claim" → cob_info (cap_api), NOT pricing_info
- "generic alternatives / therapeutic equivalents" → generic_availability (cap_api), NOT Generic
- "fill date / when was prescription filled" → fill_date_info, NOT rx_details
- "PA overview/summary" → pa_summary (override_domain), NOT prior_auth_info

The key distinction for approval_info vs prior_auth_info:
- approval_info = WHY was a claim APPROVED (overrides, TF, BPG) — looks at the APPROVAL LOGIC
- prior_auth_info = does the claim NEED PA / what's the PA STATUS — looks at PA REQUIREMENTS
- If query mentions "approval", "approved", "approval status", "approval messages" → approval_info
- If query mentions "PA required", "does claim need PA", "PA status" → prior_auth_info

OUTPUT FORMAT (JSON only):
{{"intent": "<one of the candidates>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}}
"""

    user_prompt = f"Query: {query}\n"
    if ensemble_intent:
        user_prompt += (
            f"Note: Primary classifier suggested '{ensemble_intent}' "
            f"({ensemble_confidence:.0%} confidence) but was uncertain.\n"
        )

    # ── Retry with exponential backoff ───────────────────────────────
    backoff = _LLM_INITIAL_BACKOFF
    for attempt in range(_LLM_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=100,
                    system_instruction=system_instruction,
                ),
            )
            text = response.text.strip()
            break  # success
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

    # ── Parse response ───────────────────────────────────────────────
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

        # Validate: exact match (case-insensitive) against candidates
        for c in candidates:
            if c.lower() == predicted.lower():
                return c

        # Validate: exact match against all known intents
        if predicted in INTENT_TO_DOMAIN:
            return predicted

        # No match — log and fall back
        logger.warning(f"LLM returned unknown intent '{predicted}', using ensemble pick")
        return candidates[0]

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"LLM response parse failed: {e}, text='{text[:100]}'")
        return candidates[0]
