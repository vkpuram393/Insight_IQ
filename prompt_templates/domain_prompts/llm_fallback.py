"""
LLM Fallback Orchestrator — Domain-Aware Intent Classification Fallback

This module is the MAIN entry point for LLM-based fallback classification.
When the embedding classifier's confidence is below threshold, this module:

  1. Determines which domain(s) the top-5 candidates span
  2. Selects the appropriate domain-specific prompt (or multi-domain prompt)
  3. Calls the LLM with highly detailed, domain-expert instructions
  4. Parses and validates the response

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  Input: query + top-5 intents from ensemble                 │
  │  ↓                                                          │
  │  Domain Router: determine candidate domains                 │
  │  ↓                                                          │
  │  Build System Prompt:                                       │
  │    base_prompt (entities, acronyms, output format)          │
  │    + domain_prompt(s) (intent definitions, decision trees)  │
  │    + candidate focus (restrict to top-5 + their domains)    │
  │  ↓                                                          │
  │  LLM Call (Gemini 2.0 Flash, temperature=0.0)               │
  │  ↓                                                          │
  │  Parse JSON → Validate intent → Return                      │
  └─────────────────────────────────────────────────────────────┘

Usage:
  from prompt_templates.domain_prompts.llm_fallback import llm_fallback_classify

  result = llm_fallback_classify(
      query="Settlement details for claim 220133725669000 sequence 001.",
      top5_intents=[("settlement_info", 0.45), ("Settlement", 0.30), ...],
      ensemble_intent="settlement_info",
      ensemble_confidence=0.45,
      conversation_history=[],
  )
  # result = {"intent": "settlement_info", "confidence": 0.95, "domain": "cap_api", ...}
"""

import os
import json
import re
import logging
import time
from typing import List, Dict, Any, Tuple, Optional

from prompt_templates.domain_prompts.base_prompt import (
    INTENT_TO_DOMAIN,
    ENTITY_EXTRACTION_RULES,
    CONVERSATION_HISTORY_RULES,
    PBM_ACRONYM_GLOSSARY,
    OUTPUT_FORMAT,
    get_candidate_domains,
)
from prompt_templates.domain_prompts.cap_api_prompt import CAP_API_PROMPT
from prompt_templates.domain_prompts.benefits_api_prompt import BENEFITS_API_PROMPT
from prompt_templates.domain_prompts.claim_history_search_prompt import CLAIM_HISTORY_SEARCH_PROMPT
from prompt_templates.domain_prompts.member_domain_prompt import MEMBER_DOMAIN_PROMPT
from prompt_templates.domain_prompts.override_domain_prompt import OVERRIDE_DOMAIN_PROMPT
from prompt_templates.domain_prompts.general_prompt import GENERAL_PROMPT

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN → PROMPT MAPPING
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_PROMPT_MAP = {
    "cap_api": CAP_API_PROMPT,
    "benefits_api": BENEFITS_API_PROMPT,
    "claim_history_search": CLAIM_HISTORY_SEARCH_PROMPT,
    "member_domain": MEMBER_DOMAIN_PROMPT,
    "override_domain": OVERRIDE_DOMAIN_PROMPT,
    "general": GENERAL_PROMPT,
}

# Intent descriptions for candidate-focused prompts
INTENT_DESC = {
    # cap_api
    "claim_status": "General claim status, adjudication outcome, paid/rejected/pending",
    "multi_claim_summary": "Summary of ALL/MULTIPLE claims for a member",
    "pharmacy_info": "Dispensing pharmacy name, location, NCPDP for ONE claim",
    "prescriber_info": "Prescribing physician/doctor name, NPI for ONE claim",
    "pricing_info": "Copay, ingredient cost, dispensing fee, patient pay for ONE claim",
    "reimbursement_info": "Amount paid TO pharmacy, reimbursement for ONE claim",
    "rejection_reasons": "Rejection codes, failed edits, denial reasons for ONE claim",
    "settlement_info": "Settlement codes, pharmacy response codes for ONE claim",
    "rx_details": "RX number, fill number, quantity, days supply for ONE claim",
    "reversal_info": "Claim reversal, R&R, manual adjustments",
    "cob_info": "Coordination of Benefits, other insurance for ONE claim",
    "generic_availability": "Generic alternatives, therapeutic equivalents",
    "daw_info": "DAW status, brand vs generic requirement",
    "government_claim_type": "Medicare/Medicaid claim type classification",
    "mail_order_info": "Mail order/home delivery prescription details",
    "medicare_part_d": "Medicare Part D pricing, PDE, MEDD for a CLAIM",
    "network_info": "Pharmacy network details for ONE claim",
    "prior_auth_info": "Prior authorization status for ONE specific claim",
    # benefits_api
    "approval_info": "Claim approval logic, plan overrides, TF, BPG, Smart PA",
    "audit_info": "Audit trail, change history, timestamps, add/change dates",
    "beneficiary_info": "Member benefit phase, coverage tier, accumulations for a claim",
    "plan_summary": "Benefit plan overview, active plan snapshot",
    "plan_history": "Plan change log, revision history, amendments",
    "plan_finder": "Search/find available benefit plans",
    # claim_history_search
    "compound_info": "Compound medication, MIC breakdown, ingredient costs",
    "date_range_claims": "Claims within date range, deductible/accumulation history",
    "drug_info": "Drug name, NDC, GPI, therapeutic class, formulary status",
    "drug_interaction_info": "DUR edits, drug interaction alerts",
    "fill_date_info": "Date prescription was filled, service date",
    "Refills": "Search claims by refill count, refill history",
    "DaysSupply": "Filter claims by days supply duration",
    "PriorAuth": "Search claims that required prior authorization",
    "Diagnosis": "Filter claims by ICD-10 diagnosis code",
    "Settlement": "Search/filter claims by settlement code NUMBER",
    "PharmType": "Filter claims by pharmacy type (retail, mail, specialty)",
    "Plan": "Filter claims by insurance plan code",
    "Pharmacy": "Search claims from a specific pharmacy name/store",
    "Prescriber": "Search claims by prescriber name/NPI across history",
    "Pricing": "Search drug cost/pricing across MULTIPLE claims",
    "Status": "Filter/list claims by status (paid, rejected, pending)",
    "RejectCode": "Search claims by NCPDP rejection code number",
    "DrugLast": "When was a drug last dispensed for a member",
    "Month": "Filter claims by calendar month",
    "ClaimNum": "Look up a specific claim by claim number",
    "NDC": "Search claims by NDC number",
    "Manufacturer": "Filter claims by drug manufacturer",
    "Generic": "Filter for generic drug claims only",
    "Brand": "Filter for brand name drug claims only",
    # general
    "greeting": "Hello, hi, welcome, casual greeting",
    "help": "How to use system, filing guidance, capabilities",
    "out_of_scope": "Unrelated to pharmacy — weather, sports, recipes",
    # member_domain
    "member_coverage": "Member coverage eligibility windows, active status, enrollment dates",
    "member_hierarchy": "Client/CAG hierarchy, organizational structure",
    "benefit_reset_date": "Benefit year reset date, accumulator reset",
    "family_type": "Individual vs family plan classification",
    "family_members": "List family members, dependents on same plan",
    "alternate_insurance": "Other/secondary insurance on file for member",
    "medicare_coverage": "Medicare Part D enrollment status for a MEMBER",
    "lics_status": "Low Income Subsidy (LICS/LIS) status",
    "stcob_linkage": "Short-term COB linkage, STCOB member links",
    "cvs_id_lookup": "CVS ID for the member",
    "related_cagm": "Related CAGMs by CVS ID or family ID",
    "alternate_ids": "All alternate IDs on file for the member",
    # override_domain
    "pa_summary": "Prior authorization summary, key fields overview",
    "pa_override_reject": "Will PA override reject codes 75/70/76",
    "pa_field_help": "What does a specific PA field do (documentation)",
    "pa_copay_pricing": "PA copay override impact on pricing",
    "pa_drug_coverage": "Drugs covered by this PA (GPI/NDC lists)",
    "pa_claim_usage": "How many claims used this PA, utilization count",
}


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-DOMAIN DISAMBIGUATION PROMPT
# ═══════════════════════════════════════════════════════════════════════════

CROSS_DOMAIN_DISAMBIGUATION = """
## CROSS-DOMAIN DISAMBIGUATION RULES

When candidates span multiple domains, use these rules to pick the right one:

### Single-Claim (cap_api) vs Search (claim_history_search)
- Specific claim number + "details for claim X" → cap_api
- "Show all claims...", "Filter claims by...", "Which claims..." → claim_history_search
- No claim number + search/filter language → claim_history_search

### cap_api vs benefits_api
- "Is claim paid/rejected?" → claim_status (cap_api)
- "What overrides/TF/BPG approved it?" → approval_info (benefits_api)
- "When was claim created/modified?" → audit_info (benefits_api)
- "Was claim reversed/R&R'd?" → reversal_info (cap_api)
- "Accumulation overrides" → beneficiary_info (benefits_api)

### cap_api vs member_domain
- "PA for claim X" → prior_auth_info (cap_api)
- "Does member have Part D coverage?" → medicare_coverage (member_domain)
- "Part D pricing for claim X" → medicare_part_d (cap_api)
- "Other insurance for member" → alternate_insurance (member_domain)
- "COB for claim X" → cob_info (cap_api)

### cap_api vs override_domain
- "PA status for claim X" → prior_auth_info (cap_api)
- "PA summary/overview" → pa_summary (override_domain)
- "Will PA override reject 75?" → pa_override_reject (override_domain)
- "Why was claim rejected?" → rejection_reasons (cap_api)

### claim_history_search vs override_domain
- "Claims with reject code 75" → RejectCode (claim_history_search)
- "Does PA handle reject 75?" → pa_override_reject (override_domain)
- "Claims that required PA" → PriorAuth (claim_history_search)
- "How many claims used PA?" → pa_claim_usage (override_domain)

### Matching caps: settlement_info vs Settlement, pricing_info vs Pricing, etc.
- settlement_info = settlement for ONE claim | Settlement = SEARCH by settlement code
- pricing_info = pricing for ONE claim | Pricing = drug cost across MANY claims
- pharmacy_info = pharmacy for ONE claim | Pharmacy = claims FROM a pharmacy
- prescriber_info = prescriber for ONE claim | Prescriber = claims BY a prescriber
- prior_auth_info = PA for ONE claim | PriorAuth = SEARCH claims needing PA
- rejection_reasons = WHY one claim rejected | RejectCode = SEARCH by reject code
- claim_status = status of ONE claim | Status = FILTER claims by status

### GOLDEN RULE:
- "for claim [NUMBER]" or "of claim [NUMBER]" → usually cap_api (details about ONE claim)
- "Show/list/filter/search claims..." → usually claim_history_search (across many claims)
"""


def get_domain_prompt(domain: str) -> str:
    """Get the detailed prompt for a specific domain."""
    return DOMAIN_PROMPT_MAP.get(domain, GENERAL_PROMPT)


def _build_system_prompt(
    candidate_intents: List[str],
    candidate_domains: set,
    top5_intents: List[Tuple[str, float]],
) -> str:
    """Build a comprehensive system prompt combining base + domain-specific instructions.

    Strategy:
      - Always include base components (entities, acronyms, output format)
      - Include ALL domain prompts for domains represented in the top-5
      - If multiple domains, include cross-domain disambiguation
      - Add a focused CANDIDATE section to constrain the LLM's output
    """
    parts = []

    # ── Header ──────────────────────────────────────────────────────────
    parts.append("""# PBM Intent Classification — Expert Fallback System

You are an expert intent classification system for a Pharmacy Benefit Manager (PBM) platform.
The primary classifier was UNCERTAIN about this query. Your task is to accurately re-classify
the user's query into exactly ONE intent.

IMPORTANT: You MUST choose from the candidate intents listed below OR, if none fits,
choose the most appropriate intent from any domain. Prefer candidates when possible.
""")

    # ── Candidate Focus ──────────────────────────────────────────────────
    parts.append("## CANDIDATE INTENTS (ranked by primary classifier confidence):\n")
    for name, prob in top5_intents:
        domain = INTENT_TO_DOMAIN.get(name, "unknown")
        desc = INTENT_DESC.get(name, name)
        parts.append(f"  {prob:>5.1%}  [{domain}] {name}: {desc}")
    parts.append("")

    # ── Domain-Specific Prompts ──────────────────────────────────────────
    for domain in sorted(candidate_domains):
        prompt = DOMAIN_PROMPT_MAP.get(domain)
        if prompt:
            parts.append(f"\n{'='*70}")
            parts.append(f"  DOMAIN: {domain.upper()}")
            parts.append(f"{'='*70}")
            parts.append(prompt)

    # ── Cross-Domain Disambiguation ──────────────────────────────────────
    if len(candidate_domains) > 1:
        parts.append(CROSS_DOMAIN_DISAMBIGUATION)

    # ── Shared Components ────────────────────────────────────────────────
    parts.append(PBM_ACRONYM_GLOSSARY)
    parts.append(ENTITY_EXTRACTION_RULES)
    parts.append(CONVERSATION_HISTORY_RULES)
    parts.append(OUTPUT_FORMAT)

    # ── Final Rules ──────────────────────────────────────────────────────
    parts.append("""
## FINAL CLASSIFICATION RULES
1. Read the query carefully. Identify key phrases and entities.
2. Match against the candidate intents FIRST. Most of the time, the correct answer is one of the top-5.
3. If the query clearly matches a non-candidate intent, use that instead.
4. Apply the decision tree for each relevant domain.
5. Use disambiguation rules for confusing pairs.
6. Extract ALL entities mentioned in the query.
7. Set confidence based on clarity: 0.85+ (clear match), 0.7-0.85 (good match), 0.5-0.7 (uncertain).
8. Return ONLY the JSON object. No other text.
""")

    return "\n".join(parts)


def _build_user_prompt(
    query: str,
    ensemble_intent: Optional[str] = None,
    ensemble_confidence: float = 0.0,
    conversation_history: Optional[List[Dict]] = None,
) -> str:
    """Build the user prompt with query and optional conversation context."""
    parts = []

    # Include conversation history if available
    if conversation_history and len(conversation_history) > 0:
        parts.append("Conversation History:")
        for msg in conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                parts.append(f"{role.capitalize()}: {content}")
        parts.append("")

    # Add current user message
    parts.append(f"Current User message: {query}")
    parts.append("")

    # Add ensemble context as advisory
    if ensemble_intent:
        parts.append(
            f"Note: The primary embedding classifier suggested '{ensemble_intent}' "
            f"with {ensemble_confidence:.0%} confidence, but was uncertain. "
            f"Please re-evaluate carefully using the detailed domain knowledge above."
        )

    return "\n".join(parts)


def _parse_llm_response(
    response_text: str,
    candidate_intents: List[str],
) -> Dict[str, Any]:
    """Parse and validate the LLM's JSON response.

    Returns a dict with: intent, confidence, domain, entities, reasoning
    Falls back to the first candidate if parsing fails.
    """
    # Clean response
    text = response_text.strip()

    # Remove markdown code blocks if present
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Try to extract JSON
    json_match = re.search(r'\{[^{}]*"intent"[^{}]*\}', text, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group(0))
    else:
        result = json.loads(text)

    predicted = result.get("intent", "")
    confidence = float(result.get("confidence", 0.85))
    domain = result.get("domain", INTENT_TO_DOMAIN.get(predicted, "unknown"))
    entities = result.get("entities", {})
    reasoning = result.get("reasoning", "")

    # Filter out None entities
    if entities:
        entities = {k: v for k, v in entities.items() if v is not None}

    # Validate: prefer exact match in candidates
    for c in candidate_intents:
        if c.lower() == predicted.lower():
            return {
                "intent": c,
                "confidence": confidence,
                "domain": INTENT_TO_DOMAIN.get(c, domain),
                "entities": entities,
                "reasoning": reasoning,
            }

    # Partial match
    for c in candidate_intents:
        if c.lower() in predicted.lower() or predicted.lower() in c.lower():
            return {
                "intent": c,
                "confidence": confidence,
                "domain": INTENT_TO_DOMAIN.get(c, domain),
                "entities": entities,
                "reasoning": reasoning,
            }

    # Not in candidates but valid intent
    if predicted in INTENT_TO_DOMAIN:
        return {
            "intent": predicted,
            "confidence": confidence,
            "domain": INTENT_TO_DOMAIN[predicted],
            "entities": entities,
            "reasoning": reasoning,
        }

    # Fallback
    logger.warning(f"LLM returned unknown intent '{predicted}', using first candidate")
    fallback = candidate_intents[0] if candidate_intents else "unknown"
    return {
        "intent": fallback,
        "confidence": 0.5,
        "domain": INTENT_TO_DOMAIN.get(fallback, "unknown"),
        "entities": entities,
        "reasoning": f"Fallback: LLM returned '{predicted}' which is not recognized",
    }


def llm_fallback_classify(
    query: str,
    top5_intents: List[Tuple[str, float]],
    ensemble_intent: Optional[str] = None,
    ensemble_confidence: float = 0.0,
    conversation_history: Optional[List[Dict]] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    max_output_tokens: int = 200,
) -> Dict[str, Any]:
    """
    Domain-aware LLM fallback classifier.

    This is the MAIN entry point. Call this when the embedding classifier
    confidence is below the threshold.

    Args:
        query: User's original natural language query (NOT normalized — LLM needs entities)
        top5_intents: List of (intent_name, probability) from the ensemble, ordered by probability
        ensemble_intent: The ensemble's top prediction
        ensemble_confidence: The ensemble's confidence score
        conversation_history: Optional list of prior conversation turns
        project_id: GCP project ID (defaults to env var)
        location: GCP location (defaults to env var)
        model: LLM model to use
        temperature: LLM temperature (0.0 = deterministic)
        max_output_tokens: Max tokens for response

    Returns:
        Dict with keys: intent, confidence, domain, entities, reasoning, source
    """
    from google import genai
    from google.genai import types

    # Defaults
    project = project_id or os.getenv("PROJECT_ID", "pbm-poc-coderev-genai-poc")
    loc = location or os.getenv("LOCATION", "us-central1")

    # Determine candidate domains and intents
    candidate_domains = get_candidate_domains(top5_intents)
    candidate_intents = [name for name, _ in top5_intents]

    # Build prompts
    system_prompt = _build_system_prompt(candidate_intents, candidate_domains, top5_intents)
    user_prompt = _build_user_prompt(
        query, ensemble_intent, ensemble_confidence, conversation_history
    )

    logger.info(f"LLM Fallback: query='{query[:80]}...', domains={candidate_domains}")
    logger.info(f"  Candidates: {candidate_intents}")

    try:
        client = genai.Client(vertexai=True, project=project, location=loc)

        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_prompt,
            ),
        )

        response_text = response.text.strip()
        logger.info(f"  LLM response: {response_text[:200]}")

        result = _parse_llm_response(response_text, candidate_intents)
        result["source"] = "llm_fallback"
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"LLM JSON parse failed: {e}")
        fallback = candidate_intents[0] if candidate_intents else "unknown"
        return {
            "intent": fallback,
            "confidence": ensemble_confidence,
            "domain": INTENT_TO_DOMAIN.get(fallback, "unknown"),
            "entities": {},
            "reasoning": f"JSON parse failed: {e}",
            "source": "llm_fallback_error",
        }
    except Exception as e:
        logger.error(f"LLM fallback failed: {e}")
        fallback = candidate_intents[0] if candidate_intents else "unknown"
        return {
            "intent": fallback,
            "confidence": ensemble_confidence,
            "domain": INTENT_TO_DOMAIN.get(fallback, "unknown"),
            "entities": {},
            "reasoning": f"LLM call failed: {e}",
            "source": "llm_fallback_error",
        }


# ═══════════════════════════════════════════════════════════════════════════
# ASYNC VERSION — for use in production nodes (llm_judge_node)
# ═══════════════════════════════════════════════════════════════════════════

async def llm_fallback_classify_async(
    query: str,
    top5_intents: List[Tuple[str, float]],
    ensemble_intent: Optional[str] = None,
    ensemble_confidence: float = 0.0,
    conversation_history: Optional[List[Dict]] = None,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    max_output_tokens: int = 200,
) -> Dict[str, Any]:
    """
    Async version of llm_fallback_classify for use in async production nodes.
    Runs the sync Gemini call in a thread pool executor.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: llm_fallback_classify(
            query=query,
            top5_intents=top5_intents,
            ensemble_intent=ensemble_intent,
            ensemble_confidence=ensemble_confidence,
            conversation_history=conversation_history,
            project_id=project_id,
            location=location,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
