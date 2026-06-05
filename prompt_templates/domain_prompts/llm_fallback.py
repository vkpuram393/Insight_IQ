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
    # cap_api — ONE specific claim (requires claim number + sequence)
    "claim_status": "Full summary/adjudication details for ONE specific claim (by claim# + seq) — paid/rejected/pending and all processing fields. NOT a history search.",
    "multi_claim_summary": "Aggregate view of ALL claims for a member — total claim count, overall spend, utilization overview, most-frequent drugs. No specific claim# or date filter required.",
    "pharmacy_info": "Dispensing PHARMACY NAME, address, NCPDP store ID/number for ONE claim — NOT the dispensing channel or pharmacy type (mail vs retail).",
    "prescriber_info": "Prescribing physician/doctor NAME and NPI for ONE specific claim",
    "pricing_info": "Copay, total drug cost, dispensing fee, plan pay, patient-pay AMOUNTS for ONE claim — NOT compound ingredient-level breakdown.",
    "reimbursement_info": "Amount paid TO the pharmacy (plan reimbursement) for ONE claim",
    "rejection_reasons": "NCPDP rejection codes, failed edits, denial reasons, override options to RESOLVE denial for ONE rejected claim",
    "settlement_info": "Settlement response codes and pharmacy transaction codes for ONE claim",
    "rx_details": "RX number, fill number, quantity dispensed, days supply for ONE claim",
    "reversal_info": "Claim reversal, R&R transaction, manual adjustments to ONE claim",
    "cob_info": "Coordination of Benefits (COB), other/secondary insurance for ONE claim",
    "generic_availability": "Whether a GENERIC EQUIVALENT EXISTS for the drug on ONE claim — BPG/generic interchange, brand dispensed when generic on formulary, generic availability flags",
    "daw_info": "DAW (dispense-as-written) CODE ONLY for ONE claim — e.g. DAW 1=brand required, DAW 0=generic allowed. NOT about generic availability.",
    "government_claim_type": "Medicare/Medicaid government claim type classification for ONE claim",
    "mail_order_info": "Dispensing CHANNEL TYPE (mail-order/retail/specialty) and NCPDP PHARMACY TYPE for ONE claim — NOT the pharmacy name or store ID.",
    "medicare_part_d": "Medicare Part D pricing fields for ONE claim: PDE, MEDD, TrOOP amounts, coverage phase (gap/catastrophic) for a specific claim#",
    "network_info": "Pharmacy network type (MAC, specialty network) for ONE claim",
    "prior_auth_info": "PA TYPE applied to ONE claim: Smart PA / Member PA / Plan PA — PA status (active/expired/pending), PA number, auth outcome",
    # benefits_api
    "approval_info": "WHY a claim was approved via plan-level override: Transition Fill (TF) logic, BPG approval, formulary exception. NOT for PA status — use prior_auth_info for PA questions.",
    "audit_info": "Audit trail, creation/modification timestamps, add/change dates for ONE claim",
    "beneficiary_info": "Member benefit-phase ACCUMULATION AMOUNTS (deductible, OOP, TrOOP running totals) and coverage tier for ONE claim — NOT the coverage phase of a claim",
    "plan_summary": "Benefit plan overview, active plan snapshot",
    "plan_history": "Plan change log, revision history, amendments",
    "plan_finder": "Search/find available benefit plans",
    # claim_history_search — SEARCH/FILTER across MANY claims
    "compound_info": "Compound drug ingredient-level MIC breakdown for ONE compound claim — funded vs unfunded ingredient costs, individual ingredient amounts. NOT a pharmacy type filter.",
    "date_range_claims": "Claims filtered by an EXPLICIT START-to-END date window (e.g. 01/01/2024 to 06/30/2024, Q3-Q4, April to June, benefit year period, PDE reconciliation period) — any claims in that date range",
    "DateRange": "Rolling RECENT date window — last N days/weeks/months (no explicit from/to dates specified)",
    "drug_info": "Drug NAME, NDC, GPI, therapeutic class, formulary status for a SPECIFIC CLAIM",
    "DrugList": "Complete medication list for a member, or filter by drug class/GPI across all claims",
    "drug_interaction_info": "DUR edits, drug-drug interaction alerts, clinical screening results, DUR override status for ONE claim — includes 'were DUR edits overridden?'",
    "fill_date_info": "Date prescription was filled/dispensed, service date, dispensing date for ONE claim",
    "Refills": "Search claims by refill count, refill history, remaining refills, or prescriptions overdue/due for a refill",
    "DaysSupply": "Filter claims by days supply duration",
    "PriorAuth": "SEARCH member history for claims that REQUIRED PA, had Smart PA applied, PA was denied, or PA was pending at adjudication",
    "Diagnosis": "Filter claims by ICD-10 diagnosis code",
    "Settlement": "Search/filter claims by settlement response code NUMBER across history",
    "PharmType": "Filter claims by pharmacy TYPE across history — retail, mail-order, specialty, compounding pharmacy. Use for 'compounding pharmacy claims in history'.",
    "Plan": "Filter claims by insurance PLAN code — 'claims under plan HMO1' → Plan (plan filter takes priority)",
    "Pharmacy": "Search claims FROM a specific pharmacy name/store/location, or by network status (out-of-network pharmacy claims)",
    "Prescriber": "Search claims by prescriber name/NPI across history",
    "Pricing": "Drug cost/OOP/TrOOP/total spend ACROSS MULTIPLE claims — cost trend, total spend on a specific drug, year-to-date spend",
    "Status": "Filter/list claims by PROCESSING STATUS (paid, rejected, pending, reversed) — NOT for PA or refill status",
    "RejectCode": "Search/identify claims by NCPDP rejection code number",
    "DrugLast": "When was a drug last dispensed for a member",
    "Month": "Filter claims by calendar month",
    "ClaimNum": "Look up a specific claim by claim NUMBER in member HISTORY — 'find claim X in the history' → ClaimNum",
    "NDC": "Search/filter claims by NDC (National Drug Code) number — includes PDE records filtered by NDC",
    "Manufacturer": "Filter claims by drug manufacturer",
    "Generic": "Filter for generic drug claims — generic substituted for brand, list of generic fills, 'generic claims for member'",
    "Brand": "Filter for brand name drug claims — brand drug fills including when generic was available, count of brand fills",
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
    "medicare_coverage": "Medicare Part D enrollment status for a MEMBER (not a claim)",
    "lics_status": "Low Income Subsidy (LICS/LIS) status",
    "stcob_linkage": "Short-term COB linkage, STCOB member links",
    "cvs_id_lookup": "CVS ID for the member",
    "related_cagm": "Related CAGMs by CVS ID or family ID",
    "alternate_ids": "All alternate IDs on file for the member",
    "member_demographics": "Member name, DOB, gender, person code, relationship code",
    "member_contact_info": "Member email, phone, mailing/postal address",
    "member_eligibility_copay": "Copay fields: copayBrand, copayGeneric, copay3, copay4",
    "member_transition_status": "Member transition fill status and start date",
    "member_dur_config": "DUR review key and process flag configuration",
    "member_mbi_number": "Medicare Beneficiary Identifier (MBI) number",
    "member_caretaker_info": "Caretaker name and address from Part D",
    "member_language_pref": "Member language code/preference",
    "member_discount_program": "Discount program type for the member",
    "member_override_plan": "Member override plan ID from eligibility",
    # override_domain
    "pa_summary": "Prior authorization summary, key fields overview",
    "pa_override_reject": "Will PA override reject codes 75/70/76",
    "pa_field_help": "What does a specific PA field do (documentation)",
    "pa_copay_pricing": "PA copay override impact on pricing",
    "pa_drug_coverage": "Drugs covered by this PA (GPI/NDC lists)",
    "pa_claim_usage": "How many claims used this PA, utilization count",
    "pa_reason_code": "PA reason code (U1, LC, OD, OA, US, U3)",
    "pa_effective_dates": "PA effective begin/end dates, expiration",
    "pa_agent_code": "Agent/source code on PA (A, C, 3, H)",
    "pa_ignore_status": "Ignore status code (Y, P, 3)",
    "pa_specialty_rx_override": "Specialty Rx reject override indicator",
    "pa_clinical_admin_code": "Clinical administration code (A, C, blank)",
    "pa_transform_care": "Transform care type on PA",
    "pa_follow_me_logic": "Follow me logic indicator on PA",
    "pa_drug_type_indicator": "Authorized drug type (G=GPI, N=NDC)",
    "pa_modification_history": "PA last modified date/time",
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
- "What edits FAILED on this claim?" → rejection_reasons (cap_api), NOT approval_info
- "Is this a new fill or continuation/refill fill?" → rx_details (cap_api), NOT approval_info
- "Is there a Smart PA attached to this claim?" → prior_auth_info (cap_api), NOT approval_info
- "Are there any claims with a TF tag?" → Status (claim_history_search), NOT approval_info
- approval_info is ONLY for: WHY was this ONE claim approved (override logic, TF logic, BPG)

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
- pricing_info = pricing for ONE claim | Pricing = drug cost/OOP/TrOOP across MANY claims
- pharmacy_info = pharmacy NAME/ID for ONE claim | Pharmacy = claims FROM a specific pharmacy
- prescriber_info = prescriber for ONE claim | Prescriber = claims BY a prescriber
- prior_auth_info = PA for ONE claim | PriorAuth = SEARCH claims needing PA
- rejection_reasons = WHY one claim rejected | RejectCode = SEARCH claims by reject code/reason
- claim_status = full details of ONE claim | Status = FILTER claims by processing status
- mail_order_info = dispensing CHANNEL TYPE for ONE claim (mail order vs retail vs specialty) | PharmType = FILTER claims by pharmacy TYPE across history
- daw_info = DAW CODE for ONE claim | Generic = FILTER all claims where generic was dispensed
- DateRange = rolling date window (last N days, no explicit from/to) | date_range_claims = explicit start-to-end date window
- DrugList = complete medication list for a member | drug_info = drug details for a specific claim

### GOLDEN RULE:
- "for claim [NUMBER]" or "of claim [NUMBER]" → usually cap_api (details about ONE claim)
- "Show/list/filter/search claims..." → usually claim_history_search (across many claims)

## ADDITIONAL DISAMBIGUATION — OBSERVED HIGH-FREQUENCY CONFUSION PAIRS

### prior_auth_info (cap_api) vs approval_info (benefits_api) [CRITICAL — 4 misclassifications]
- "Smart PA applied on claim X?" → prior_auth_info (PA TYPE, not WHY approved)
- "Was Member PA or Plan PA used on claim X?" → prior_auth_info (PA type identification)
- "Prior authorization status for claim X" → prior_auth_info
- "PA currently active/expired for claim X?" → prior_auth_info
- "Retrieve prior auth status and approval details for claim X" → prior_auth_info
- "Did Smart PA override govern the adjudication?" → prior_auth_info (which PA was applied)
- "Why was claim X approved as TF/BPG?" → approval_info (plan-level override reason)
- "Was Transition Fill applied?" → approval_info (TF is a plan override, not PA)
- RULE: prior_auth_info = WHAT PA TYPE was applied and its status. approval_info = WHY approved via plan logic (TF/BPG). NEVER use approval_info for "was PA applied" or "PA status" questions.

### multi_claim_summary (cap_api) vs Pricing/date_range_claims/Status [CRITICAL — 9 misclassifications]
- "Summary of all prescriptions filled by member X" → multi_claim_summary
- "Pull every claim on file for member X and summarize total cost" → multi_claim_summary
- "Total number of claims and overall spend for this member" → multi_claim_summary
- "How many claims did member X have this quarter / year?" → multi_claim_summary
- "Aggregate all claims for this member" → multi_claim_summary
- "Member's entire claim history — all claims" → multi_claim_summary
- "Total claim volume / total pharmacy claims for member" → multi_claim_summary
- "All claims summary / prescription utilization pattern" → multi_claim_summary
- "Total spent on drug X across all fills" / "OOP cost trend" → Pricing (drug-specific cost)
- "Claims in date range X to Y" → date_range_claims (date filter present)
- "All rejected claims" → Status (status filter present)
- RULE: multi_claim_summary = aggregate/holistic view of ALL claims for a member with NO specific filter (no date range, no drug name, no status). If a specific filter is applied, use the filter-specific intent.

### mail_order_info vs pharmacy_info [3 misclassifications]
- "Dispensing channel (mail/retail/specialty) for claim X" → mail_order_info
- "NCPDP pharmacy TYPE for claim X" → mail_order_info (TYPE = channel category)
- "Retail, mail, or specialty — what channel is claim X?" → mail_order_info
- "Pharmacy NAME / address / store for claim X" → pharmacy_info
- "NCPDP store ID/number for claim X" → pharmacy_info (store number is identity, not type)
- RULE: mail_order_info = HOW the drug was delivered (channel/type). pharmacy_info = WHERE/WHO dispensed it (name, address, ID).

### compound_info (cap_api single-claim) vs pricing_info [3 misclassifications]
- "MIC ingredient costs on claim X" → compound_info (ingredient-level, not total claim cost)
- "Funded vs unfunded ingredient costs on claim X" → compound_info
- "Which ingredients were not covered by benefit on claim X?" → compound_info
- "Copay / patient pay / total cost on claim X" → pricing_info (aggregate claim amounts)
- "Dispensing fee / ingredient cost (total) on claim X" → pricing_info
- RULE: compound_info = per-ingredient breakdown of a compound drug. pricing_info = aggregate cost fields for any claim. MIC/ingredient/funded keywords always → compound_info.

### PharmType vs compound_info [3 misclassifications]
- "Compounding pharmacy claims in member's history" → PharmType (compounding is a PHARMACY TYPE)
- "Are there any compound medication claims in this member's history?" → PharmType (history search)
- "Member's compounding claims history, any MIC breakdowns?" → PharmType (primary = pharmacy type search)
- "Ingredient breakdown / MIC costs for claim X seq Y" → compound_info (specific claim details)
- RULE: If asking about compounding in the context of member HISTORY or claim SEARCH → PharmType. If asking about compound drug INGREDIENTS for a specific claim → compound_info.

### generic_availability vs daw_info [3 misclassifications]
- "Is there a generic equivalent for the drug on claim X?" → generic_availability
- "Brand dispensed when generic on formulary for claim X?" → generic_availability
- "DAW code and generic availability for claim X" → generic_availability (availability check wins)
- "Check DAW and generic availability flags" → generic_availability (availability is the focus)
- "BPG or generic interchange for drug on claim X?" → generic_availability
- "What is the DAW code for claim X?" → daw_info (DAW code ONLY, no availability question)
- "Was brand required (DAW) on claim X?" → daw_info
- RULE: generic_availability = IS a generic available? daw_info = WHAT is the DAW code? When both DAW and generic are mentioned, the question "is a generic available?" takes precedence → generic_availability.

### date_range_claims vs DateRange vs multi_claim_summary vs fill_date_info vs Status [5 misclassifications]
- "Claims from July 2024 through December 2024" (explicit from-to) → date_range_claims
- "All claims between 01/01/2024 and 06/30/2024" → date_range_claims
- "Claims filed between April 2025 and current date" → date_range_claims
- "Claims between Q3 and Q4 last year" → date_range_claims
- "PDE reconciliation — all claims in 01/01/2024-06/30/2024" → date_range_claims
- "Adjudicated claims between Q3 and Q4" → date_range_claims (date filter wins over 'adjudicated')
- "Claims in the last 30/90 days / last 2 months" (rolling, no from date) → DateRange
- "All claims for member / total claim count" (no date) → multi_claim_summary
- "When was claim X filled?" → fill_date_info (fill date for ONE claim)
- RULE: date_range_claims = explicit START date AND END date. DateRange = rolling last-N period. The presence of BOTH a start and end date → always date_range_claims.

### PriorAuth (claim_history_search) vs Status [2 misclassifications]
- "Retrieve claims that show Smart PA was applied" → PriorAuth
- "Show claims where prior auth was still pending at adjudication" → PriorAuth
- "Claims where PA was denied / required / applied" → PriorAuth
- "Filter claims that needed prior authorization" → PriorAuth
- "Show all rejected claims / paid claims" → Status (processing status filter)
- "Claims in paid/rejected/pending status" → Status
- RULE: PriorAuth = filter by PA involvement. Status = filter by PROCESSING status. 'Pending at adjudication' in a PA context → PriorAuth, not Status.

### Refills vs Status [1 misclassification]
- "Prescriptions showing as overdue for a refill" → Refills
- "Refill history / remaining refills / refill count" → Refills
- "Show all rejected / pending / paid claims" → Status
- RULE: 'overdue for refill' and 'refill due' always → Refills, NOT Status.

### rejection_reasons vs approval_info [1 misclassification]
- "Override options to RESOLVE the denial on claim X" → rejection_reasons (resolve = fix a rejection)
- "Rejection codes and how to override them" → rejection_reasons
- "Why was claim X approved (despite edits)?" → approval_info
- RULE: rejection_reasons covers both WHY rejected AND how to override/resolve it. approval_info is never about resolving a rejection.

### drug_interaction_info vs approval_info [1 misclassification]
- "Were DUR clinical edits overridden on claim X?" → drug_interaction_info (DUR is clinical, not plan)
- "Drug-drug interaction alert overridden?" → drug_interaction_info
- "TF or BPG override on claim X?" → approval_info (plan-level override)
- RULE: DUR/clinical edit overrides → drug_interaction_info. Plan-level overrides (TF, BPG) → approval_info.

### claim_status vs ClaimNum [2 misclassifications in BOTH directions]
- "Claim summary for claim X seq Y" → claim_status (cap_api — full adjudication details)
- "Show all details for claim X sequence Y — what happened?" → claim_status (cap_api)
- "Find claim X in the history and show its status" → ClaimNum (history context wins)
- "Look up claim X" (in search/history context) → ClaimNum
- RULE: 'in the history' language → ClaimNum. 'Full claim summary/details' without history context → claim_status.

### NDC vs medicare_part_d [1 misclassification]
- "Cross-reference PDE records against NDC 00169-4175-11" → NDC (NDC filter on history search)
- "PDE records filtered by NDC number" → NDC (search by NDC in claim history)
- "Part D pricing / MEDD / TrOOP for claim X" → medicare_part_d (Part D amounts for ONE claim)
- RULE: When NDC is the search FILTER criteria → NDC. When Part D AMOUNTS are the focus for ONE claim → medicare_part_d.

### Brand vs Generic vs daw_info [3 misclassifications]
- "Claims for a brand drug that had a generic available" → Brand (brand drug claims list)
- "Count of specialty brand drug fills for member" → Brand
- "List of all DAW code 1 overrides or generic claims for member X" → Generic (primary ask is generic claims list)
- "Which fills dispensed a generic instead of brand?" → Generic
- "DAW code for claim X" → daw_info (DAW code for ONE specific claim)
- RULE: Brand/Generic = filter claim history. daw_info = lookup DAW code for ONE claim.

### Plan vs date_range_claims [1 misclassification]
- "Claims under HMO1 for the last six months" → Plan (plan filter is the primary criteria)
- "Claims under plan code XYZ" → Plan
- "Claims from January to June (no plan specified)" → date_range_claims
- RULE: When a plan code/name is explicitly specified, Plan wins even if date context is present.

### Pharmacy vs PharmType [1 misclassification]
- "Any claims from an out-of-network pharmacy?" → Pharmacy (network status is a pharmacy attribute)
- "Claims filled at CVS/Walgreens/specific pharmacy name" → Pharmacy
- "Filter for retail / mail-order / specialty pharmacy claims" → PharmType
- "Compounding pharmacy claims in history" → PharmType
- RULE: Specific pharmacy NAME or network status (out-of-network) → Pharmacy. Channel/category filter → PharmType.

### medicare_part_d vs beneficiary_info [1 misclassification]
- "Is claim X in coverage gap phase or catastrophic phase under Part D?" → medicare_part_d
- "Coverage gap / catastrophic phase for a specific claim number" → medicare_part_d
- "Member's current benefit phase / deductible accumulation" → beneficiary_info
- "TrOOP accumulation amount for the member" → beneficiary_info
- RULE: Coverage phase for a SPECIFIC CLAIM under Part D → medicare_part_d. Member's accumulated benefit amounts → beneficiary_info.

### generic_availability vs drug_info [1 misclassification]
- "Does drug on claim X have a BPG or generic interchange?" → generic_availability
- "Generic substitution available for claim X?" → generic_availability
- "Drug name / NDC / GPI for claim X" → drug_info
- RULE: 'BPG', 'generic interchange', 'generic available?' → generic_availability, NOT drug_info.
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
5. Use disambiguation rules for confusing pairs (see CROSS-DOMAIN DISAMBIGUATION above).
6. Extract ALL entities mentioned in the query.
7. Set confidence using the calibration table below — do NOT default to 0.5 for clear matches.
8. Return ONLY the JSON object. No other text.

## CONFIDENCE CALIBRATION (use these benchmarks, do not default to 0.5)
- 0.95+: CANONICAL match — the query contains the exact trigger phrase for one intent and only that intent.
  High-confidence examples:
  * "fill date / dispensing date / when was claim X filled" -> fill_date_info at 0.95
  * "DUR edits / drug interaction alerts / clinical screening / were DUR edits overridden?" -> drug_interaction_info at 0.95
  * "compound claim / MIC costs / ingredient breakdown / funded vs unfunded ingredients" -> compound_info at 0.95
  * "claims from DATE1 to DATE2 / claims between Q3 and Q4 / between 01/01 and 06/30" -> date_range_claims at 0.95
  * "total claim count / all claims summary / aggregate all claims for member" -> multi_claim_summary at 0.95
  * "PA status / Smart PA applied / Member PA or Plan PA used / prior auth status" -> prior_auth_info at 0.95
  * "dispensing channel / NCPDP pharmacy type / mail vs retail vs specialty channel" -> mail_order_info at 0.95
  * "generic available? / brand dispensed when generic on formulary / BPG interchange" -> generic_availability at 0.95
  * "DAW code for claim X" (DAW code ONLY, no availability question) -> daw_info at 0.95
  * "claim summary / full details for claim X seq Y" -> claim_status at 0.95
  * "compounding pharmacy claims in member history" -> PharmType at 0.95
  * "claims where PA was applied / required / denied / pending" -> PriorAuth at 0.95
  * "prescriptions overdue for refill / remaining refills" -> Refills at 0.95
  * "brand drug claims / brand fill count for member" -> Brand at 0.95
  * "generic drug claims for member" -> Generic at 0.95
  * "claims under plan HMO1 / plan code" -> Plan at 0.95
  * "Part D coverage gap phase / catastrophic phase for claim X" -> medicare_part_d at 0.95
  * "TrOOP accumulation / deductible accumulation for member on claim" -> beneficiary_info at 0.95
  * "transition fill details / BPG approval / WHY claim was approved" -> approval_info at 0.95

- 0.90: STRONG match — clear intent signal, minor ambiguity with one other intent.
- 0.80-0.89: GOOD match — intent is likely correct, some overlap with another candidate.
- 0.70-0.79: REASONABLE — plausible but two intents could fit.
- 0.50-0.69: UNCERTAIN — genuinely ambiguous; use ONLY when you truly cannot distinguish.

IMPORTANT: Only use 0.5 when the query is genuinely ambiguous. For queries with clear trigger phrases listed above, always use 0.90+.
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

    # Add step-by-step analysis guidance to reduce anchor bias and improve reasoning
    parts.append(
        "Classify this query step by step:\n"
        "  Step 1 — Core action: What is the user's primary goal? "
        "(when/which/list/filter/compare/how-much/show/was)\n"
        "  Step 2 — Scope: Is this about ONE specific claim (cap_api) or "
        "SEARCHING across MULTIPLE claims (claim_history_search)?\n"
        "  Step 3 — Filter criterion: What specific attribute is being filtered/searched? "
        "(drug name, date range, reject code, status, pricing, refill, PA, prescriber, etc.)\n"
        "  Step 4 — Intent match: Which single intent from the candidates matches "
        "Steps 1-3? Apply the domain decision tree.\n"
        "  Step 5 — Confidence: How clear is the match? Apply the routing-aware scale "
        "(>= 0.88 for clear single-intent matches).\n"
    )

    # Ensemble context framed as secondary advisory — avoid anchor bias
    if ensemble_intent:
        parts.append(
            f"Advisory only (do NOT anchor on this): The primary classifier "
            f"suggested '{ensemble_intent}' ({ensemble_confidence:.0%}) but was below "
            f"the confidence threshold. Classify PURELY from the query text above "
            f"using your PBM expertise — the classifier suggestion may be wrong."
        )

    return "\n".join(parts)


def _extract_first_json_object(text: str):
    """Return the first complete JSON object in text, handling arbitrary nesting."""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


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

    # Extract the first complete JSON object (handles nested braces in entities etc.)
    json_str = _extract_first_json_object(text)
    result = json.loads(json_str if json_str is not None else text)

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
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
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
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
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
