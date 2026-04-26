"""
Base prompt utilities shared across all domain-specific LLM fallback prompts.

Provides:
  - Common entity extraction rules
  - Shared PBM acronym glossary
  - Output format specification
  - Domain routing logic
"""

# ═══════════════════════════════════════════════════════════════════════════
# INTENT → DOMAIN MAPPING (single source of truth)
# ═══════════════════════════════════════════════════════════════════════════

INTENT_TO_DOMAIN = {
    # ── cap_api (single-claim operations) ──────────────────────────────────
    "claim_status": "cap_api", "multi_claim_summary": "cap_api",
    "pharmacy_info": "cap_api", "prescriber_info": "cap_api",
    "pricing_info": "cap_api", "reimbursement_info": "cap_api",
    "rejection_reasons": "cap_api", "settlement_info": "cap_api",
    "rx_details": "cap_api", "reversal_info": "cap_api",
    "cob_info": "cap_api", "generic_availability": "cap_api",
    "daw_info": "cap_api", "government_claim_type": "cap_api",
    "mail_order_info": "cap_api", "medicare_part_d": "cap_api",
    "network_info": "cap_api", "prior_auth_info": "cap_api",
    # ── benefits_api ───────────────────────────────────────────────────────
    "approval_info": "benefits_api", "audit_info": "benefits_api",
    "beneficiary_info": "benefits_api", "plan_summary": "benefits_api",
    "plan_history": "benefits_api", "plan_finder": "benefits_api",
    # ── claim_history_search ───────────────────────────────────────────────
    "compound_info": "claim_history_search", "date_range_claims": "claim_history_search",
    "drug_info": "claim_history_search", "drug_interaction_info": "claim_history_search",
    "fill_date_info": "claim_history_search",
    "Refills": "claim_history_search", "DaysSupply": "claim_history_search",
    "PriorAuth": "claim_history_search", "Diagnosis": "claim_history_search",
    "Settlement": "claim_history_search", "PharmType": "claim_history_search",
    "Plan": "claim_history_search", "Pharmacy": "claim_history_search",
    "Prescriber": "claim_history_search", "Pricing": "claim_history_search",
    "Status": "claim_history_search", "RejectCode": "claim_history_search",
    "DrugLast": "claim_history_search", "Month": "claim_history_search",
    "ClaimNum": "claim_history_search", "NDC": "claim_history_search",
    "Manufacturer": "claim_history_search", "Generic": "claim_history_search",
    "Brand": "claim_history_search",
    # ── general ────────────────────────────────────────────────────────────
    "greeting": "general", "help": "general", "out_of_scope": "general",
    # ── member_domain ──────────────────────────────────────────────────────
    "member_coverage": "member_domain", "member_hierarchy": "member_domain",
    "benefit_reset_date": "member_domain", "family_type": "member_domain",
    "family_members": "member_domain", "alternate_insurance": "member_domain",
    "medicare_coverage": "member_domain", "lics_status": "member_domain",
    "stcob_linkage": "member_domain", "cvs_id_lookup": "member_domain",
    "related_cagm": "member_domain", "alternate_ids": "member_domain",
    # ── override_domain ────────────────────────────────────────────────────
    "pa_summary": "override_domain", "pa_override_reject": "override_domain",
    "pa_field_help": "override_domain", "pa_copay_pricing": "override_domain",
    "pa_drug_coverage": "override_domain", "pa_claim_usage": "override_domain",
}

# ═══════════════════════════════════════════════════════════════════════════
# SHARED COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

ENTITY_EXTRACTION_RULES = """
## Entity Extraction Rules
- **claim_number**: A 12-18 digit numeric string identifying a specific claim. Extract AS-IS.
- **sequence_number**: A 1-3 digit string (e.g., "001", "002"). Extract AS-IS.
- **member_id**: Member identifier string. Extract AS-IS.
- **drug_name**: Name of medication/drug mentioned. Extract AS-IS.
- **ndc**: National Drug Code (11-digit). Extract AS-IS.
- **npi**: National Provider Identifier (10-digit). Extract AS-IS.
- **date_range**: Any date or date range mentioned.
- **pharmacy_name**: Name of pharmacy mentioned.
- **prescriber_name**: Name of prescriber/doctor mentioned.
- **reject_code**: NCPDP rejection code number.
- **settlement_code**: Settlement/response code number.
- **pa_number**: Prior authorization number.

**Masked Token Handling:**
When you see tokens like [CLAIM_ID_XXXXXXXX] or [MEMBER_ID_XXXXXXXX], extract them AS-IS as entity values.
NEVER replace masked tokens with example values from this prompt.
"""

CONVERSATION_HISTORY_RULES = """
## Conversation History Handling
When conversation history is provided:
1. Extract entities (claim numbers, drug names, member IDs) from history if missing in current query
2. Resolve references like "it", "that claim", "the drug", "this member" using history context
3. Use history to understand follow-up questions and maintain conversational continuity
4. If the current query is a follow-up (e.g., "What about the pricing?" after a claim status query),
   carry forward the claim_number and sequence_number from the prior turn
"""

PBM_ACRONYM_GLOSSARY = """
## PBM Acronym Glossary
Expand these acronyms when encountered in queries:

**Claims:** COB (Coordination Of Benefits), COBA (Coordination of Benefit Agreement), EOB (Explanation Of Benefits), EOP (Explanation of Payment)
**Drug/Rx:** DUR (Drug Utilization Review), DAW (Dispense as Written), NDC (National Drug Code), NCPDP (National Council of Prescription Drug Programs), Rx (Prescription), MIC (Multi-Ingredient Compound), GPI (Generic Product Identifier)
**Coverage:** PA (Prior Authorization), BEN (Benefits), BV (Benefit Verification), GAP (Coverage Gap), LICS/LIS (Low Income Cost Sharing/Low Income Subsidy), STCOB (Single Transaction Coordination of Benefits), TF (Transition Fill)
**Network:** NPI (National Provider Identifier), NABP (National Association of Boards of Pharmacy), MO (Mail Order)
**Pricing:** AWP (Average Wholesale Price), WAC (Wholesale Acquisition Cost), MAC (Maximum Allowable Cost), OOP (Out of Pocket)
**Organization:** PBM (Pharmacy Benefit Manager), CMS (Centers for Medicare & Medicaid Services), CAG (Client Account Group), CAGM (Client Account Group Member), BPG (Benefit Plan Group), LOE (Level of Evidence)
**Medicare:** MEDD (Medicare Part D), PDE (Prescription Drug Event), LTC (Long Term Care)
**Override/PA:** EPA (Electronic Prior Authorization), QL (Quantity Limit), ST (Step Therapy)

**Process:** Identify acronyms → expand → re-interpret query → classify based on expanded meaning.
"""

OUTPUT_FORMAT = """
## Output Format (JSON only — no markdown, no explanation outside JSON)
```json
{
  "intent": "<exact_intent_name>",
  "confidence": <0.0-1.0>,
  "domain": "<domain_name>",
  "entities": {
    "claim_number": "<value_or_null>",
    "sequence_number": "<value_or_null>",
    "member_id": "<value_or_null>",
    "drug_name": "<value_or_null>",
    "pharmacy_name": "<value_or_null>",
    "prescriber_name": "<value_or_null>",
    "npi": "<value_or_null>",
    "ndc": "<value_or_null>",
    "reject_code": "<value_or_null>",
    "settlement_code": "<value_or_null>",
    "date_range": "<value_or_null>",
    "pa_number": "<value_or_null>"
  },
  "reasoning": "<one-sentence justification>"
}
```
Only include entity keys that have non-null values. Omit null-valued keys.
"""

# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN ROUTING
# ═══════════════════════════════════════════════════════════════════════════

def get_candidate_domains(top5_intents):
    """Given top-5 intent predictions from the ensemble, determine which
    domain(s) the candidates span. Returns a set of domain names."""
    domains = set()
    for intent_name, _prob in top5_intents:
        domain = INTENT_TO_DOMAIN.get(intent_name, "unknown")
        domains.add(domain)
    return domains


def select_domain_prompt(top5_intents):
    """Select the most appropriate domain-specific prompt based on
    the ensemble's top-5 candidates.

    Strategy:
      - If all top-5 are in one domain → use that domain's prompt
      - If mixed → use a multi-domain prompt that covers all candidate domains
      - Always include the candidate intents in the prompt for focus
    """
    domains = get_candidate_domains(top5_intents)
    candidate_intents = [name for name, _ in top5_intents]
    return domains, candidate_intents
