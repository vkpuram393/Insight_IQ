"""
Multidomain Intent Detection — Configuration & Mappings
========================================================

Central registry for:
  - Intent → Domain mapping
  - Intent descriptions (for LLM fallback prompts)
  - Domain → API endpoint mapping
  - Domain friendly names

All six production domains:
  cap_api, benefits_api, claim_history_search,
  member_domain, override_domain, general
"""

from typing import Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Intent → Domain mapping
# ─────────────────────────────────────────────────────────────────────────────

INTENT_TO_DOMAIN: Dict[str, str] = {
    # ── cap_api (single-claim operations) ────────────────────────────────
    "claim_status": "cap_api",
    "multi_claim_summary": "cap_api",
    "pharmacy_info": "cap_api",
    "prescriber_info": "cap_api",
    "pricing_info": "cap_api",
    "reimbursement_info": "cap_api",
    "rejection_reasons": "cap_api",
    "settlement_info": "cap_api",
    "rx_details": "cap_api",
    "reversal_info": "cap_api",
    "cob_info": "cap_api",
    "generic_availability": "cap_api",
    "daw_info": "cap_api",
    "government_claim_type": "cap_api",
    "mail_order_info": "cap_api",
    "medicare_part_d": "cap_api",
    "network_info": "cap_api",
    "prior_auth_info": "cap_api",
    # ── benefits_api ─────────────────────────────────────────────────────
    "plan_summary": "benefits_api",
    "plan_history": "benefits_api",
    "plan_finder": "benefits_api",
    # ── cap_api additions (moved from benefits_api / claim_history_search) ─
    "approval_info": "cap_api",
    "audit_info": "cap_api",
    "beneficiary_info": "cap_api",
    # ── claim_history_search ─────────────────────────────────────────────
    "compound_info": "cap_api",
    "date_range_claims": "claim_history_search",
    "drug_info": "cap_api",
    "drug_interaction_info": "cap_api",
    "fill_date_info": "cap_api",
    "Refills": "claim_history_search",
    "DaysSupply": "claim_history_search",
    "PriorAuth": "claim_history_search",
    "Diagnosis": "claim_history_search",
    "Settlement": "claim_history_search",
    "PharmType": "claim_history_search",
    "Plan": "claim_history_search",
    "Pharmacy": "claim_history_search",
    "Prescriber": "claim_history_search",
    "Pricing": "claim_history_search",
    "Status": "claim_history_search",
    "RejectCode": "claim_history_search",
    "DrugLast": "claim_history_search",
    "Month": "claim_history_search",
    "ClaimNum": "claim_history_search",
    "NDC": "claim_history_search",
    "Manufacturer": "claim_history_search",
    "Generic": "claim_history_search",
    "Brand": "claim_history_search",
    # ── general ──────────────────────────────────────────────────────────
    "greeting": "general",
    "help": "general",
    "out_of_scope": "general",
    # ── member_domain ────────────────────────────────────────────────────
    "member_coverage": "member_domain",
    "member_hierarchy": "member_domain",
    "benefit_reset_date": "member_domain",
    "family_type": "member_domain",
    "family_members": "member_domain",
    "alternate_insurance": "member_domain",
    "medicare_coverage": "member_domain",
    "lics_status": "member_domain",
    "stcob_linkage": "member_domain",
    "cvs_id_lookup": "member_domain",
    "related_cagm": "member_domain",
    "alternate_ids": "member_domain",
    "member_demographics": "member_domain",
    "member_contact_info": "member_domain",
    "member_eligibility_copay": "member_domain",
    "member_transition_status": "member_domain",
    "member_dur_config": "member_domain",
    "member_mbi_number": "member_domain",
    "member_caretaker_info": "member_domain",
    "member_language_pref": "member_domain",
    "member_discount_program": "member_domain",
    "member_override_plan": "member_domain",
    # ── override_domain ──────────────────────────────────────────────────
    "pa_summary": "override_domain",
    "pa_override_reject": "override_domain",
    "pa_field_help": "override_domain",
    "pa_copay_pricing": "override_domain",
    "pa_drug_coverage": "override_domain",
    "pa_claim_usage": "override_domain",
    "pa_reason_code": "override_domain",
    "pa_effective_dates": "override_domain",
    "pa_agent_code": "override_domain",
    "pa_ignore_status": "override_domain",
    "pa_specialty_rx_override": "override_domain",
    "pa_clinical_admin_code": "override_domain",
    "pa_transform_care": "override_domain",
    "pa_follow_me_logic": "override_domain",
    "pa_drug_type_indicator": "override_domain",
    "pa_modification_history": "override_domain",
    # ── override_domain (new operational how-to intents) ─────────────────
    "pa_contingent_therapy_override": "override_domain",
    "pa_smart_pa_override": "override_domain",
    "pa_part_b_override": "override_domain",
    "pa_esrd_override": "override_domain",
    "pa_skip_deductible": "override_domain",
    "pa_send_expiration": "override_domain",
    "pa_tf_letter_setup": "override_domain",
    "pa_copay_setup": "override_domain",
    "pa_suggest_override": "override_domain",
    "pa_reason_code_fields": "override_domain",
}

# ─────────────────────────────────────────────────────────────────────────────
# Domain → API endpoint mapping
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_ENDPOINTS: Dict[str, Optional[str]] = {
    "cap_api": "/myclaims/claims/v1/claim/byclaimnumber",
    "benefits_api": "/myclaims/benefits/v1/member",
    "claim_history_search": "/myclaims/claims/v1/claim/history",
    "member_domain": "/myclaims/members/v1/member",
    "override_domain": "/myclaims/overrides/v1/pa",
    "general": None,
}

# ─────────────────────────────────────────────────────────────────────────────
# Domain → Friendly name
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_NAMES: Dict[str, str] = {
    "cap_api": "Cap-API",
    "benefits_api": "Benefits API",
    "claim_history_search": "Claim History Search",
    "member_domain": "Member Domain",
    "override_domain": "Override Domain",
    "general": "General",
}

# ─────────────────────────────────────────────────────────────────────────────
# Intent descriptions  (used by LLM fallback for disambiguation)
# ─────────────────────────────────────────────────────────────────────────────

INTENT_DESCRIPTIONS: Dict[str, str] = {
    # ── cap_api (single-claim) ────────────────────────────────────────────
    "claim_status": "General claim status, adjudication outcome, paid/rejected/pending",
    "multi_claim_summary": "Summary of ALL/MULTIPLE claims for a member",
    "pharmacy_info": "Dispensing pharmacy name, location, address, NCPDP, store for ONE claim",
    "prescriber_info": "Prescribing physician/doctor name, NPI, credentials for ONE claim",
    "pricing_info": "Copay, ingredient cost, dispensing fee, patient pay breakdown for ONE claim",
    "reimbursement_info": "Amount paid TO pharmacy, reimbursement rationale, payment",
    "rejection_reasons": "Rejection codes, failed edits, denial reasons, how to resolve for ONE claim",
    "settlement_info": "Settlement codes, pharmacy response/feedback codes for ONE claim",
    "rx_details": "RX number, fill number, quantity, days supply, strength",
    "reversal_info": "Claim reversal, R&R, manual adjustments, resubmission",
    "cob_info": "Coordination of Benefits, other insurance, secondary payer, dual coverage",
    "generic_availability": "Generic alternatives, therapeutic equivalents, formulary substitutes",
    "daw_info": "DAW status, brand vs generic requirement, substitution",
    "government_claim_type": "Medicare/Medicaid claim type, government program",
    "mail_order_info": "Mail order/home delivery prescription, shipping",
    "medicare_part_d": "Medicare Part D summary, PDE, MEDD pricing, LICS for a claim",
    "network_info": "Pharmacy network details, which network processed claim",
    "prior_auth_info": "Prior authorization status, Smart PA, authorization requirements for ONE claim",
    # ── benefits_api ─────────────────────────────────────────────────────
    "approval_info": "Claim approval, plan overrides, transition fill (TF), BPG, Smart PA",
    "audit_info": "Audit trail, change history, modification records, timestamps",
    "beneficiary_info": "Member benefit phase, coverage tier, eligibility, accumulations",
    "plan_summary": "Benefit plan overview, active plan snapshot, current coverage summary",
    "plan_history": "Plan change log, revision history, amendment timeline, past plan updates",
    "plan_finder": "Search/find available benefit plans, plan catalog lookup, plan matching",
    # ── claim_history_search (SEARCH/FILTER multiple claims) ─────────────
    "compound_info": "Compound medication, MIC breakdown, ingredient costs",
    "date_range_claims": "Claims within date range, deductible claims, accumulation history",
    "drug_info": "Drug name, NDC, GPI, therapeutic class, formulary status",
    "drug_interaction_info": "DUR edits, drug interaction alerts, clinical screening",
    "fill_date_info": "Date prescription was filled, dispensing date, service date",
    "Refills": "Search claims by refill count, refill history, remaining refills",
    "DaysSupply": "Filter claims by days supply duration (7, 14, 30, 60, 90 days)",
    "PriorAuth": "Search claims that required prior authorization approval",
    "Diagnosis": "Filter claims by ICD-10 diagnosis code",
    "Settlement": "Search/filter claims by settlement response code NUMBER",
    "PharmType": "Filter claims by pharmacy type/channel (retail, mail-order, specialty)",
    "Plan": "Filter claims by insurance plan code",
    "Pharmacy": "Search claims from a specific pharmacy name/store/location",
    "Prescriber": "Search claims by prescriber name or NPI across claim history",
    "Pricing": "Search claims by cost/pricing for a specific DRUG across MULTIPLE claims",
    "Status": "Filter/list claims by status (paid, rejected, pending, reversed)",
    "RejectCode": "Search claims by NCPDP rejection code number",
    "DrugLast": "When was a specific drug last dispensed/filled for a member",
    "Month": "Filter claims by calendar month (January, February, etc.)",
    "ClaimNum": "Look up a specific claim by its claim number",
    "NDC": "Search claims by NDC (National Drug Code) number",
    "Manufacturer": "Filter claims by drug manufacturer name",
    "Generic": "Filter for generic drug claims only",
    "Brand": "Filter for brand name drug claims only",
    # ── general ──────────────────────────────────────────────────────────
    "greeting": "Hello, hi, welcome, good morning/afternoon/evening",
    "help": "How to submit claims, steps to avoid rejection, filing guidance",
    "out_of_scope": "Unrelated to pharmacy — weather, recipes, sports, gibberish",
    # ── member_domain ────────────────────────────────────────────────────
    "member_coverage": "Member coverage eligibility windows, active coverage status, enrollment dates",
    "member_hierarchy": "Client/CAG hierarchy, client-account-group membership, organizational structure",
    "benefit_reset_date": "Benefit year reset date, accumulator reset, plan year anniversary",
    "family_type": "Individual vs family plan classification, coverage tier type",
    "family_members": "List family members, dependents, subscriber and dependents on same plan",
    "alternate_insurance": "Other/secondary insurance on file, dual coverage, alternate payer",
    "medicare_coverage": "Medicare Part D enrollment status, Med-D plan assignment for a MEMBER",
    "lics_status": "Low Income Subsidy (LICS/LIS) status, subsidy level, cost-sharing reduction",
    "stcob_linkage": "Short-term COB linkage, STCOB member links and records",
    "cvs_id_lookup": "CVS ID associated with the member, CVS member identifier",
    "related_cagm": "Related CAGMs by CVS ID or family ID, linked CAGM records",
    "alternate_ids": "All alternate IDs on file for the member, cross-reference identifiers",
    "member_demographics": "Member name (first/last/middle), date of birth, gender, person code, relationship code",
    "member_contact_info": "Member email, phone number, mailing/postal address, city, state, zip code",
    "member_eligibility_copay": "Eligibility copay fields: copayBrand, copayGeneric, copay3, copay4",
    "member_transition_status": "Member transition fill status and start date from eligibility",
    "member_dur_config": "Drug utilization review (DUR) key and process flag, DUR configuration",
    "member_mbi_number": "Medicare Beneficiary Identifier (MBI) number from Part D record",
    "member_caretaker_info": "Caretaker name and address from Medicare Part D record",
    "member_language_pref": "Member language code/preference (mbrLangCode), communication language",
    "member_discount_program": "Discount program type assigned to the member",
    "member_override_plan": "Member-level override plan ID from eligibility (memberOverridePlan)",
    # ── override_domain ──────────────────────────────────────────────────
    "pa_summary": "Prior authorization summary of key fields, PA overview and configuration",
    "pa_override_reject": "Will PA override specific reject codes (75, 70, 76), PA reject handling",
    "pa_field_help": "Explanation of what a specific PA field does, PA field documentation",
    "pa_copay_pricing": "PA copay override impact on pricing, copay influence on cost",
    "pa_drug_coverage": "Drugs covered by this PA (GPI/NDC lists), PA drug scope",
    "pa_claim_usage": "How many claims used/referenced this PA, PA utilization count",
    "pa_reason_code": "PA reason code (U1, LC, OD, OA, US, U3), override reason classification",
    "pa_effective_dates": "PA effective period begin/end dates, when PA is active or expired",
    "pa_agent_code": "Agent/source code on PA (A, C, 3, H), who created or modified the PA",
    "pa_ignore_status": "Ignore status code on PA (Y, P, 3), processing bypass indicator",
    "pa_specialty_rx_override": "Specialty Rx reject override indicator, bypass specialty rejection",
    "pa_clinical_admin_code": "Clinical administration code on PA (A, C, blank)",
    "pa_transform_care": "Transform care type on PA, care transformation program",
    "pa_follow_me_logic": "Follow me logic indicator, PA follows member across plan changes",
    "pa_drug_type_indicator": "Authorized drug type (G=GPI, N=NDC), drug matching method",
    "pa_modification_history": "PA modification date/time (modifyDateTime), last update timestamp",
    # ── override_domain (new operational how-to intents) ─────────────────
    "pa_contingent_therapy_override": "How to bypass/override contingent therapy on a PA, flip contingent therapy flag",
    "pa_smart_pa_override": "How to bypass/override Smart PA processing, enter Smart PA criteria number",
    "pa_part_b_override": "How to make a claim pay as Medicare Part-B using PA override reason MB",
    "pa_esrd_override": "How to override ESRD reject using PA override reason ES",
    "pa_skip_deductible": "How to skip deductible for a member using PA, flip skip DED flag",
    "pa_send_expiration": "How to send PA expiration date on a claim, flip send expiration date flag",
    "pa_tf_letter_setup": "How to setup TF (Transition Fill) letter tag on a PA, TF letter type configuration",
    "pa_copay_setup": "How to configure a different copay schedule on a PA override, copay setup process",
    "pa_suggest_override": "What PA to enter for overriding a reject, suggest override feature, find appropriate PA",
    "pa_reason_code_fields": "What fields are applicable or required for a specific PA override reason code",
}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_domain_for_intent(intent: str) -> str:
    """Return the domain string for a given intent, or 'unknown'."""
    return INTENT_TO_DOMAIN.get(intent, "unknown")


def get_endpoint_for_domain(domain: str) -> Optional[str]:
    """Return the API endpoint for a domain, or None."""
    return DOMAIN_ENDPOINTS.get(domain)


def get_all_intents() -> list:
    """Return a sorted list of all known intent names."""
    return sorted(INTENT_TO_DOMAIN.keys())


def get_all_domains() -> list:
    """Return a sorted list of all unique domain names."""
    return sorted(set(INTENT_TO_DOMAIN.values()))


def get_intents_for_domain(domain: str) -> list:
    """Return all intents that belong to a specific domain."""
    return sorted(k for k, v in INTENT_TO_DOMAIN.items() if v == domain)
