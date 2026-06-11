"""
Rendering Themes — generic rendering infrastructure constants.

Domain-specific intelligence (code maps, field remaps, null rules) lives in
claims_rendering_config.py. This file contains only rendering-engine constants
that are domain-agnostic or intent-to-title mappings.
"""

# Intents that are guaranteed to NEVER produce renderable API data.
# Every other intent passes through to the rendering agent.
NO_RENDER_INTENTS = frozenset({
    "greeting",
    "help",
    "out_of_scope",
    "empty_query",
    "unknown",
})

# Output format is decided by the LLM's render_mode choice and then validated by
# _pick_visual_mode() which always renders as table when render_mode="table".
# The LLM uses the universal row-count rule: table only when answer has 2+ rows.

# Valid values for render_mode in the LLM JSON envelope.
VALID_RENDER_MODES = frozenset({"text_only", "table"})

# Intent -> table heading. Intents not listed here use "Data Results" fallback.
TABLE_TITLES = {
    "claim_list":           "Claims Data",
    "claim_status":         "Claim Status",
    "claim_details":        "Claim Details",
    "claim_pending":        "Pending Claims",
    "claim_summary":        "Claims Summary",
    "rejection_reasons":    "Rejection Details",
    "expensive_claims":     "High-Cost Claims",
    "date_range_search":    "Claims Data",
    "member_info":          "Member Information",
    "benefits_info":        "Benefits Information",
    "copay_info":           "Copay Schedule",
    "deductible_info":      "Deductible & Accumulator Summary",
    "cob_info":             "Coordination of Benefits",
    "drug_info":            "Drug Details",
    "pharmacy_info":        "Pharmacy Information",
    "prescriber_info":      "Prescriber Information",
    "pricing_info":         "Pricing Details",
    "approval_info":        "Approval Details",
    "reimbursement_info":   "Reimbursement Details",
    "rx_details":           "Prescription Details",
    "beneficiary_info":     "Beneficiary Information",
    "audit_info":           "Audit Details",
    "settlement_info":      "Settlement Details",
    "reversal_info":        "Reversal Details",
    "compound_info":        "Compound Drug Details",
    "medicare_part_d":      "Medicare Part D Details",
    "government_claim_type":"Government Claim Details",
    "mail_order_info":      "Mail Order Details",
    "prior_auth_info":      "Prior Authorization Details",
    # ── claim_history_search domain (multi-claim, member-history) ──────────
    "DateRange":             "Recent Claims",
    "date_range_claims":     "Claims by Date Range",
    "DrugList":              "Member Medication List",
    "DrugLast":              "Most Recent Fill",
    "Refills":               "Refill Activity",
    "DaysSupply":            "Days Supply",
    "PriorAuth":             "Prior Authorization Claims",
    "Diagnosis":             "Diagnosis Codes",
    "Settlement":            "Settlement Codes",
    "PharmType":             "Pharmacy Type",
    "Plan":                  "Plan-filtered Claims",
    "Pharmacy":              "Pharmacy History",
    "Prescriber":            "Prescriber History",
    "Pricing":               "Cost & Pricing",
    "Status":                "Claims by Status",
    "RejectCode":            "Rejected Claims",
    "Month":                 "Claims by Month",
    "ClaimNum":              "Claim Lookup",
    "NDC":                   "NDC-filtered Claims",
    "Manufacturer":          "Manufacturer Claims",
    "Generic":               "Generic Drug Claims",
    "Brand":                 "Brand Drug Claims",
    "fill_date_info":        "Fill Date Lookup",
    "drug_interaction_info": "DUR / Drug Interactions",
    "multi_claim_summary":   "Member Claims Summary",
}

# CSS class prefix — all generated HTML is scoped under .mc-poc
CSS_SCOPE = "mc-poc"

# Fallback title when intent has no entry in TABLE_TITLES.
DEFAULT_TABLE_TITLE = "Data Results"

# Human-readable intent descriptions sent to the LLM as context for structure extraction.
# The LLM uses these to decide which fields are most relevant to the user's query.
INTENT_DESCRIPTIONS = {
    "claim_status":         "User wants the current status of a specific claim",
    "claim_list":           "User wants a list of their pharmacy claims",
    "claim_details":        "User wants detailed information about a specific claim",
    "claim_pending":        "User wants to see claims that are still being processed",
    "claim_summary":        "User wants a summary of their claim history and totals",
    "rejection_reasons":    "User wants to understand why a claim was denied — show reject codes and denial reasons",
    "expensive_claims":     "User wants their highest-cost claims",
    "date_range_search":    "User wants claims within a specific date range",
    "copay_info":           "User wants copay amounts by drug tier",
    "cob_info":             "User wants coordination of benefits information — show COB type, other carrier, amounts paid",
    "benefits_info":        "User wants their pharmacy benefit details",
    "deductible_info":      "User wants deductible and accumulator status",
    "drug_info":            "User wants drug details — show drug name, NDC, quantity, days supply, generic indicator",
    "pharmacy_info":        "User wants pharmacy information — show pharmacy name, ID, city, state, phone",
    "prescriber_info":      "User wants prescriber information — show prescriber name, NPI, DEA",
    "member_info":          "User wants their member profile information",
    "pricing_info":         "User wants pricing breakdown — show ingredient cost, dispensing fee, patient pay, plan paid, basis of reimbursement",
    "approval_info":        "User wants approval status — show claim status, approval date, adjudication details",
    "reimbursement_info":   "User wants reimbursement details — show ingredient cost, dispensing fee, plan reimbursement, basis of reimbursement",
    "rx_details":           "User wants prescription details — show drug name, NDC, quantity, days supply, fill date",
    "beneficiary_info":     "User wants beneficiary information — show member name, ID, DOB, person code",
    "audit_info":           "User wants audit trail — show claim number, status, dates, adjudication details",
    "settlement_info":      "User wants settlement information — show amounts settled, payment details",
    "reversal_info":        "User wants reversal information — show original claim, reversal date, reversal reason",
    "compound_info":        "User wants compound drug details — show compound ingredients, quantities, costs",
    "medicare_part_d":      "User wants Medicare Part D information — show coverage stage, TrOOP amounts",
    "government_claim_type":"User wants government claim type information — show claim type, coverage details",
    "mail_order_info":      "User wants mail order details — show pharmacy type, days supply, refill info",
    "prior_auth_info":      "User wants prior authorization details — show auth number, status, dates",
    # ── claim_history_search domain (multi-claim, member-history) ──────────
    "DateRange":            "User wants claims within a rolling recent window (last N days/weeks/months)",
    "date_range_claims":    "User wants claims within an explicit from-to date range",
    "DrugList":             "User wants the full medication list across all of the member's claims",
    "DrugLast":             "User wants the most recent fill for a specific drug",
    "Refills":              "User wants refill activity / counts / overdue refills",
    "DaysSupply":           "User wants claims filtered by days-supply duration",
    "PriorAuth":            "User wants claims that required prior authorization",
    "Diagnosis":            "User wants claims filtered by diagnosis / ICD code",
    "Settlement":           "User wants claims filtered by settlement code",
    "PharmType":            "User wants claims filtered by pharmacy type (retail / mail order / specialty)",
    "Plan":                 "User wants claims filtered by plan / carrier",
    "Pharmacy":             "User wants pharmacy history across claims",
    "Prescriber":           "User wants prescriber history across claims",
    "Pricing":              "User wants cost / pricing across claims (totals, patient pay, plan pay)",
    "Status":               "User wants claims filtered by status (paid / rejected / reversed)",
    "RejectCode":           "User wants rejected claims with reject codes and reasons",
    "Month":                "User wants claims filtered by month name",
    "ClaimNum":             "User wants to look up by claim number",
    "NDC":                  "User wants claims filtered by NDC",
    "Manufacturer":         "User wants claims filtered by drug manufacturer",
    "Generic":              "User wants generic drug claims",
    "Brand":                "User wants brand-name drug claims",
    "fill_date_info":       "User wants the fill date for a claim",
    "drug_interaction_info":"User wants DUR / drug-interaction details across claims",
    "multi_claim_summary":  "User wants a cross-claim summary or aggregate over the member's history",
}

# Valid format type identifiers the LLM may return in column definitions.
# Any format not in this set is silently defaulted to "text".
# Domain-specific null-handling variants:
#   stcob_currency — STCOB pricing fields: null renders as $0.00 (not —)
#   med_d_currency — Medicare Part D financial fields: null renders as $0.00
VALID_FORMAT_TYPES = frozenset({
    "text",
    "date",
    "currency",
    "stcob_currency",
    "med_d_currency",
    "status_badge",
    "reject_codes",
    "title",
})

# ---------------------------------------------------------------------------
# DEPRECATED back-compat aliases for the v3 tier model.
#
# In rendering-agent-v4 the rendering engine no longer consults a
# MUST_RENDER allowlist or a STATUS_CODE_MAP — the LLM`s render_mode
# choice is honoured directly and per-domain status codes live in
# claims_rendering_config.py / claim_history_rendering_config.py.
#
# These aliases are kept ONLY so the existing test suite
# (tests/test_rendering_scenarios.py) and any external scripts that import
# them continue to load. New code must NOT use them.
# ---------------------------------------------------------------------------
MUST_RENDER_INTENTS = frozenset({
    "pricing_info",
    "cob_info",
    "deductible_info",
    "copay_info",
    "claim_list",
    "rejection_reasons",
    "compound_info",
    "medicare_part_d",
})

# v3 single-char status codes — use claims_rendering_config.CLAIM_STATUS_CODES
# (or the per-domain config) for new code. Kept here only for legacy imports.
from agents.post_processing.claims_rendering_config import (
    CLAIM_STATUS_CODES as STATUS_CODE_MAP,
)
