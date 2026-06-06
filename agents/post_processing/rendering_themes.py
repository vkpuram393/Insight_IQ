"""
Rendering Themes — domain constants for MyClaims HTML rendering.

Uses a BLOCKLIST (NO_RENDER_INTENTS) instead of an allowlist so that any
new intent from the LLM judge or keyword classifier that produces successful
tool_results is automatically rendered without code changes.
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

# Tier 1 — always render a full table regardless of LLM decision or question phrasing.
# Only intents whose data structure is invariantly multi-column regardless of question.
MUST_RENDER_INTENTS = frozenset({
    "pricing_info",       # always 5-column cost breakdown
    "cob_info",           # always primary/secondary/final pivot
    "deductible_info",    # always accumulator rows
    "copay_info",         # always tier → copay amount
    "claim_list",         # always multiple claim rows
    "rejection_reasons",  # always code + reason pairs
    "compound_info",      # always ingredient rows
    "medicare_part_d",    # always TrOOP stage rows
})
# Moved to LLM-decides tier (structure varies by data/question):
#   approval_info, prior_auth_info, mail_order_info, rx_details, government_claim_type,
#   claim_summary, date_range_search, expensive_claims, reversal_info, audit_info,
#   reimbursement_info, settlement_info

# Tier 2 — output format is decided purely by data shape inside _pick_visual_mode():
#   multiple rows or sections      → html_table
#   single row, ≥6 visible columns → html_table  (too complex for prose)
#   single row, <6 visible columns → text        (LLM prose sufficient)
# No intent names live here — the LLM decides render_mode and data shape decides format.

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
}

# Single-char status codes from the API -> human-readable label.
STATUS_CODE_MAP = {
    "P": "Paid",
    "R": "Rejected",
    "D": "Denied",
    "V": "Reversed",
    "A": "Adjusted",
    "C": "Cancelled",
    "E": "Error",
    "X": "Voided",
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
}

# Valid format type identifiers the LLM may return in column definitions.
# Any format not in this set is silently defaulted to "text".
VALID_FORMAT_TYPES = frozenset({
    "text",
    "date",
    "currency",
    "status_badge",
    "reject_codes",
    "title",
})
