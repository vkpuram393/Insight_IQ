"""
Claims-domain rendering configuration — all domain-specific rendering intelligence.

Separates domain knowledge from generic rendering logic (Single Responsibility Principle).
Future domains (member portal, overrides) create their own *_rendering_config.py with
their own field remaps, code maps, and null-handling rules.

All mappings are verified against the claims domain system prompt. Entries sourced from
industry convention (not explicitly in the prompt) are annotated as such.
"""

# ---------------------------------------------------------------------------
# Field name aliases → authoritative CAP API field names
#
# Corrects common LLM alias errors in DSL column field names. When the LLM
# writes a field alias (e.g. "drugName" instead of "drugLabelName"), the
# rendering engine normalises it to the real API field name before lookup.
#
# All 28 entries verified against claims domain prompt trusted field names.
# ---------------------------------------------------------------------------
FIELD_REMAP: dict[str, str] = {
    # Fill date aliases — claim domain prompt uses date2 and dateOfFill interchangeably
    "date2": "dateOfFill",
    "fillDate2": "dateOfFill",
    "filldate2": "dateOfFill",
    "fillDate": "dateOfFill",
    # Date of birth aliases
    "date8": "dateOfBirth",
    "dob": "dateOfBirth",
    "birthDate": "dateOfBirth",
    "dateOfBirth": "dateOfBirth",       # no-op safety mapping (idempotent)
    # Reject code aliases
    "code": "responseRejectCode",
    "rejectCode": "responseRejectCode",
    "reject_code": "responseRejectCode",
    "rejectionCode": "responseRejectCode",
    # Status aliases → statusDescription which already holds the friendly label
    "status": "statusDescription",
    "claimStatus": "statusDescription",
    "claimStatusDescription": "statusDescription",
    "approvalStatus": "statusDescription",
    "adjudicationStatus": "statusDescription",
    # Drug name aliases
    "drugName": "drugLabelName",
    "productName": "drugLabelName",
    "medicationName": "drugLabelName",
    "drug": "drugLabelName",
    # Member name aliases → lastNameFirstName is the canonical combined name field
    "memberName": "lastNameFirstName",
    "beneficiaryName": "lastNameFirstName",
    "patientName": "lastNameFirstName",
    "name": "lastNameFirstName",
    # Member ID aliases
    "id": "memberId",
    "memberID": "memberId",
    "cardholderID": "memberId",
    # NDC aliases → submittedProductId is the CAP API field for the NDC
    "ndc": "submittedProductId",
    "ndcNumber": "submittedProductId",
    # Settlement message alias
    "message": "settlementMessage",
    # Claim number aliases
    "claim": "claimNumber",
    "claimId": "claimNumber",
}

# ---------------------------------------------------------------------------
# Claim status code → human-readable display label
#
# Source: claims domain prompt (Intelligence 2 — code-to-label mapping)
#
# IMPORTANT — "X" corrected to "Reversed":
#   The prior STATUS_CODE_MAP in rendering_themes.py mapped X → "Voided".
#   This is WRONG. The claims domain prompt explicitly states at lines 524
#   and 1038: "Status is 'P' (Paid) or 'X' (Reversed)" and "X - Reversed".
#   Fixed here: X → "Reversed".
#
# "V" maps to "Reversed" (the prompt also uses "Reversed/Cancelled" for V,
#   but "Reversed" is the canonical short form).
#
# "D", "A", "C", "E": NOT explicitly defined in the claims domain prompt's
#   code mapping table. Retained as standard NCPDP/PBM industry conventions
#   that are broadly accepted. Showing a friendly label is better UX than
#   displaying the raw single-character code.
# ---------------------------------------------------------------------------
CLAIM_STATUS_CODES: dict[str, str] = {
    "P": "Paid",        # claims domain prompt: "P - Paid" (verified)
    "R": "Rejected",    # claims domain prompt: "R - Rejected" (verified)
    "V": "Reversed",    # claims domain prompt: "Reversed/Cancelled" → canonical "Reversed" (verified)
    "X": "Reversed",    # CORRECTED from "Voided" — prompt explicitly: "X - Reversed" (verified)
    "D": "Denied",      # industry convention — not in prompt code table (unverified but standard)
    "A": "Adjusted",    # industry convention — not in prompt code table (unverified but standard)
    "C": "Cancelled",   # industry convention — not in prompt code table (unverified but standard)
    "E": "Error",       # industry convention — not in prompt code table (unverified but standard)
}

# ---------------------------------------------------------------------------
# DSL format types where a null cell value renders as "$0.00" rather than "—"
#
# These implement domain-specific null-handling rules from the claims domain
# prompt (Intelligence 3). Two rules are encoded here:
#
# "stcob_currency" — STCOB pricing context (all linkedClaim.stcob financial fields)
#   Claims domain prompt line 685 (verbatim):
#     "For ALL fields in the STCOB pricing table above, when the value in the
#      claim data is null, report $0.00 (do NOT say 'not available' or
#      'not populated')"
#   Also covers STCOB submitted patient pay (prompt lines 1088–1090):
#     "If patientPaidAmount is null, report as $0.00."
#
# "med_d_currency" — Medicare Part D financial accumulation fields
#   Claims domain prompt lines 1301–1307 (verbatim):
#     "For ALL financial fields in the Medicare D tables... Field value is
#      0, 0.0, 0.00, or null → report as '$0.00' — this IS the correct
#      answer. NEVER say 'not available' for a financial field that is 0
#      or null in these tables."
# ---------------------------------------------------------------------------
NULL_AS_ZERO_CURRENCY_FORMATS: frozenset[str] = frozenset({
    "stcob_currency",
    "med_d_currency",
})

# ---------------------------------------------------------------------------
# Field names that must never appear in DSL column output
#
# These are internal API fields that either do not exist in the real API
# response or contain data that is unsuitable for direct table display.
#
# Source: claims domain prompt validation rules.
#   "description43Name — DOES NOT EXIST. Strength and dosage form are parsed
#    from the drug name, NOT from a separate field." (prompt line 3161)
# ---------------------------------------------------------------------------
BLOCKED_FIELDS: frozenset[str] = frozenset({
    "description43Name",
})
