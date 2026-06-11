"""
Claim-History (member-history search) rendering configuration.

Mirrors the structure of `claims_rendering_config.py` but holds the field
aliases, status code map, null-rules, and blocked-fields specific to the
slim-claim shape produced by `Claims_search_api.llm_query_responder.
prepare_claim_history_data()` — which trims the raw /claims/search
response into the 7 sections:
    claimInformation, drug, pricing, prescription,
    priorAuthorization, overrides, messages

The rendering engine (myclaims_rendering_agent.py) is fully
domain-agnostic. When the post-graph rendering pipeline detects a
claim_history_search domain (either via state["domain"] or via
tool_results.data.is_claim_history_search), it loads THIS module
instead of `claims_rendering_config.py`.

Adding a new domain (member, overrides, ...) follows the same pattern:
  1. drop a `<domain>_rendering_config.py` next to this file
  2. register it in `domain_configs.py`
  3. add the domain's intents to rendering_themes.TABLE_TITLES /
     INTENT_DESCRIPTIONS
  4. append the rendering DSL contract to that domain's prompt builder
"""

# ---------------------------------------------------------------------------
# Field name aliases  ->  authoritative slim-claim key names
#
# The claim-history slim-claim shape (see _CLAIM_FIELD_WHITELIST in
# Claims_search_api/llm_query_responder.py) uses different field names
# than the single-claim CAPI response. The LLM frequently writes the
# claims-domain alias even when answering a claim-history query, so we
# remap to the CHS keys here.
# ---------------------------------------------------------------------------
FIELD_REMAP: dict[str, str] = {
    # Fill date — claims-domain prompt uses dateOfFill / date2; CHS uses fillDate
    "dateOfFill":           "fillDate",
    "date2":                "fillDate",
    "fillDate2":            "fillDate",
    "filldate2":            "fillDate",
    "fillDate":             "fillDate",     # idempotent safety
    # Claim number / sequence
    "claim":                "claimNumber",
    "claimId":              "claimNumber",
    "sequence":             "claimSequenceNumber",
    "sequenceNumber":       "claimSequenceNumber",
    # Drug name — CHS shape: drug.productName
    "drugLabelName":        "productName",
    "drugName":             "productName",
    "drug":                 "productName",
    "medicationName":       "productName",
    # NDC — CHS shape: drug.productNdc
    "ndc":                  "productNdc",
    "ndcNumber":            "productNdc",
    "submittedProductId":   "productNdc",
    # Status — CHS exposes both claimStatus (single char) and claimStatusDescription
    "status":               "claimStatusDescription",
    "statusDescription":    "claimStatusDescription",
    "approvalStatus":       "claimStatusDescription",
    "adjudicationStatus":   "claimStatusDescription",
    # Reject codes — CHS keeps them under messages.rejectCodes
    "responseRejectCode":   "rejectCodes",
    "rejectCode":           "rejectCodes",
    "reject_code":          "rejectCodes",
    "rejectionCode":        "rejectCodes",
    # Pricing — CHS uses pricing.patientPay / pricing.clientPay
    "approvedPatientPayAmount":  "patientPay",
    "approvedTotalAmount":       "amountDueApproved",
    "approvedIngredientCost":    "drugCostApproved",
    "approvedDispensingFee":     "dispensingFeeApproved",
    "clientPatientPayAmount":    "patientPay",
    "clientTotalAmount":         "amountDueApproved",
    # Prescription — prescriber/pharmacy lives under prescription.*
    "prescriberLastName":   "prescriberLastName",   # idempotent
    "prescriberFirstName":  "prescriberFirstName",  # idempotent
    "lastNameFirstName":    "prescriberLastName",
    "pharmacyName":         "pharmacyName",         # idempotent
    "submitDate":           "submitDate",           # idempotent
    "reversalDate":         "reversalDate",         # idempotent
    # Days supply / quantity
    "submittedQuantityDispensed":  "quantity",
    "submittedDaysSupply":         "daysSupplied",
    "daysSupply":                  "daysSupplied",
}

# ---------------------------------------------------------------------------
# Single-char status codes -> human-readable label
# CHS keeps the same single-letter codes as the single-claim CAPI.
# IMPORTANT: "X" -> "Reversed" (NEVER "Voided") — same correction as
# claims_rendering_config.
# ---------------------------------------------------------------------------
CLAIM_STATUS_CODES: dict[str, str] = {
    "P": "Paid",
    "R": "Rejected",
    "V": "Reversed",
    "X": "Reversed",        # corrected — matches v4 claims-domain config
    "D": "Denied",
    "A": "Adjusted",
    "C": "Cancelled",
    "E": "Error",
}

# ---------------------------------------------------------------------------
# DSL format types where a null cell renders as "$0.00" rather than "—".
# CHS does not expose STCOB or Med-D financial subtrees (the slim-claim
# whitelist drops them), so this set is empty by design.
# ---------------------------------------------------------------------------
NULL_AS_ZERO_CURRENCY_FORMATS: frozenset = frozenset()

# ---------------------------------------------------------------------------
# Field names that must never appear in DSL columns for this domain.
# CHS has no equivalent of `description43Name`. Empty by default.
# ---------------------------------------------------------------------------
BLOCKED_FIELDS: frozenset = frozenset()
