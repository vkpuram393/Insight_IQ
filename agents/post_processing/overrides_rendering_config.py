"""
Override-domain (Prior Authorization) rendering configuration.

Mirrors the structure of `claims_rendering_config.py` and
`claim_history_rendering_config.py` but holds the field aliases, status code
maps, null rules, and blocked-field set specific to PA records produced by
`Overrides_api.response_trimmer.trim_overrides_response()`.

The PA slim-record shape (see Overrides_api.response_trimmer.PA_FIELD_WHITELIST):
    paReferenceNumber, paStatusCode, paStatusDescription,
    drugName, ndc, gpi,
    effectiveDate, terminationDate,
    quantityAllowed, daysSupplyAllowed, approvedRefillCount,
    rejectReasonCode, rejectReasonDescription,
    overrideCode, overrideDescription,
    agentCode, ignoreStatus, drugTypeIndicator, transformCarePlanIndicator,
    followMeIndicator, clinicalAdminCode, specialtyRxOverrideIndicator,
    copayAmount, pricingInfo, modificationHistory

Selection: when state["domain"] == "override_domain" (or
tool_results.data.is_override_search is True), the rendering engine
loads THIS module instead of the claims / CHS configs.
"""

# ---------------------------------------------------------------------------
# Field name aliases -> authoritative PA-slim key names
#
# The LLM frequently emits a claims-domain alias even when answering a PA
# query. Remap any near-synonym to the canonical PA-slim field name so the
# rendering engine can locate the column data.
# ---------------------------------------------------------------------------
FIELD_REMAP: dict[str, str] = {
    # Reference number
    "paReference":          "paReferenceNumber",
    "paRef":                "paReferenceNumber",
    "paNumber":             "paReferenceNumber",
    "priorAuthNumber":      "paReferenceNumber",
    "priorAuthorizationNumber": "paReferenceNumber",
    "paId":                 "paReferenceNumber",
    "paReferenceNumber":    "paReferenceNumber",   # idempotent safety

    # Status
    "status":               "paStatusDescription",
    "paStatus":             "paStatusDescription",
    "statusDescription":    "paStatusDescription",
    "paStatusCode":         "paStatusCode",
    "paStatusDescription":  "paStatusDescription",   # idempotent

    # Drug
    "drug":                 "drugName",
    "medicationName":       "drugName",
    "drugLabelName":        "drugName",
    "productName":          "drugName",
    "drugName":             "drugName",              # idempotent
    "ndcNumber":            "ndc",
    "productNdc":           "ndc",
    "submittedProductId":   "ndc",
    "ndc":                  "ndc",                   # idempotent
    "gpi":                  "gpi",                   # idempotent

    # Effective / termination
    "effectiveStart":       "effectiveDate",
    "startDate":            "effectiveDate",
    "fromDate":             "effectiveDate",
    "effectiveDate":        "effectiveDate",         # idempotent
    "effectiveEnd":         "terminationDate",
    "endDate":              "terminationDate",
    "throughDate":          "terminationDate",
    "terminationDate":      "terminationDate",       # idempotent

    # Reject / override codes
    "rejectCode":           "rejectReasonCode",
    "rejectionCode":        "rejectReasonCode",
    "rejectReason":         "rejectReasonDescription",
    "rejectDescription":    "rejectReasonDescription",
    "rejectReasonCode":     "rejectReasonCode",          # idempotent
    "rejectReasonDescription": "rejectReasonDescription", # idempotent
    "overrideType":         "overrideDescription",
    "overrideReason":       "overrideDescription",
    "overrideCode":         "overrideCode",              # idempotent
    "overrideDescription":  "overrideDescription",       # idempotent

    # Quantity / supply / refills
    "quantity":             "quantityAllowed",
    "qtyAllowed":           "quantityAllowed",
    "daysSupply":           "daysSupplyAllowed",
    "daysSupplied":         "daysSupplyAllowed",
    "refillsApproved":      "approvedRefillCount",
    "refillCount":          "approvedRefillCount",

    # Indicators / flags
    "followMeLogic":        "followMeIndicator",
    "followMe":             "followMeIndicator",
    "specialtyRxOverride":  "specialtyRxOverrideIndicator",
    "transformCare":        "transformCarePlanIndicator",
    "drugType":             "drugTypeIndicator",

    # Agent
    "agent":                "agentCode",
    "processingAgent":      "agentCode",

    # Pricing
    "copay":                "copayAmount",
    "copayAmount":          "copayAmount",          # idempotent
}


# ---------------------------------------------------------------------------
# PA single/two-letter status codes -> human-readable label
#
# Sourced from CVS PA business mappings. Approval/Rejection are the dominant
# states; A/R/P/C are well-attested. X exists in some PA exports as
# "Reversed" — keep parallel to the claims-domain X->Reversed correction.
# ---------------------------------------------------------------------------
CLAIM_STATUS_CODES: dict[str, str] = {
    "A":  "Approved",
    "AP": "Approved",
    "R":  "Rejected",
    "RJ": "Rejected",
    "P":  "Pending",
    "PE": "Pending",
    "C":  "Cancelled",
    "CN": "Cancelled",
    "X":  "Reversed",     # parallel to claims-domain X correction
    "E":  "Error",
    "D":  "Denied",
}


# ---------------------------------------------------------------------------
# DSL format types where a null cell renders as "$0.00" rather than em-dash.
#
# PA records have a single financial field (copayAmount). When present-but-zero
# (a $0.00 covered drug), the renderer should display "$0.00" rather than "—".
# ---------------------------------------------------------------------------
NULL_AS_ZERO_CURRENCY_FORMATS: frozenset = frozenset({"pa_copay_currency"})


# ---------------------------------------------------------------------------
# Field names that must never appear in DSL columns.
#
# - modificationHistory and pricingInfo are sub-objects, not scalar columns
#   suitable for table rendering. The LLM may still cite them in prose.
# - PII fields are blocked here as a defense-in-depth backstop; the
#   response_trimmer already strips them, so this should be unreachable.
# ---------------------------------------------------------------------------
BLOCKED_FIELDS: frozenset = frozenset({
    "modificationHistory",
    "pricingInfo",
    # PII backstop — these should already be excluded by the trimmer
    "memberId",
    "carrierId",
    "accountId",
    "groupId",
    "personCode",
})
