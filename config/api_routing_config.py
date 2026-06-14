"""
CVS MyClaims API Routing Configuration
Maps intents to actual CVS API endpoints

⚠️  IMPORTANT NOTE - FOR GUIDANCE ONLY:
    This file documents the intent-to-API mapping for reference purposes.
    When you need to return specific slots from the API, use the intention
    and expected entities listed under each intention in this configuration.
    
    The actual API routing is handled by tools/claims_api.py which uses
    tools/api_repository.py (not this file) for dynamic API matching.
    
    This file serves as:
    - Documentation for what each intent should fetch
    - Reference for required entities per intent
    - Guide for expected API response fields

Based on CVS MyClaims Capability API v1.0:
- /myclaims/claims/v1/claim/byclaimnumber (Basic search)
- /myclaims/claims/v1/claim/byclaimnumberandseq (Detailed info)
"""

# CVS API Base Configuration
CVS_API_BASE_URL = "/myclaims/claims/v1"

# API Endpoints
ENDPOINTS = {
    "basic_search": f"{CVS_API_BASE_URL}/claim/byclaimnumber",
    "detailed_info": f"{CVS_API_BASE_URL}/claim/byclaimnumberandseq"
}

# Intent to API Endpoint Mapping
# UPDATED: All claim-related intents now require BOTH claim_number AND sequence
# If either is missing, the system routes to clarification engine
INTENT_API_ROUTING = {
    # ============================================================
    # BASIC SEARCH API (Faster, lighter)
    # Use for: Quick status checks, list views
    # ============================================================
    
    "claim_status": {
        "api_endpoint": ENDPOINTS["basic_search"],
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.status",
            "claimList[0].primary.statusDescription",
            "claimList[0].primary.number",
            "claimList[0].primary.sequence"
        ],
        "fallback_api": "detailed_info",
        "description": "Quick claim status lookup"
    },
    
    "date_range_claims": {
        "api_endpoint": ENDPOINTS["basic_search"],
        "method": "POST",
        "required_entities": ["date_range"],
        "optional_entities": ["member_id"],
        "response_fields": [
            "claimList[].primary.number",
            "claimList[].primary.status",
            "claimList[].primary.submitted.dateOfFill",
            "claimList[].primary.drug.productName",
            "claimList[].primary.pricing.patientPay"
        ],
        "fallback_api": None,
        "description": "Search claims by date range"
    },
    
    "multi_claim_summary": {
        "api_endpoint": ENDPOINTS["basic_search"],
        "method": "POST",
        "required_entities": ["claim_numbers"],  # Multiple
        "optional_entities": [],
        "response_fields": [
            "claimList[].primary.number",
            "claimList[].primary.status",
            "claimList[].primary.drug.productName",
            "claimList[].primary.pricing.patientPay"
        ],
        "fallback_api": None,
        "description": "Get multiple claims summary"
    },
    
    # ============================================================
    # DETAILED INFO API (Comprehensive, slower)
    # Use for: Detailed queries, specific field requests
    # ============================================================
    
    "rejection_reasons": {
        "api_endpoint": ENDPOINTS["basic_search"],  # statusDetails only in basic search!
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.status",
            "claimList[0].primary.statusDescription",
            "claimList[0].primary.statusDetails.rejectDetails[].code",
            "claimList[0].primary.statusDetails.rejectDetails[].description",
            "claimList[0].primary.statusDetails.messages",
            "claimList[0].primary.statusDetails.settlementCodes",
            "claimList[0].primary.durExistenceStatus",
            "claimList[0].primary.number",
            "claimList[0].primary.drug.productName"
        ],
        "fallback_api": "detailed_info",
        "description": "Detailed rejection information"
    },
    
    "drug_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # drug object clearer in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.drug.productName",
            "claimList[0].primary.drug.productId",
            "claimList[0].primary.drug.gpiNumber",
            "claimList[0].primary.drug.genericIndicator",
            "claimList[0].primary.submitted.quantityDispensed",
            "claimList[0].primary.submitted.daysSupply"
        ],
        "fallback_api": "detailed_info",
        "description": "Medication/drug details"
    },
    
    "pricing_info": {
        "api_endpoint": ENDPOINTS["detailed_info"],
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimDetails.primary.medD.approvedTotalAmount",
            "claimDetails.primary.medD.approvedIngredientCost",
            "claimDetails.primary.medD.approvedDispensingFee",
            "claimDetails.primary.medD.calculatedDispensingFee",
            "claimDetails.primary.medD.totalOtherPayerAmount",
            "claimDetails.primary.medD.usualCustomary"
        ],
        "fallback_api": "basic_search",
        "description": "Detailed pricing breakdown"
    },
    
    "pharmacy_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # pharmacy object only in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.pharmacy.name",
            "claimList[0].primary.pharmacy.city",
            "claimList[0].primary.pharmacy.state",
            "claimList[0].primary.pharmacy.zip",
            "claimList[0].primary.pharmacyPhone"
        ],
        "fallback_api": "detailed_info",
        "description": "Pharmacy location and contact"
    },
    
    "prescriber_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # prescriber object only in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.prescriber.firstName",
            "claimList[0].primary.prescriber.lastName",
            "claimList[0].primary.prescriber.id",
            "claimList[0].primary.prescriber.qualifier"
        ],
        "fallback_api": "detailed_info",
        "description": "Prescriber/doctor information"
    },
    
    "beneficiary_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # beneficiary object only in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.beneficiary.firstName",
            "claimList[0].primary.beneficiary.lastName",
            "claimList[0].primary.beneficiary.memberId",
            "claimList[0].primary.beneficiary.dateOfBirth",
            "claimList[0].primary.beneficiary.gender"
        ],
        "fallback_api": "detailed_info",
        "description": "Member/patient information"
    },
    
    "prior_auth_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # priorAuthorization only in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.priorAuthorization.number",
            "claimList[0].primary.priorAuthorization.reasonCode",
            "claimList[0].primary.priorAuthorization.reasonDescription",
            "claimList[0].primary.priorAuthorization.type"
        ],
        "fallback_api": "detailed_info",
        "description": "Prior authorization details"
    },
    
    "audit_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # audit object only in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.audit.addDate",
            "claimList[0].primary.audit.changeDate",
            "claimList[0].primary.audit.addTime",
            "claimList[0].primary.audit.changeTime"
        ],
        "fallback_api": "detailed_info",
        "description": "Claim audit trail"
    },
    
    "fill_date_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # submitted data in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.submitted.dateOfFill",
            "claimList[0].primary.submitted.date",
            "claimList[0].primary.pharmacy.name"
        ],
        "fallback_api": "detailed_info",
        "description": "Prescription fill date"
    },
    
    "rx_details": {
        "api_endpoint": ENDPOINTS["basic_search"],  # submitted data in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.submitted.rxNumber",
            "claimList[0].primary.submitted.fillNumber",
            "claimList[0].primary.submitted.quantityDispensed",
            "claimList[0].primary.submitted.daysSupply",
            "claimList[0].primary.drug.productName"
        ],
        "fallback_api": "detailed_info",
        "description": "Prescription details"
    },
    
    "approval_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # statusDetails in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.status",
            "claimList[0].primary.statusDescription",
            "claimList[0].primary.statusDetails.approvedMessages[].code",
            "claimList[0].primary.statusDetails.approvedMessages[].description",
            "claimList[0].primary.statusDetails.messages",
            "claimList[0].primary.number"
        ],
        "fallback_api": "detailed_info",
        "description": "Approval messages and codes"
    },
    
    "settlement_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # statusDetails in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.statusDetails.settlementCodes[].code",
            "claimList[0].primary.statusDetails.settlementCodes[].description",
            "claimList[0].primary.status",
            "claimList[0].primary.number"
        ],
        "fallback_api": "detailed_info",
        "description": "Settlement information"
    },
    
    # ============================================================
    # CVS-SPECIFIC INTENTS (Detailed API)
    # ============================================================
    
    "compound_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # compound flag in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.compound",
            "claimList[0].primary.submitted.compoundCode",
            "claimList[0].primary.drug.productName",
            "claimList[0].primary.number"
        ],
        "fallback_api": "detailed_info",
        "description": "Compound medication details"
    },
    
    "medicare_part_d": {
        "api_endpoint": ENDPOINTS["detailed_info"],  # medD nested object only in detailed
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimDetails.primary.medD.approvedTotalAmount",
            "claimDetails.primary.medD.approvedIngredientCost",
            "claimDetails.primary.medD.approvedDispensingFee",
            "claimDetails.primary.medD.claimStatus",
            "claimDetails.additionalDetails.partDDrug",
            "claimDetails.additionalDetails.medBDrugIndicator"
        ],
        "fallback_api": "basic_search",
        "description": "Medicare Part D information"
    },
    
    "daw_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # DAW fields in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.dispenseAsWritten",
            "claimList[0].primary.drug.dawproductSelectionCode",
            "claimList[0].primary.drug.productName",
            "claimList[0].primary.number"
        ],
        "fallback_api": "detailed_info",
        "description": "Dispense as written / DAW code"
    },
    
    "cob_info": {
        "api_endpoint": ENDPOINTS["detailed_info"],  # STCOB data only in detailed
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimDetails.primary.medD.cobClaimIndicator",
            "claimDetails.linkedClaim.stcob.responsePatientPayAmount",
            "claimDetails.linkedClaim.stcob.responsetotalAmountPaid",
            "claimDetails.linkedClaim.stcob.responseotherPayerAmountRecg"
        ],
        "fallback_api": "basic_search",
        "description": "Coordination of benefits"
    },
    
    "network_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # pharmacyNetwork in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.pharmacyNetwork",
            "claimList[0].primary.pharmacy.name",
            "claimList[0].primary.number"
        ],
        "fallback_api": "detailed_info",
        "description": "Pharmacy network status"
    },
    
    "reimbursement_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # reimbursementType in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.reimbursementType",
            "claimList[0].primary.pricing.patientPay",
            "claimList[0].primary.number"
        ],
        "fallback_api": "detailed_info",
        "description": "Reimbursement type and details"
    },
    
    "government_claim_type": {
        "api_endpoint": ENDPOINTS["basic_search"],  # governmentClaimType in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.governmentClaimType",
            "claimList[0].primary.status",
            "claimList[0].primary.number"
        ],
        "fallback_api": "detailed_info",
        "description": "Government program type"
    },
    
    "mail_order_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # mail flag in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.mail",
            "claimList[0].primary.pharmacy.name",
            "claimList[0].primary.number"
        ],
        "fallback_api": "detailed_info",
        "description": "Mail order pharmacy indicator"
    },
    
    "generic_availability": {
        "api_endpoint": ENDPOINTS["basic_search"],  # multiSourceInd in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.drug.multiSourceInd",
            "claimList[0].primary.drug.genericIndicator",
            "claimList[0].primary.drug.productName",
            "claimList[0].primary.number"
        ],
        "fallback_api": "detailed_info",
        "description": "Generic alternatives available"
    },
    
    "drug_interaction_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # durExistenceStatus in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.durExistenceStatus",
            "claimList[0].primary.drug.productName",
            "claimList[0].primary.number"
        ],
        "fallback_api": "detailed_info",
        "description": "Drug utilization review status"
    },
    
    "reversal_info": {
        "api_endpoint": ENDPOINTS["basic_search"],  # rnR flag in basic search
        "method": "POST",
        "required_entities": ["claim_number", "sequence"],  # UPDATED: sequence now required
        "optional_entities": [],
        "response_fields": [
            "claimList[0].primary.rnR",
            "claimList[0].primary.submitted.reversalDate",
            "claimList[0].primary.status",
            "claimList[0].primary.number"
        ],
        "fallback_api": "detailed_info",
        "description": "Claim reversal information"
    },
    
    # ============================================================
    # NON-API INTENTS (Use LLM/FAQ)
    # ============================================================
    
    "greeting": {
        "api_endpoint": None,
        "method": None,
        "required_entities": [],
        "optional_entities": [],
        "response_fields": [],
        "requires_llm": True,
        "description": "User greeting - no API needed"
    },
    
    "help": {
        "api_endpoint": None,
        "method": None,
        "required_entities": [],
        "optional_entities": [],
        "response_fields": [],
        "requires_llm": True,
        "description": "General help - use FAQ/LLM"
    },
    
    "out_of_scope": {
        "api_endpoint": None,
        "method": None,
        "required_entities": [],
        "optional_entities": [],
        "response_fields": [],
        "requires_llm": True,
        "description": "Out of scope - route to LLM"
    },
    
    "appeal_info": {
        "api_endpoint": None,
        "method": None,
        "required_entities": [],
        "optional_entities": [],
        "response_fields": [],
        "requires_llm": True,
        "description": "Appeal process - use FAQ/LLM"
    }
}


# ============================================================================
# CLAIM HISTORY SEARCH (member-level, multi-claim) intents
# ============================================================================
# These intents map to the multidomain_intent_detection "claim_history_search"
# domain.  They share a single search endpoint that returns ALL claims for the
# member identified by the supplied claim_number.  The response is filtered
# in-process (Claims_search_api/search.py) and summarized by the response
# agent using the claim-history specific prompt branch.
#
# Required entity is just `claim_number` (NOT sequence) because the upstream
# /claims/search endpoint accepts a claim_id and returns all sibling claims
# for the same member.
# ============================================================================

CLAIM_HISTORY_SEARCH_ENDPOINT = "/myclaims/claims/exp/v1/claims/search"

_CLAIM_HISTORY_SEARCH_BASE = {
    "api_endpoint": CLAIM_HISTORY_SEARCH_ENDPOINT,
    "method": "POST",
    "required_entities": ["claim_number"],
    "optional_entities": [],
    "response_fields": [],          # Trimmed/filtered in Claims_search_api
    "fallback_api": None,
    "domain": "claim_history_search",
    "description": "Member-level claim history search (multi-claim filter)",
}

_CLAIM_HISTORY_INTENTS = [
    "NDC", "Manufacturer", "Generic", "Brand", "Refills", "DaysSupply",
    "PriorAuth", "Diagnosis", "Settlement", "PharmType", "Plan",
    "Pharmacy", "Prescriber", "Pricing", "Status", "RejectCode",
    "DrugLast", "Month", "ClaimNum",
]

for _ch_intent in _CLAIM_HISTORY_INTENTS:
    INTENT_API_ROUTING.setdefault(_ch_intent, dict(_CLAIM_HISTORY_SEARCH_BASE))


# ─────────────────────────────────────────────────────────────────────────────
# OVERRIDE_DOMAIN — Prior Authorization (PA) lookup
#
# All 16 PA intents share the same routing config: claim_number is the only
# required entity (Step 1 resolves it to a member CAGM, then Step 2 fetches
# PA records). Endpoint resolved at call time from settings.overrides_api_path.
# ─────────────────────────────────────────────────────────────────────────────

# Use settings if available, else hardcode the SIT1 default. Done once at module
# load so callers can rely on a stable api_endpoint string.
try:
    from config.config import settings as _settings  # noqa: WPS433 (intentional late-bind)
    _OVERRIDES_ENDPOINT = (
        getattr(_settings, "overrides_api_base_url", "https://internal-sit1-apix.cvshealth.com").rstrip("/")
        + getattr(_settings, "overrides_api_path", "/pss/myclaims/override/exp/v1/priorauth/search")
    )
except Exception:
    _OVERRIDES_ENDPOINT = (
        "https://internal-sit1-apix.cvshealth.com"
        "/pss/myclaims/override/exp/v1/priorauth/search"
    )


_OVERRIDE_DOMAIN_BASE = {
    "api_endpoint":      _OVERRIDES_ENDPOINT,
    "method":            "POST",
    "required_entities": ["claim_number"],   # claim_id is mandatory per user spec
    "optional_entities": [],
    "response_fields":   [],                  # Trimmed in Overrides_api.response_trimmer
    "fallback_api":      None,
    "domain":            "override_domain",
    "description":       "Prior Authorization (PA) record lookup and field analysis",
    "tool_name":         "overrides_v1",
    "requires_llm":      True,
}

_OVERRIDE_INTENTS = [
    "pa_summary",
    "pa_override_reject",
    "pa_field_help",
    "pa_copay_pricing",
    "pa_drug_coverage",
    "pa_claim_usage",
    "pa_reason_code",
    "pa_effective_dates",
    "pa_agent_code",
    "pa_ignore_status",
    "pa_specialty_rx_override",
    "pa_clinical_admin_code",
    "pa_transform_care",
    "pa_follow_me_logic",
    "pa_drug_type_indicator",
    "pa_modification_history",
]

for _pa_intent in _OVERRIDE_INTENTS:
    INTENT_API_ROUTING.setdefault(_pa_intent, dict(_OVERRIDE_DOMAIN_BASE))


# Required Headers for CVS API
REQUIRED_HEADERS = {
    "x-correlation-id": "W3C Trace Context identifier",
    "x-consumerAppName": "Consumer application name",
    "x-clientRefId": "Client reference ID"
}


# Helper Functions
def get_api_config(intent: str):
    """Get API configuration for an intent"""
    return INTENT_API_ROUTING.get(intent, {
        "api_endpoint": None,
        "requires_llm": True
    })


def get_domain_for_intent(intent: str) -> str:
    """
    Return the high-level domain for an intent.

    Currently only used to detect 'claim_history_search' intents so the
    LangGraph router can dispatch them to the member-history search node
    instead of the standard single-claim tool node.
    """
    cfg = INTENT_API_ROUTING.get(intent) or {}
    return cfg.get("domain", "")


def is_claim_history_search_intent(intent: str) -> bool:
    """True if the intent should be routed to the claims-search pipeline."""
    return get_domain_for_intent(intent) == "claim_history_search"


def is_override_domain_intent(intent: str) -> bool:
    """True if the intent should be routed to the Overrides_api pipeline (PA lookup)."""
    return get_domain_for_intent(intent) == "override_domain"


def requires_api(intent: str) -> bool:
    """Check if intent requires API call"""
    config = get_api_config(intent)
    return config.get("api_endpoint") is not None


def get_fallback_api(intent: str) -> str:
    """Get fallback API for an intent"""
    config = get_api_config(intent)
    fallback = config.get("fallback_api")
    if fallback:
        return ENDPOINTS.get(fallback)
    return None


def get_required_headers():
    """Get required headers for CVS API"""
    return REQUIRED_HEADERS.copy()

