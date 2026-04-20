"""
Test script for the response_trimmer + llm_formatter pipeline.

Demonstrates:
  1. Raw API response size vs trimmed size
  2. Pre-filtering by different user queries
  3. Compact text output ready for LLM
"""
import json
import sys
import os

# Add parent directory to path so Claims_search_api package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Claims_search_api.response_trimmer import trim_api_response, trim_single_claim_response
from Claims_search_api.llm_formatter import format_claims_for_llm, format_claims_as_compact_json


# ---------------------------------------------------------------------------
# Paste/load your API response here for quick testing
# ---------------------------------------------------------------------------
SAMPLE_RESPONSE = {
    "success": True,
    "message": "Claims retrieved successfully (14 claims, all data returned)",
    "totalCount": 14,
    "claims": [
        {
            "claimInformation": {
                "claimNumber": "260358603285176", "claimSequenceNumber": "1",
                "claimStatus": "P", "claimStatusDescription": "Paid",
                "fillDate": "2026-02-05", "cobIndicator": "01",
                "originationFlag": "T", "originationFlagDescription": "Electronic transaction",
                "otherCoverage": "0", "stcob": "P", "type": None,
                "adjudicationEnvironment": "Production", "locationCode": "00",
                "medD": None, "compound": "N", "pharmacyNetwork": "MCVRTL",
                "speciality": "N", "reimbursementType": "P",
                "reimbursementTypeDescription": "Pharmacy",
                "extractStatus": None, "extractStatusDescription": "",
                "adminType": None, "adminTypeDescription": "",
                "participationCode": None, "participationCodeDescription": "",
                "governmentClaimType": None, "rnR": "No",
                "dispenseAsWritten": "0", "quantity": "30.0",
                "daysSupplied": "30", "scc": None,
                "claimType": "Point of Sale", "coverageType": "STCOB Primary",
                "tag": "STCOB Primary",
                "secondaryClaimNumber": "260358603304176",
                "secondaryClaimSequence": "1",
                "secondaryClaimStatus": None,
                "secondaryClaimStatusDescription": None,
                "primaryClaimNumber": None, "primaryClaimSequence": None,
                "primaryClaimStatus": None, "primaryClaimStatusDescription": None,
                "addDate": "2026-02-04", "addTime": "23:53:52",
                "addUser": "APCSOWN", "addProgram": "RCNCP032",
                "changeDate": "2026-02-12", "changeTime": "14:50:05",
                "changeUser": "MASKED", "changeProgram": "RCNCP032",
                "trackingnumber": None
            },
            "member": {
                "memberId": "4807045053", "lastName": "PROANOCONTR",
                "firstName": "ESNA", "middleInitial": None,
                "dateOfBirth": "1976-07-05", "gender": "1",
                "genderDescription": "Male", "age": "49",
                "relationship": "1", "relationshipDescription": "Card Holder",
                "eligibilityFrom": None, "eligibilityThru": None,
                "clientId": "COMM", "carrierId": "25BP",
                "accountId": "01", "groupId": "001",
                "basePlanId": None, "cardholderId": "4807045053",
                "clientPlanCode": "CCA264861", "finalPlanCode": "LICS2",
                "personCode": "00", "clientPlanId": "CCA264861",
                "planId": "LICS2", "ssn": None, "memberPhone": None,
                "memberState": None, "memberProductCode": None,
                "memberRiderCode": None
            },
            "drug": {
                "gpi": "28100010100315", "productNdc": "33342-0395-44",
                "productName": "LEVOTHYROXINE SODIUM",
                "genericIndicator": "Y", "productIDQualifier": "03",
                "productIDQualifierDescription": "National Drug Code (NDC)",
                "manufacturer": "MACLEODS", "multiSourceIndicator": "Y",
                "multiSourceIndicatorDescription": "Generic",
                "metricQuantity": None, "unitOfMeasure": None,
                "productSelectionCode": "0",
                "productSelectionCodeDescription": "NO PROD SELECTN INDICATED"
            },
            "pricing": {
                "patientPay": "1.54", "clientPay": None,
                "drugCostSubmitted": None, "drugCostApproved": None,
                "dispensingFeeSubmitted": None, "dispensingFeeApproved": None,
                "taxSubmitted": None, "taxApproved": None,
                "otherFeeSubmitted": None, "otherFeeApproved": None,
                "opapSubmitted": None, "opapApproved": None,
                "opprSubmitted": None, "opprApproved": None,
                "otherAmountSubmitted": None, "otherAmountApproved": None,
                "patientPaySubmitted": None, "patientPayApproved": None,
                "amountDueSubmitted": None, "amountDueApproved": None,
                "ucwSubmitted": None, "ucwApproved": None
            },
            "prescription": {
                "prescriberID": "2800855639",
                "prescriberFirstName": "EVATOR", "prescriberLastName": "NOEUV",
                "prescriberQualifier": "01",
                "prescriberQualifierDescription": "National Provider (NPI)",
                "submitDate": "2026-02-04", "refillNumber": "08",
                "pharmacyNcpdp": "2224018", "pharmacyQualifier": "07",
                "pharmacyQualifierDescription": "NCPDP Provider ID",
                "pharmacyName": "CVS PHARMACY 00610",
                "pharmacyPhone": "508-999-0790",
                "pharmacyCity": "FAIRHAVEN", "pharmacyState": "MA",
                "pharmacyZip": "02719", "pharmacyType": "Retail",
                "rxNumber": "9751232", "rxNumberQualifier": "1",
                "rxNumberQualifierDescription": "RX BILLING",
                "personCode": "00", "transactionCode": "B1",
                "versionReleaseNumber": None, "binNumber": "004336",
                "processControlNumber": "MEDDADV", "groupNumber": "RX25BP",
                "reversalDate": None, "diagnosisCodeQualifier": "",
                "submittedDiagnosisCodeIndicator": None,
                "clarificationCodes": None
            },
            "priorAuthorization": {
                "number": "260358603285176", "paIndicator": None,
                "reasonCode": None, "reasonDescription": None,
                "layered": None, "type": None, "typeDescription": None
            },
            "overrides": {
                "paType": None, "paNumber": None, "paReasonCode": None,
                "paLayered": None, "priorAuthorizationUsed": None,
                "submissionClarificationCode": "No",
                "drugutiliztionReview": "No", "drugListUsed": "No",
                "smartPriorAuthorizationUsed": "No"
            },
            "messages": {
                "rejectCodes": None,
                "approvedMessages": [{"code": "023", "description": None}],
                "settlementCodes": None, "messages": None
            },
            "audit": {"addDate": None, "addTime": None,
                       "changeDate": "2026-02-12", "changeTime": "14:50:05"},
            "additionalDetails": None,
            "pricingAdditionalDTO": None
        },
        {
            "claimInformation": {
                "claimNumber": "260302639954275", "claimSequenceNumber": "1",
                "claimStatus": "R", "claimStatusDescription": "Rejected",
                "fillDate": "2026-01-30", "cobIndicator": "01",
                "originationFlag": "T", "originationFlagDescription": "Electronic transaction",
                "otherCoverage": "0", "stcob": None, "type": None,
                "adjudicationEnvironment": "Production", "locationCode": "00",
                "medD": None, "compound": "N", "pharmacyNetwork": "MCVEDS",
                "speciality": "N", "reimbursementType": "P",
                "reimbursementTypeDescription": "Pharmacy",
                "extractStatus": None, "extractStatusDescription": "",
                "adminType": None, "adminTypeDescription": "",
                "participationCode": None, "participationCodeDescription": "",
                "governmentClaimType": None, "rnR": "No",
                "dispenseAsWritten": "0", "quantity": "9.0",
                "daysSupplied": "90", "scc": None,
                "claimType": "Point of Sale", "coverageType": "Primary",
                "tag": None,
                "secondaryClaimNumber": None, "secondaryClaimSequence": None,
                "secondaryClaimStatus": None, "secondaryClaimStatusDescription": None,
                "primaryClaimNumber": None, "primaryClaimSequence": None,
                "primaryClaimStatus": None, "primaryClaimStatusDescription": None,
                "addDate": "2026-01-30", "addTime": "07:19:59",
                "addUser": "APCSOWN", "addProgram": "RCNCP032",
                "changeDate": "2026-02-12", "changeTime": "14:50:05",
                "changeUser": "MASKED", "changeProgram": "RCNCP032",
                "trackingnumber": None
            },
            "member": {
                "memberId": "4807045053", "lastName": "PROANOCONTR",
                "firstName": "ESNA", "middleInitial": None,
                "dateOfBirth": "1976-07-05", "gender": "1",
                "genderDescription": "Male", "age": "49",
                "relationship": "1", "relationshipDescription": "Card Holder",
                "eligibilityFrom": None, "eligibilityThru": None,
                "clientId": "COMM", "carrierId": "25BP",
                "accountId": "01", "groupId": "001",
                "basePlanId": None, "cardholderId": "4807045053",
                "clientPlanCode": "CCA264861", "finalPlanCode": "26C1#BAACR",
                "personCode": "00", "clientPlanId": "CCA264861",
                "planId": "26C1#BAACR", "ssn": None, "memberPhone": None,
                "memberState": None, "memberProductCode": None,
                "memberRiderCode": None
            },
            "drug": {
                "gpi": "97202012046300", "productNdc": "08627-0077-01",
                "productName": "DEXCOM G7 SENSOR",
                "genericIndicator": "N", "productIDQualifier": "02",
                "productIDQualifierDescription": "Health Related Item (HRI)",
                "manufacturer": "DEXCOM", "multiSourceIndicator": "N",
                "multiSourceIndicatorDescription": "Single-Source Not Generic",
                "metricQuantity": None, "unitOfMeasure": None,
                "productSelectionCode": "0",
                "productSelectionCodeDescription": "NO PROD SELECTN INDICATED"
            },
            "pricing": {
                "patientPay": None, "clientPay": None,
                "drugCostSubmitted": None, "drugCostApproved": None,
                "dispensingFeeSubmitted": None, "dispensingFeeApproved": None,
                "taxSubmitted": None, "taxApproved": None,
                "otherFeeSubmitted": None, "otherFeeApproved": None,
                "opapSubmitted": None, "opapApproved": None,
                "opprSubmitted": None, "opprApproved": None,
                "otherAmountSubmitted": None, "otherAmountApproved": None,
                "patientPaySubmitted": None, "patientPayApproved": None,
                "amountDueSubmitted": None, "amountDueApproved": None,
                "ucwSubmitted": None, "ucwApproved": None
            },
            "prescription": {
                "prescriberID": "2800855639",
                "prescriberFirstName": "EVATOR", "prescriberLastName": "NOEUV",
                "prescriberQualifier": "01",
                "prescriberQualifierDescription": "National Provider (NPI)",
                "submitDate": "2026-01-30", "refillNumber": "01",
                "pharmacyNcpdp": "2224018", "pharmacyQualifier": "07",
                "pharmacyQualifierDescription": "NCPDP Provider ID",
                "pharmacyName": "CVS PHARMACY 00610",
                "pharmacyPhone": "508-999-0790",
                "pharmacyCity": "FAIRHAVEN", "pharmacyState": "MA",
                "pharmacyZip": "02719", "pharmacyType": "Retail",
                "rxNumber": "0892914", "rxNumberQualifier": "1",
                "rxNumberQualifierDescription": "RX BILLING",
                "personCode": "00", "transactionCode": "B1",
                "versionReleaseNumber": None, "binNumber": "004336",
                "processControlNumber": "MEDDADV", "groupNumber": "RX25BP",
                "reversalDate": None,
                "diagnosisCodeQualifier": "ICD10 - INTL CLS DISEASES",
                "submittedDiagnosisCodeIndicator": "E1129",
                "clarificationCodes": None
            },
            "priorAuthorization": {
                "number": "260302639954275", "paIndicator": "1009TY06B",
                "reasonCode": None, "reasonDescription": None,
                "layered": None, "type": None, "typeDescription": None
            },
            "overrides": {
                "paType": "MB", "paNumber": None, "paReasonCode": None,
                "paLayered": None, "priorAuthorizationUsed": "Yes",
                "submissionClarificationCode": "No",
                "drugutiliztionReview": "No", "drugListUsed": "No",
                "smartPriorAuthorizationUsed": "No"
            },
            "messages": {
                "rejectCodes": [{"code": "79", "description": "Refill Too Soon"}],
                "approvedMessages": None,
                "settlementCodes": [{"code": "358", "description": "REFILL TOO SOON"}],
                "messages": [
                    "NEXT AVAILABLE FILL DATE 20260314",
                    "LAST FILL DT 20260105 FILLED AT PHARMACY",
                    "CVS PHARMACY 00610,PHONE #5089990790"
                ]
            },
            "audit": {"addDate": None, "addTime": None,
                       "changeDate": "2026-02-12", "changeTime": "14:50:05"},
            "additionalDetails": None,
            "pricingAdditionalDTO": None
        }
    ]
}


def main():
    raw_json = json.dumps(SAMPLE_RESPONSE, indent=None)
    print(f"=== RAW response size: {len(raw_json)} chars ===\n")

    # --- Trim only ---
    trimmed = trim_api_response(SAMPLE_RESPONSE)
    trimmed_json = json.dumps(trimmed, indent=None)
    print(f"=== TRIMMED response size: {len(trimmed_json)} chars ===")
    print(f"=== Reduction: {100 - (len(trimmed_json) / len(raw_json) * 100):.1f}% ===\n")

    # --- LLM text format (no query filter) ---
    print("=" * 60)
    print("LLM TEXT OUTPUT (all claims, no filter)")
    print("=" * 60)
    text_output = format_claims_for_llm(SAMPLE_RESPONSE, user_query=None)
    print(text_output)
    print(f"\n--- Text output size: {len(text_output)} chars ---\n")

    # --- LLM text format with query filter ---
    test_queries = [
        # Original queries
        "show me all claims with reject code 79",
        "give me all the claims for this member in january",
        # The 4 example queries from reference.txt
        "When was LEVOTHYROXINE taken last for this member?",
        "when was this LEVOTHYROXINE taken last?",
        "What was the last claim for this LEVOTHYROXINE?",
        "give me all the claims for this member in this month",
        "give me all the claims for this member last month",
        # Drug by name
        "give me the claims related to the drug DEXCOM G7 SENSOR",
        # Status filter
        "show me all rejected claims",
        # Pharmacy queries
        "show me claims filled at CVS PHARMACY 00610",
        "which claims were from pharmacy in FAIRHAVEN",
        # Prescriber queries
        "show claims by prescriber NOEUV",
        "claims for NPI 2800855639",
        # Pricing / cost queries
        "how much did the member pay for LEVOTHYROXINE?",
        "show me the cost details for all claims",
        # Rx number
        "show claim for rx number 9751232",
        # --- NEW SCENARIOS ---
        # Claim number lookup
        "show me claim number 260302639954275",
        # NDC lookup
        "find claims for NDC 33342-0395-44",
        # Manufacturer
        "show claims manufactured by MACLEODS",
        # Generic vs Brand
        "show all generic drug claims",
        "show brand name claims",
        # Refill
        "show all refills for this member",
        # Days supply
        "show claims with 90 day supply",
        # Prior authorization
        "which claims used prior authorization?",
        # Diagnosis code
        "show claims with diagnosis code E1129",
        # Settlement code
        "show claims with settlement code 358",
        # Pharmacy type
        "show retail pharmacy claims",
        # Plan
        "show claims under plan LICS2",
    ]

    for query in test_queries:
        print("=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)
        text_output = format_claims_for_llm(SAMPLE_RESPONSE, user_query=query)
        print(text_output)
        print(f"--- Text output size: {len(text_output)} chars ---\n")

    # --- Compact JSON format ---
    print("=" * 60)
    print("COMPACT JSON OUTPUT (all claims)")
    print("=" * 60)
    json_output = format_claims_as_compact_json(SAMPLE_RESPONSE)
    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)
    print(f"\n--- JSON output size: {len(json_output)} chars ---")


if __name__ == "__main__":
    main()
