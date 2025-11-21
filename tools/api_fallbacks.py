"""
Dynamic API Fallback Data Generator

This module provides fallback data when external APIs fail.
The fallback data is dynamically generated based on input parameters.

Usage:
    from tools.api_fallbacks import get_fallback_details, get_fallback_list
    
    details = get_fallback_details("253152732536005", "1")
    claim_list = get_fallback_list("253152732536005")
"""

import json
from typing import Dict, Any
from core.logger import get_logger

logger = get_logger(__name__)


def get_fallback_details(claim_number: str, claim_sequence: str) -> Dict[str, Any]:
    """
    Generate fallback claim details data dynamically.
    
    Args:
        claim_number: The claim number to use in fallback data
        claim_sequence: The claim sequence to use in fallback data
        
    Returns:
        Dict containing claim details in expected API format
    """
    logger.info(f"🔄 Generating fallback claim details for claimNumber={claim_number}, claimSequence={claim_sequence}")
    
    # Template with placeholders that will be replaced
    template = """
    {{
        "status": "success",
        "header": {{
          "xcorrelationid": "069d197a-1e14-453f-8cf4-63f74eb626ff",
          "xconsumerAppName": "PSS-MYCLAIMSPOC-CLAIM-MFE"
        }},
        "claimDetails": {{
          "primary": {{
            "medD": {{
              "accountId": "A-B7157401",
              "approvedDispensingFee": "0000010.00",
              "approvedIncentiveAmount": "0000002.00",
              "approvedIngredientCost": "0000116.24",
              "approvedTotalAmount": "0000128.24",
              "calculatedDispensingFee": "0000010.00",
              "calculatedIncentiveFeeAmount": "0000002.00",
              "calulatedIngredientCost": "0000116.24",
              "carrierId": "C-B715741",
              "character1": "1",
              "claimStatus": "R",
              "cobClaimIndicator": "43",
              "compoundCode": "1",
              "dawProductSelectionCode": "9",
              "finalPlanCode": "P71574@01",
              "gpiNumber": "39400010100310",
              "groupId": "G-B7157401",
              "submitDate": "20251003",
              "submittedBinNumber": "004336",
              "submittedDaysSupply": "10",
              "submittedFillNumber": "00",
              "submittedPrescriberId": "2293825677",
              "submittedPrescriberIdQl": "01",
              "submittedProductId": "00071015540",
              "submittedProductIdQualifier": "03",
              "submittedQuantityDispensed": "10.000",
              "submittedRxNumber": "28877181916",
              "submittedRxNumberQualifier": "1",
              "submittedServiceProviderId": "AP711",
              "submittedTransactionCode": "B1",
              "submittedVersionReleaseNumber": "D0"
            }},
            "accountId": "A-B7157401",
            "carrierId": "C-B715741",
            "claimStatus": "R",
            "submitDate": "20251003",
            "submittedProductId": "00071015540"
          }},
          "additionalDetails": {{
            "claimType": "RX",
            "cmsContractId": "S1234",
            "cmsPlanId": "001",
            "cobClaimIndicator": "43",
            "governmentClaimType": "MEDICAID",
            "planType": "COMMERCIAL",
            "submitDate": "2025-10-03",
            "partDDrugIndicator": "Y",
            "formularyId": "FORM001",
            "drugLists": [
              {{
                "accountId": "A-B7157401",
                "carrierId": "C-B715741",
                "clientCode": "AETNA",
                "gpiNumber": "39400010100310",
                "groupId": "G-B7157401",
                "listId": "GHC7157@01",
                "planCode": "P71574@01",
                "submittedProductId": "00071015540"
              }}
            ],
            "xrefDetails": [
              {{
                "benefitType": "RX",
                "clientCode": "AETNA",
                "clientPlanCode": "AET001",
                "planEffectiveDate": "2024-01-01",
                "planType": "COMMERCIAL",
                "fromDate": "2024-01-01",
                "thruDate": "2024-12-31"
              }}
            ]
          }},
          "linkedClaim": {{
            "stcob": {{
              "claimNumber": "{claim_number}",
              "claimSequence": "{claim_sequence}"
            }}
          }}
        }}
      }}
    """
    
    try:
        # Replace placeholders with actual values
        json_str = template.format(claim_number=claim_number, claim_sequence=claim_sequence)
        fallback_data = json.loads(json_str)
        logger.debug(f"✅ Successfully generated fallback claim details")
        return fallback_data
    except Exception as e:
        logger.error(f"❌ Failed to generate fallback claim details: {e}")
        # Return minimal valid structure on error
        return {
            "status": "error",
            "message": "Fallback data generation failed",
            "claimDetails": {}
        }


def get_fallback_list(claim_id: str, claim_sequence: str) -> Dict[str, Any]:
    """
    Generate fallback claim list data dynamically.
    
    Args:
        claim_id: The claim ID to use in fallback data
        claim_sequence: The claim sequence to use in fallback data
        
    Returns:
        Dict containing claim list in expected API format
    """
    logger.info(f"🔄 Generating fallback claim list for claimId={claim_id}, claimSequence={claim_sequence}")
    
    # Template with placeholders - using double braces {{}} to escape them in JSON,
    # and single braces {} for Python format placeholders
    template = """
    {{
        "claims": [
            {{
                "claimInformation": {{
                  "claimNumber": "{claim_id}",
                  "claimSequenceNumber": "{claim_sequence}",
                  "claimStatus": "P",
                  "claimStatusDescription": "Paid",
                  "fillDate": "2025-05-01",
                  "cobIndicator": "01",
                  "originationFlag": "T",
                  "originationFlagDescription": "Electronic transaction",
                  "otherCoverage": "0",
                  "stcob": null,
                  "type": null,
                  "locationCode": "00",
                  "medD": null,
                  "compound": "N",
                  "pharmacyNetwork": "GOVCLP",
                  "speciality": "N",
                  "reimbursementType": "P",
                  "reimbursementTypeDescription": "Pharmacy",
                  "extractStatus": null,
                  "extractStatusDescription": "",
                  "adminType": null,
                  "adminTypeDescription": "",
                  "participationCode": null,
                  "participationCodeDescription": "",
                  "governmentClaimType": null,
                  "rnR": "No",
                  "dispenseAsWritten": "1",
                  "quantity": "10.0",
                  "daysSupplied": "90",
                  "scc": null,
                  "claimType": "Point of Sale",
                  "coverageType": "Primary",
                  "secondaryClaimNumber": null,
                  "secondaryClaimSequence": null,
                  "secondaryClaimStatus": null,
                  "secondaryClaimStatusDescription": null,
                  "primaryClaimNumber": null,
                  "primaryClaimSequence": null,
                  "primaryClaimStatus": null,
                  "primaryClaimStatusDescription": null,
                  "addDate": "2025-11-11",
                  "addTime": "07:35:25",
                  "addUser": "Z340100",
                  "addProgram": "RCNCP051",
                  "changeDate": "2025-11-11",
                  "changeTime": "07:35:25",
                  "changeUser": "Z340100",
                  "changeProgram": "RCNCP051"
                }},
                "member": {{
                  "memberId": "78318GG3001",
                  "lastName": "CHLOE",
                  "firstName": "ROBERTS",
                  "middleInitial": null,
                  "dateOfBirth": "1980-01-01",
                  "gender": "2",
                  "genderDescription": "Female",
                  "age": "45",
                  "relationship": "1",
                  "relationshipDescription": "Card Holder",
                  "eligibilityFrom": null,
                  "eligibilityThru": null,
                  "clientId": "CAPENSION",
                  "carrierId": "25CY",
                  "accountId": "PERSPLTBASPPO",
                  "groupId": "AB01FJ",
                  "basePlanId": null,
                  "cardholderId": "78318GG3001",
                  "clientPlanCode": "CALPERPPO1",
                  "finalPlanCode": "CALP#NAAA",
                  "personCode": "01",
                  "clientPlanId": "CALPERPPO1",
                  "planId": "CALP#NAAA",
                  "ssn": null,
                  "memberPhone": null,
                  "memberState": null,
                  "memberProductCode": null,
                  "memberRiderCode": null
                }},
                "drug": {{
                  "gpi": "30402020000320",
                  "productNdc": "23155-0823-73",
                  "productName": "CABERGOLINE",
                  "genericIndicator": "Y",
                  "productIDQualifier": "03",
                  "productIDQualifierDescription": "National Drug Code (NDC)",
                  "manufacturer": "AVET PHARM",
                  "multiSourceIndicator": "Y",
                  "multiSourceIndicatorDescription": "Generic",
                  "metricQuantity": null,
                  "unitOfMeasure": null,
                  "productSelectionCode": "1",
                  "productSelectionCodeDescription": "SUB NOT ALLOWED BY PRESCR"
                }},
                "pricing": {{
                  "patientPay": "10.00",
                  "clientPay": null
                }},
                "prescription": {{
                  "prescriberID": "2397069099",
                  "prescriberFirstName": "AlbenITa",
                  "prescriberLastName": "ALKA",
                  "prescriberQualifier": "01",
                  "prescriberQualifierDescription": "National Provider (NPI)",
                  "submitDate": "2025-11-11",
                  "refillNumber": "01",
                  "pharmacyNcpdp": "0100052",
                  "pharmacyQualifier": "07",
                  "pharmacyQualifierDescription": "NCPDP Provider ID",
                  "pharmacyName": "SMITHERMANS PHARMACY",
                  "pharmacyPhone": "205-665-2574",
                  "pharmacyCity": "MONTEVALLO",
                  "pharmacyState": "AL",
                  "pharmacyZip": "35115",
                  "pharmacyType": "Retail",
                  "rxNumber": "67567875082",
                  "rxNumberQualifier": "1",
                  "rxNumberQualifierDescription": "RX BILLING",
                  "personCode": "01",
                  "transactionCode": "B1",
                  "versionReleaseNumber": null,
                  "binNumber": "004336",
                  "processControlNumber": "*",
                  "groupNumber": "*",
                  "reversalDate": null,
                  "diagnosisCodeQualifier": "",
                  "submittedDiagnosisCodeIndicator": null,
                  "clarificationCodes": null
                }},
                "priorAuthorization": {{
                  "number": "{claim_id}",
                  "paIndicator": null,
                  "reasonCode": null,
                  "reasonDescription": null,
                  "layered": null,
                  "type": null,
                  "typeDescription": null
                }},
                "overrides": {{
                  "paType": null,
                  "paNumber": null,
                  "paReasonCode": null,
                  "paLayered": null,
                  "priorAuthorizationUsed": null,
                  "submissionClarificationCode": "No",
                  "drugutiliztionReview": "No",
                  "drugListUsed": "No",
                  "smartPriorAuthorizationUsed": "No"
                }},
                "messages": {{
                  "rejectCodes": null,
                  "approvedMessages": null,
                  "settlementCodes": null,
                  "messages": null
                }},
                "audit": {{
                  "addDate": null,
                  "addTime": null,
                  "changeDate": "2025-11-11",
                  "changeTime": "07:35:25"
                }},
                "additionalDetails": null
              }}
            ]
    }}
    """
    
    try:
        # Replace placeholders with actual values
        json_str = template.format(claim_id=claim_id, claim_sequence=claim_sequence)
        fallback_data = json.loads(json_str)
        logger.debug(f"✅ Successfully generated fallback claim list")
        return fallback_data
    except Exception as e:
        logger.error(f"❌ Failed to generate fallback claim list: {e}")
        # Return minimal valid structure on error
        return {
            "claims": [],
            "message": "Fallback data generation failed"
        }

