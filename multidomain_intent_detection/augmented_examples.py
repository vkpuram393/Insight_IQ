"""
Multidomain Intent Detection — Augmented Training Examples
============================================================

Single source of truth for augmented training examples used to bridge
the gap between generic VamsiSir.py templates and real-world test query
phrasing patterns.

These examples are used by both:
  - multidomain_intent_detection/training.py
  - Intent_detection_system/intent_detection_v3.py

Organization:
  1. CONFUSION-PAIR TARGETED — addresses top misclassification patterns
  2. CAP_API ANCHORS — single-claim operations with real claim numbers
  3. CLAIM_HISTORY_SEARCH ANCHORS — multi-claim search/filter queries
  4. GENERAL — greetings, out-of-scope
  5. MEMBER_DOMAIN ANCHORS — member demographics, coverage, identifiers
  6. OVERRIDE_DOMAIN ANCHORS — PA configuration and management fields
  7. BENEFITS_API ANCHORS — plan summary, history, finder

When adding new examples:
  - Add to the appropriate section below
  - Include 5-10 examples per intent
  - Use real-world phrasing (not just template rewrites)
  - For claim-related intents, include claim numbers in examples
    (the normalizer strips them before embedding)
"""

from typing import Dict, List

AUGMENTED_EXAMPLES: Dict[str, List[str]] = {

    # ══════════════════════════════════════════════════════════════════════
    # 1. CONFUSION-PAIR TARGETED EXAMPLES
    # ══════════════════════════════════════════════════════════════════════
    # These address the top misclassification patterns identified from
    # evaluation results. Each block adds examples with phrasing that the
    # ensemble was confusing with a neighbor intent.

    # ── approval_info: confused with prior_auth_info & claim_status ─────
    "approval_info": [
        "Approval status and messages for claim 222492018072002 sequence 001.",
        "PA approval details for claim 253603736282009 sequence 001.",
        "PA type and approval for claim 201990819198000 sequence 001.",
        "Authorization details for claim 220133725669000 sequence 001.",
        "What is the approval status of claim 132435151040074 sequence 001?",
        "Show the approval logic and overrides for this claim.",
        "What plan overrides led to the approval of this claim?",
        "What BPG configuration approved this claim?",
        "Comprehensive summary of claim approval and overrides.",
        "Adjudication outcome: what overrides were applied to approve?",
    ],

    # ── rejection_reasons: confused with help & claim_status ────────────
    "rejection_reasons": [
        "Steps to avoid claim rejection for claim 132435151040074 sequence 001.",
        "Guidelines for avoiding rejections for claim 201990819198000 sequence 001.",
        "Instructions to prevent denials for claim 221775171449000 sequence 003.",
        "Is claim 180571470939000 sequence 001 approved or denied?",
        "Claim 240051073172000 sequence 001: approval or rejection?",
        "How to resolve the rejection on this claim.",
        "What steps should I take to overturn this denial?",
        "Options to fix a denied claim.",
    ],

    # ── ClaimNum: confused with claim_status & rx_details ───────────────
    "ClaimNum": [
        "260302639954275",
        "claim number 260302639954275",
        "Look up claim 480517832091643.",
        "Retrieve claim number 710924356178502.",
        "Find claim 392081547623918.",
        "Get claim number 853746291035824.",
        "claim 641285739401562",
        "Pull up claim number 205638914720483.",
    ],

    # ── compound_info: confused with pricing_info ───────────────────────
    "compound_info": [
        "Show the compound medication details for this claim.",
        "What are the individual ingredient costs in the compound?",
        "Display the MIC breakdown for this compound claim.",
        "Is this a compounded medication? Show ingredients.",
        "Compound formulation details and ingredient list for this claim.",
        "Funded versus unfunded compound ingredient costs.",
    ],

    # ── settlement_info (single claim) vs Settlement (search) ──────────
    "settlement_info": [
        "Settlement details for claim 220133725669000 sequence 001.",
        "Settlement information for claim 222492018072002 sequence 001.",
        "Settlement report for claim 222492117457002 sequence 001.",
        "Settlement status for claim 241774768475148 sequence 003.",
        "Settlement summary for claim 242831720377166 sequence 002.",
        "Settlement feedback for claim 243122443413000 sequence 001.",
        "Response information for claim 250023213779000 sequence 001.",
        "What settlement codes were sent back to the pharmacy for this claim?",
        "Show the pharmacy feedback codes for this specific claim.",
    ],

    # ── beneficiary_info: confused with approval_info & member_coverage ─
    "beneficiary_info": [
        "What benefit phase is the member in for this claim?",
        "Show the accumulation status for the member on this claim.",
        "Display the coverage tier used for this claim.",
        "What is the member's benefit type for this claim?",
        "Are medical dollars included in the member's accumulation?",
        "Linked LOE information for the member on this claim.",
    ],

    # ── greeting: confused with out_of_scope ───────────────────────────
    "greeting": [
        "Hello", "Hi there", "Welcome", "Hiya",
        "Hello, how are you?", "Hi, good to see you",
        "Hey", "What's up", "Good morning", "Good afternoon",
        "Yo!", "Greetings", "Howdy",
    ],

    # ── out_of_scope: confused with greeting ───────────────────────────
    "out_of_scope": [
        "What is the weather today?", "Tell me a joke.",
        "Who won the Super Bowl?", "How do I cook pasta?",
        "What is the meaning of life?",
        "Tell me about the latest iPhone.",
        "Calculate my taxes for 2025.",
        "Book me a flight to Hawaii.",
        "asdfghjkl qwertyuiop",
        "lorem ipsum dolor sit amet",
    ],

    # ══════════════════════════════════════════════════════════════════════
    # 2. CAP_API ANCHORS (single-claim operations)
    # ══════════════════════════════════════════════════════════════════════

    # ── generic_availability: confused with Generic & daw_info ─────────
    "generic_availability": [
        "Are there generic alternatives for the drug on this claim?",
        "Show the therapeutic equivalents for this prescription.",
        "Generic availability information for claim 201990819198000 sequence 001.",
        "Generic substitutes for claim 220992183436835 sequence 002.",
        "Generic alternative details for claim 230381673488000 sequence 001.",
        "Substitute drug information for claim 231865207544000 sequence 001.",
        "List the cheaper generic options for the medication on this claim.",
        "What formulary alternatives exist for this drug?",
    ],

    "prescriber_info": [
        "Prescriber details for claim 132435151040074 sequence 001.",
        "Physician information for claim 150692388845000 sequence 001.",
        "Doctor's name for claim 160060096136030 sequence 001.",
        "Prescriber NPI for claim 191406379285000 sequence 001.",
        "Who prescribed the medication on claim 130041467416065 sequence 001?",
        "Which physician wrote the prescription for claim 180571470939000?",
        "Prescribing physician information for claim 221663541811000 sequence 001.",
        "Prescriber NPI and name for claim 221775171449000 sequence 003.",
        "Who ordered the medication on claim 221172865083001 sequence 001?",
        "Ordering provider information for claim 230624075311000 sequence 002.",
    ],
    "audit_info": [
        "When was claim 132435151040074 sequence 001 first created?",
        "Claim 201503823714118 sequence 001 add date and change date.",
        "Who last modified claim 201752592251000 sequence 001 and when?",
        "What is the creation timestamp of claim 211263773300000 sequence 004?",
        "When was claim 221172865083001 sequence 001 added to the system?",
    ],
    "reversal_info": [
        "R&R information for claim 242905816136000 sequence 001.",
        "R&R status for claim 242905816136000 sequence 001.",
        "R&R report for claim 253603736282009 sequence 001.",
        "Was claim 231181462825000 sequence 001 reversed?",
        "Claim modifications for claim 260021649904000 sequence 005.",
    ],
    "claim_status": [
        "What is the current status of claim 130041467416065 sequence 001?",
        "Is claim 220133725669000 sequence 001 paid, rejected, or pending?",
        "Quick status check on claim 230381673488000 sequence 001.",
        "What was the result of processing claim 230624075311000 sequence 002?",
        "Adjudication outcome for claim 191406379285000 sequence 001.",
    ],
    "pricing_info": [
        "What is the copay on claim 132435151040074 sequence 001?",
        "Show the pricing breakdown for claim 220133725669000 sequence 001.",
        "What did the patient pay on this specific claim?",
        "Ingredient cost and fees for claim 191406379285000 sequence 001.",
        "Copay calculation steps for this claim.",
    ],
    "pharmacy_info": [
        "Which pharmacy dispensed claim 132435151040074 sequence 001?",
        "Pharmacy details for claim 220133725669000 sequence 001.",
        "Where was this specific claim filled?",
        "Dispensing pharmacy name for this claim.",
        "Store location for claim 191406379285000 sequence 001.",
    ],

    # ══════════════════════════════════════════════════════════════════════
    # 3. CLAIM_HISTORY_SEARCH ANCHORS (search/filter multiple claims)
    # ══════════════════════════════════════════════════════════════════════

    "Pricing": [
        "How much did the member pay for METFORMIN across all fills?",
        "Show me the total spent on ATORVASTATIN prescriptions.",
        "What was the copay trend for LISINOPRIL fills over time?",
        "Compare costs across multiple SERTRALINE claims.",
        "List the pricing for all GABAPENTIN claims this year.",
    ],
    "Settlement": [
        "Show claims with settlement code 358.",
        "Filter by settlement code 001 across all claims.",
        "Which claims returned settlement code 425?",
        "List fills that received settlement code 310.",
        "Retrieve claims with pharmacy settlement response 200.",
    ],
    "Pharmacy": [
        "Show claims filled at CVS PHARMACY 00610.",
        "List fills dispensed by WALGREENS 04528.",
        "Retrieve claims from RITE AID 11237.",
        "Which fills were filled at TARGET PHARMACY 01893?",
        "Give me claims processed at WALMART PHARMACY 10340.",
    ],
    "Prescriber": [
        "Show claims by prescriber NOEUV.",
        "List claims written by Dr. PATEL.",
        "Retrieve fills prescribed by NPI 1234567890.",
        "Which claims were ordered by prescriber SMITH?",
        "Display claims from prescriber Dr. JOHNSON.",
    ],
    "Status": [
        "Show all rejected claims for this member.",
        "List claims in paid status this year.",
        "Which claims are currently pending?",
        "Display all denied claims across all drugs.",
        "Give me claims in reversed status.",
    ],
    "RejectCode": [
        "Show claims with reject code 79.",
        "Filter claims by rejection code 75.",
        "Which claims have NCPDP reject code MR?",
        "List claims rejected under code 76.",
        "Retrieve claims with reject code 70.",
    ],
    "PriorAuth": [
        "Which claims required prior authorization?",
        "Show fills that went through a PA process.",
        "List claims where PA was approved.",
        "Retrieve prescriptions with an active prior auth on file.",
        "Display PA-approved claims for specialty drugs.",
    ],

    # ══════════════════════════════════════════════════════════════════════
    # 4. GENERAL (greeting & out_of_scope are in section 1 above)
    # ══════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════
    # 5. MEMBER_DOMAIN ANCHORS
    # ══════════════════════════════════════════════════════════════════════

    "member_coverage": [
        "Does this member have active coverage as of today?",
        "Show me the coverage eligibility windows for this member.",
        "What are the eligibility dates for member John Doe?",
        "When does this member's coverage begin and end?",
        "Is member 555123456 eligible right now?",
    ],
    "member_hierarchy": [
        "Which client does this member belong to?",
        "Show me the CAG hierarchy for this member.",
        "What client account group is this member under?",
        "Display the hierarchy information for this member.",
        "Give me the client and group assignment for this member.",
    ],
    "benefit_reset_date": [
        "What is the benefit reset date for this member?",
        "When does the benefit year reset for this member?",
        "Tell me when the accumulators reset for this member.",
        "Show me the plan year reset date for this member.",
        "When do the deductible and OOP accumulators reset?",
    ],
    "medicare_coverage": [
        "Does this member have Part D coverage?",
        "Is this member enrolled in Medicare?",
        "Show the Medicare Part D status for this member.",
        "Is this member a Medicare beneficiary?",
        "Tell me the Medicare coverage status for this member.",
    ],
    "lics_status": [
        "Is this member LICS?",
        "Does this member qualify for low income subsidy?",
        "Show the LICS status for this member.",
        "What LICS level is assigned to this member?",
        "Tell me if this member is receiving low income cost sharing.",
    ],
    "cvs_id_lookup": [
        "What is the CVS ID for this member?",
        "Show the CVS ID associated with this member.",
        "Retrieve the CVS ID for member John Doe.",
        "Give me the CVS identifier for this member.",
        "Look up the CVS ID for this member.",
    ],
    "alternate_ids": [
        "List all alternate IDs for this member.",
        "Show the alternate identifiers on file for this member.",
        "What alternate IDs are assigned to this member?",
        "Retrieve all alternate member IDs for this member.",
        "Give me all alternate IDs associated with this member.",
    ],
    "member_demographics": [
        "What is the full name and date of birth for this member?",
        "Show the gender recorded for this member.",
        "Display the member's person code and relationship code.",
        "Retrieve the demographic profile for this member.",
        "Give me the member's first name, last name, and DOB.",
    ],
    "member_contact_info": [
        "What is the email address on file for this member?",
        "Show the mailing address for this member.",
        "Display the primary phone number for this member.",
        "Retrieve the member's postal address including city and zip.",
        "Give me the contact details including email and phone.",
    ],
    "member_eligibility_copay": [
        "What is the brand copay set on this member's eligibility?",
        "Show the generic copay amount for this member.",
        "Display the copay3 and copay4 values from eligibility.",
        "Retrieve all four copay fields for this member.",
        "What are the copay amounts assigned to this member's plan?",
    ],
    "member_transition_status": [
        "What is the transition status for this member?",
        "Show the transition start date from the eligibility record.",
        "Is this member currently in a transition period?",
        "Display the memberTransition status for this member.",
        "When did the transition period start for this member?",
    ],
    "member_dur_config": [
        "What is the DUR configuration key for this member?",
        "Show the drug utilization review process flag.",
        "Is DUR processing enabled for this member?",
        "Display the drugUtilizationReviewKey for this member.",
        "Retrieve the member DUR review key and process flag.",
    ],
    "member_mbi_number": [
        "What is the MBI number for this member?",
        "Show the Medicare Beneficiary Identifier on file.",
        "Retrieve the MBI from the Medicare Part D record.",
        "Display the mbiNumber for this member.",
        "What MBI number is assigned to this member?",
    ],
    "member_caretaker_info": [
        "Show the caretaker information for this Medicare member.",
        "Who is the caretaker on file for this member?",
        "Display the caretaker name and address from Part D.",
        "Is there a caretaker assigned to this member?",
        "Retrieve the caretaker details including city and state.",
    ],
    "member_language_pref": [
        "What is the language preference for this member?",
        "Show the member language code on file.",
        "Display the preferred language setting for this member.",
        "What language is set for communications with this member?",
        "Retrieve the mbrLangCode from the member base record.",
    ],
    "member_discount_program": [
        "What discount program is assigned to this member?",
        "Show the discount program type on the member record.",
        "Is this member enrolled in a discount program?",
        "Display the discountProgramType for this member.",
        "Retrieve the discount program details for this member.",
    ],
    "member_override_plan": [
        "Does this member have an override plan on file?",
        "Show the member override plan ID from eligibility.",
        "Is an override plan configured for this member?",
        "Display the memberOverridePlan from the eligibility record.",
        "What override plan is assigned to this member's eligibility?",
    ],

    # ══════════════════════════════════════════════════════════════════════
    # 6. OVERRIDE_DOMAIN ANCHORS (PA configuration and management)
    # ══════════════════════════════════════════════════════════════════════

    "pa_summary": [
        "Give me a summary of this prior authorization.",
        "Summarize the key details of this PA.",
        "Show the most important fields on this PA.",
        "Display a high-level overview of this prior authorization.",
        "Provide the PA summary including effective dates and drug coverage.",
    ],
    "pa_override_reject": [
        "Will this PA override a reject 75 PA required?",
        "Does this PA handle reject code 75?",
        "Will this prior authorization bypass a reject 70 non-formulary?",
        "Does this PA override reject 70 plan exclusion?",
        "Show me which reject codes this PA can override.",
    ],
    "pa_field_help": [
        "What does the PA type field do?",
        "Explain the purpose of the effective date field on a PA.",
        "What is the GPI override field used for?",
        "Describe what the PA status indicator means.",
        "What does the quantity limit override field do on this PA?",
    ],
    "pa_copay_pricing": [
        "Does this copay override influence the price?",
        "How does the copay on this PA affect pricing?",
        "Will the PA copay change the member's out-of-pocket cost?",
        "Show me how the copay override impacts the final price.",
        "Does the copay field on this PA modify the claim price?",
    ],
    "pa_drug_coverage": [
        "What drugs will this PA cover?",
        "Show me the drug list covered by this prior authorization.",
        "Which medications are included under this PA?",
        "List the drugs that this PA authorizes.",
        "Display the GPI range covered by this PA.",
    ],
    "pa_claim_usage": [
        "How many claims used this PA?",
        "Show the claim count for this prior authorization.",
        "How many times has this PA been applied to claims?",
        "Display the number of claims processed under this PA.",
        "Retrieve the claim usage count for this PA.",
    ],
    "pa_reason_code": [
        "What is the reason code on this PA?",
        "Show the reason code assigned to this prior authorization.",
        "Is the reason code U1 or LC on this PA?",
        "Which PAs have reason code OD?",
        "Display the reason code for this PA.",
    ],
    "pa_effective_dates": [
        "What are the effective dates for this PA?",
        "Show the start and end dates on this prior authorization.",
        "When does this PA expire?",
        "Is this PA currently within its effective period?",
        "Display the dateBegin and dateEnd for this PA.",
    ],
    "pa_agent_code": [
        "What is the agent code on this PA?",
        "Show the agent code assigned to this prior authorization.",
        "Who created this PA based on the agent code?",
        "Which PAs were created by agent code C?",
        "Display the agent code for each PA on this member.",
    ],
    "pa_ignore_status": [
        "What is the ignore status code on this PA?",
        "Show the ignore status for this prior authorization.",
        "Is the ignore status code set to Y on this PA?",
        "Which PAs have ignore status P?",
        "Display the ignoreStatusCode for this PA.",
    ],
    "pa_specialty_rx_override": [
        "Does this PA override the specialty prescription reject?",
        "Show the specialty Rx override indicator for this PA.",
        "Is the specialty Rx reject indicator enabled on this PA?",
        "Which PAs have the specialty prescription override?",
        "Display the overrideSpecialtyPrescriptionRejectIndicator.",
    ],
    "pa_clinical_admin_code": [
        "What is the clinical administration code on this PA?",
        "Show the clinical admin code for this PA.",
        "Is there a clinical administration code set on this PA?",
        "Which PAs have clinical admin code C?",
        "Display the clinicalAdministrationCode for this PA.",
    ],
    "pa_transform_care": [
        "What is the transform care type on this PA?",
        "Show the transform care setting for this PA.",
        "Is there a transform care type configured on this PA?",
        "Display the transformCare type for this prior authorization.",
        "Which PAs have a transform care type assigned?",
    ],
    "pa_follow_me_logic": [
        "Is follow me logic enabled on this PA?",
        "Show the follow me logic indicator for this PA.",
        "Does this PA use follow me logic?",
        "Which PAs have the follow me indicator set to true?",
        "Display the followMeLogicIndicator for this PA.",
    ],
    "pa_drug_type_indicator": [
        "What is the authorized drug type on this PA?",
        "Show the drug type indicator for this PA.",
        "Is the drug type set to G for GPI or N for NDC?",
        "Which PAs use NDC-based drug matching?",
        "Display the authorizedDrugType for this PA.",
    ],
    "pa_modification_history": [
        "When was this PA last modified?",
        "Show the modification date and time for this PA.",
        "Which PA was most recently modified?",
        "Display the modifyDateTime for this prior authorization.",
        "How recently was this PA updated?",
    ],

    # ══════════════════════════════════════════════════════════════════════
    # 7. BENEFITS_API ANCHORS
    # ══════════════════════════════════════════════════════════════════════

    "plan_summary": [
        "Show the current benefit plan overview for this member.",
        "Give me a snapshot of the member's active plan.",
        "What does this member's benefit plan cover?",
        "Display the current plan summary.",
        "Summarize the active benefit plan for this member.",
    ],
    "plan_history": [
        "Show the change log of this member's benefit plan.",
        "What modifications have been made to the plan over time?",
        "List past revisions of the benefits plan.",
        "Display the audit trail of plan changes.",
        "Give me the timeline of updates to the plan.",
    ],
    "plan_finder": [
        "Help me locate an available benefit plan.",
        "Search for plans that match this client.",
        "Which plans are offered to this member's group?",
        "Find a matching benefits plan.",
        "Look up what plans exist for this client.",
    ],
}
