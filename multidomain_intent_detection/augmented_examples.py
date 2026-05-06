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
        "Was a transition fill applied to this claim?",
        "TF eligibility and approval details for this fill.",
        "What transition fill type was used here?",
        "Show all plan-level overrides that approved this claim.",
        "What BPG settings triggered the approval?",
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
        "What NCPDP reject codes came back on this?",
        "Why was this claim denied?",
        "Explain the rejection reason.",
        "Show the failed edits on this claim.",
        "What reject codes were returned on this fill?",
        "Tell me the denial reason and resolution steps.",
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
        "Search for claim 132435151040074.",
        "Locate claim 220133725669000.",
    ],

    # ── compound_info: confused with pricing_info ───────────────────────
    "compound_info": [
        "Show the compound medication details for this claim.",
        "What are the individual ingredient costs in the compound?",
        "Display the MIC breakdown for this compound claim.",
        "Is this a compounded medication? Show ingredients.",
        "Compound formulation details and ingredient list for this claim.",
        "Funded versus unfunded compound ingredient costs.",
        "List all ingredients in this compound prescription.",
        "MIC information and ingredient breakdown for this compound.",
        "Is this a compounded drug? Show the formulation.",
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
        "What response code did the pharmacy receive on this specific claim?",
    ],

    # ── beneficiary_info: confused with approval_info & member_coverage ─
    "beneficiary_info": [
        "What benefit phase is the member in for this claim?",
        "Show the accumulation status for the member on this claim.",
        "Display the coverage tier used for this claim.",
        "What is the member's benefit type for this claim?",
        "Are medical dollars included in the member's accumulation?",
        "Linked LOE information for the member on this claim.",
        "What is the member's benefit phase and accumulation for this claim?",
        "Show deductible and coverage gap progress for this claim's member.",
        "Accumulator totals for this beneficiary on this specific claim.",
        # ── From failure analysis: coverage gap / YTD spend / catastrophic ─
        "How close is this member to reaching the coverage gap?",
        "Was the catastrophic coverage phase reached on this claim?",
        "What is the total year-to-date spend for this member?",
        "What are the year-to-date accumulations for this beneficiary?",
        "Coverage tier and eligibility details for this beneficiary.",
        "Has this member reached the coverage gap threshold?",
        "Show the deductible and TrOOP accumulation progress for this member.",
        "Current benefit phase based on this member's accumulations.",
        "What Phase of the Part D benefit is this member currently in?",
    ],

    # ── greeting: confused with out_of_scope ───────────────────────────
    "greeting": [
        "Hello", "Hi there", "Welcome", "Hiya",
        "Hello, how are you?", "Hi, good to see you",
        "Hey", "What's up", "Good morning", "Good afternoon",
        "Yo!", "Greetings", "Howdy",
        "Good evening", "Good day", "Morning", "Afternoon",
        "Evening", "Hey there", "Hello there",
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
        "What is the capital of France?",
        "How do I invest in stocks?",
        "Translate this to Spanish.",
    ],

    # ══════════════════════════════════════════════════════════════════════
    # 1b. WORST-PERFORMER TARGETED AUGMENTATION
    #     From accuracy report: intents with <80% hybrid accuracy
    # ══════════════════════════════════════════════════════════════════════

    # ── pa_field_help (30% accuracy!): confused with pa_ignore_status ──
    # Key: queries about "explain" or "what does X field do/mean"
    "pa_field_help": [
        "What does the PA type field do?",
        "Explain the purpose of the effective date field on a PA.",
        "What is the GPI override field used for?",
        "Describe what the PA status indicator means.",
        "What does the quantity limit override field do on this PA?",
        "Explain the meaning of the ignore status field on a PA.",
        "What does the follow me logic indicator mean?",
        "Help me understand what the agent code field represents.",
        "What is the purpose of the clinical admin code field?",
        "Explain what the specialty Rx override indicator does.",
        "Describe the function of the transform care type field.",
        "What does the drug type indicator field mean on this PA?",
        "Tell me what the reason code field is used for on a PA.",
        "I need an explanation of the PA copay override field.",
        "What does the authorized drug type field do?",
        "Explain the purpose of the PA modification date field.",
        "What is the meaning of the PA effective period field?",
        "Help me understand what the PA claim usage count represents.",
        "Describe what each field on this PA means.",
        "I need field-level documentation for this PA record.",
    ],

    # ── Prescriber (50% accuracy): confused with prescriber_info ───────
    # Key: "search/filter claims BY prescriber" (multi-claim search)
    # NOT: "get prescriber details FOR one claim" (prescriber_info)
    "Prescriber": [
        "Show claims by prescriber NOEUV.",
        "List claims written by Dr. PATEL.",
        "Retrieve fills prescribed by NPI 1234567890.",
        "Which claims were ordered by prescriber SMITH?",
        "Display claims from prescriber Dr. JOHNSON.",
        "Filter this member's claims by the prescribing doctor.",
        "Search all claims by this specific prescriber NPI.",
        "Show all fills written by this prescriber across the member's history.",
        "How many claims has this prescriber written for this member?",
        "Which prescribers wrote for this member and how many claims each?",
        "List all claims from prescriber NPI 9876543210 for this member.",
        "Filter the claim history to only show this doctor's prescriptions.",
        "Show me every claim ordered by Dr. WILLIAMS.",
        "Search claims by the prescriber who wrote this script.",
        "Pull all claims from a specific prescribing physician.",
    ],

    # ── pa_follow_me_logic (50%): confused with member_transition_status
    # Key: PA configuration field, NOT member eligibility transition
    "pa_follow_me_logic": [
        "Is follow me logic enabled on this PA?",
        "Show the follow me logic indicator for this PA.",
        "Does this PA use follow me logic?",
        "Which PAs have the follow me indicator set to true?",
        "Display the followMeLogicIndicator for this PA.",
        "Does this prior authorization follow the member across plan changes?",
        "Is the PA configured to follow the member when they change plans?",
        "Check if follow me is active on this PA override.",
        "Will this PA carry over if the member switches plans?",
        "Show whether this PA has follow-me logic enabled.",
        "Is the follow me flag set on this override?",
        "Does this PA follow the member to a new plan?",
        "Check the PA follow me configuration.",
        "Is plan portability enabled for this PA via follow me logic?",
        "Show the follow me setting on this prior authorization record.",
    ],

    # ── Pharmacy (70%): confused with pharmacy_info (cross-domain) ─────
    # Key: "search/filter claims BY pharmacy name" (multi-claim history)
    # NOT: "get pharmacy details FOR one claim" (pharmacy_info)
    "Pharmacy": [
        "Show claims filled at CVS PHARMACY 00610.",
        "List fills dispensed by WALGREENS 04528.",
        "Retrieve claims from RITE AID 11237.",
        "Which fills were filled at TARGET PHARMACY 01893?",
        "Give me claims processed at WALMART PHARMACY 10340.",
        "Filter the member's claim history by pharmacy name.",
        "Show all claims from this pharmacy location.",
        "Which pharmacy has dispensed the most claims for this member?",
        "List claims filled at Costco pharmacy for this member.",
        "Search claims from a specific pharmacy store.",
        "Show me all fills at Kroger pharmacy.",
        "Pull claims from retail pharmacy CVS.",
        "Filter claims by the dispensing pharmacy name.",
        "Which stores filled prescriptions for this member?",
        "Claims processed at a particular pharmacy location.",
    ],

    # ── pa_agent_code (70%): confused with pa_modification_history ─────
    "pa_agent_code": [
        "What is the agent code on this PA?",
        "Show the agent code assigned to this prior authorization.",
        "Who created this PA based on the agent code?",
        "Which PAs were created by agent code C?",
        "Display the agent code for each PA on this member.",
        "What agent or source code is on this PA?",
        "Show me the agent code that indicates who set up this PA.",
        "Is the agent code A, C, 3, or H on this PA?",
        "What does the agent code on this PA indicate about its source?",
        "Agent code lookup for this prior authorization.",
        "Show the source code indicating who created this override.",
        "Which agent code was used when this PA was set up?",
    ],

    # ── related_cagm (70%): confused with cvs_id_lookup, family_members
    "related_cagm": [
        "Show other CAGMs linked by the same CVS ID.",
        "Find CAGM records matching this member by family.",
        "Which members share this CVS identifier?",
        "Related CAGMs for this member.",
        "List CAGMs by family ID under this client.",
        "Who else is linked to this member by CVS ID?",
        "Show linked CAGM records for this member.",
        "Find all related CAGMs for this member.",
        "CAGM links for this enrollment.",
        "Which members share the same family identifier?",
        "Show all CAGM records connected to this member's CVS ID.",
        "Display CAGM linkages by family or CVS ID.",
        "Are there other CAGM records linked to this same CVS identifier?",
        "Pull all related CAGM records for this enrollment.",
        "Show cross-referenced CAGM members by CVS ID.",
    ],

    # ── mail_order_info (73%): confused with PharmType ─────────────────
    # Key: "was THIS specific claim filled via mail order" (single claim)
    # NOT: "filter claims by pharmacy type" (PharmType = search)
    "mail_order_info": [
        "Was this prescription filled through mail order?",
        "Is this a home delivery or retail fill?",
        "Was this medication shipped to the patient?",
        "Show the delivery channel for this claim.",
        "Home delivery information for this claim.",
        "Did this come from a mail service pharmacy?",
        "Was this prescription dispensed via home delivery?",
        "Mail order status for this specific claim.",
        "Was this particular fill shipped or picked up in store?",
        "Is this claim a mail order pharmacy fill?",
        "Was this specific prescription mailed to the member?",
        "Check if this one claim was processed via mail order.",
        "Is this a home delivery fill for this specific claim?",
        "Mail versus retail: how was this particular claim fulfilled?",
        "Was this specific fill delivered to the patient's home?",
        # ── From failure: fulfillment channel / 90-day mail order ───────
        "What is the fulfillment channel — retail or mail?",
        "Is this a 90-day mail order fill?",
        "Was this a 90-day mail order prescription?",
        "Is this home delivery or a retail pickup for this specific claim?",
    ],

    # ── medicare_part_d (73%): needs more diverse phrasing ─────────────
    "medicare_part_d": [
        "Show the Medicare Part D details for this claim.",
        "What are the PDE fields on this claim?",
        "MEDD pricing breakdown for this specific claim.",
        "Medicare Part D summary for this claim.",
        "Show the N1 segment details for this Part D claim.",
        "PDE data for this claim.",
        "Part D cost sharing breakdown on this fill.",
        "What Medicare plan paid and what the beneficiary owes.",
        "Show the true out-of-pocket amount for this Part D claim.",
        "CMS pricing fields on this claim.",
        "MEDD pricing for this claim.",
        "What Part D benefit phase applies to this claim?",
        "Was catastrophic coverage reached on this fill?",
        "Show Medicare Part D cost sharing for this specific claim.",
        "PDE segment data and MEDD pricing for this fill.",
        # ── From failure: catastrophic phase / LICS on claim pricing ────
        "Was the catastrophic coverage phase reached on this claim?",
        "Is there a low income subsidy applied to the pricing on this claim?",
        "What LICS subsidy reduction was applied to the Part D cost on this claim?",
        "Show the catastrophic phase cost-sharing breakdown for this claim.",
        "What low income cost sharing was applied to this Part D claim?",
    ],

    # ── DrugLast (80%): needs "when was X LAST filled" phrasing ────────
    "DrugLast": [
        "When was LISINOPRIL last dispensed for this member?",
        "Last fill date for METFORMIN.",
        "When was the most recent fill of ATORVASTATIN?",
        "How long ago was OMEPRAZOLE last dispensed?",
        "Show the last dispensing date for AMLODIPINE.",
        "When did this member last pick up their prescription?",
        "Most recent fill date for this drug.",
        "Last time GABAPENTIN was filled for this member.",
        "When was the last time this member got HYDROCHLOROTHIAZIDE?",
        "Show the most recent dispensing date for this medication.",
        "When was the LAST time this drug was filled?",
        "Most recent dispensing of SERTRALINE for this member.",
        "What is the latest fill date for this medication?",
        "When was this drug most recently picked up?",
        "How recently was LOSARTAN last dispensed?",
    ],

    # ── prior_auth_info (80%): for ONE SPECIFIC claim ─────────────────
    "prior_auth_info": [
        "Was prior authorization required for this claim?",
        "Show the PA status on this specific claim.",
        "Did this claim need prior authorization?",
        "Is there a Smart PA attached to this claim?",
        "PA details for this specific claim.",
        "What is the authorization number on this claim?",
        "Tell me if this prescription required prior auth to process.",
        "Prior authorization requirements for this claim.",
        "Show whether this claim passed PA validation.",
        "Does this claim have an associated PA number on file?",
        "Was PA needed for this specific fill?",
        "What PA was checked when processing this claim?",
        "Show the prior authorization status for this one claim.",
        "Is there a PA on record for this specific prescription?",
        "PA status lookup for this particular claim.",
    ],

    # ── help (was 10% ensemble, needs strong anchors) ─────────────────
    "help": [
        "How do I submit a claim correctly?",
        "What steps should I follow to avoid rejection?",
        "Guide me through the claim submission process.",
        "How can I prevent my claim from being denied?",
        "What are the best practices for submitting pharmacy claims?",
        "Help me understand the claim filing process.",
        "What documentation is needed to submit a claim?",
        "Step by step instructions for claim submission.",
        "How do I properly file a pharmacy claim?",
        "Tips for successful claim processing.",
        "What should I do if I don't know how to file this claim?",
        "General guidance on claim filing procedures.",
        # ── From failure analysis: guidance/how-to vs actual queries ───
        "I need guidance on how to query member information.",
        "How do I look up member eligibility information in this system?",
        "Can you guide me on how to find plan details?",
        "I need help understanding how to navigate this system.",
        "How do I search for a specific claim?",
        "I need some help with a claim.",
        "Help me understand how to use this tool.",
        "What can this system do for me?",
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
        "Is a generic substitute available for this specific medication?",
        "Can this prescription be switched to a generic version?",
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
        "Tell me the doctor's information for this specific prescription claim.",
        "Show prescriber details for this one claim.",
        # ── From failure: "script" = prescription in PBM context ────────
        "Who wrote this script?",
        "Who wrote the script for this claim?",
        "Which doctor wrote this script?",
        "Who is the prescriber of this script?",
        "Script writer information for this claim.",
    ],
    "audit_info": [
        "When was claim 132435151040074 sequence 001 first created?",
        "Claim 201503823714118 sequence 001 add date and change date.",
        "Who last modified claim 201752592251000 sequence 001 and when?",
        "What is the creation timestamp of claim 211263773300000 sequence 004?",
        "When was claim 221172865083001 sequence 001 added to the system?",
        "Show the modification history for this claim.",
        "What system events are logged against this claim?",
        "When was this claim record last modified?",
    ],
    "reversal_info": [
        "R&R information for claim 242905816136000 sequence 001.",
        "R&R status for claim 242905816136000 sequence 001.",
        "R&R report for claim 253603736282009 sequence 001.",
        "Was claim 231181462825000 sequence 001 reversed?",
        "Claim modifications for claim 260021649904000 sequence 005.",
        "Has this claim been reversed or resubmitted?",
        "Show reversal history for this claim.",
        "Any R&R activity on this claim?",
    ],
    "claim_status": [
        "What is the current status of claim 130041467416065 sequence 001?",
        "Is claim 220133725669000 sequence 001 paid, rejected, or pending?",
        "Quick status check on claim 230381673488000 sequence 001.",
        "What was the result of processing claim 230624075311000 sequence 002?",
        "Adjudication outcome for claim 191406379285000 sequence 001.",
        "Has this claim been fully processed yet?",
        "Paid or rejected? Tell me this claim's status.",
        "I just need to know if this claim went through.",
        "What is the current adjudication outcome for this claim?",
    ],
    "pricing_info": [
        "What is the copay on claim 132435151040074 sequence 001?",
        "Show the pricing breakdown for claim 220133725669000 sequence 001.",
        "What did the patient pay on this specific claim?",
        "Ingredient cost and fees for claim 191406379285000 sequence 001.",
        "Copay calculation steps for this claim.",
        "Break down the copay and ingredient cost for this claim.",
        "Patient pay amount for this one prescription.",
        "Dispensing fee and total cost breakdown for this fill.",
        "U&C price versus the plan-paid amount on this claim.",
    ],
    "pharmacy_info": [
        "Which pharmacy dispensed claim 132435151040074 sequence 001?",
        "Pharmacy details for claim 220133725669000 sequence 001.",
        "Where was this specific claim filled?",
        "Dispensing pharmacy name for this claim.",
        "Store location for claim 191406379285000 sequence 001.",
        "What is the NCPDP number for the pharmacy on this fill?",
        "Tell me the pharmacy name and store number on this specific claim.",
        "Where was this prescription physically dispensed?",
    ],
    "reimbursement_info": [
        "How much did the plan reimburse the pharmacy for this claim?",
        "Total payment to the dispensing pharmacy on this claim.",
        "Pharmacy reimbursement amount for this fill.",
        "What amount was paid out to the pharmacy?",
        "Show the reimbursement calculation for this claim.",
        "Reimbursement details and rationale for this claim.",
        "Was the pharmacy paid MAC or AWP on this fill?",
        "How did the plan determine the pharmacy payment amount?",
    ],
    "daw_info": [
        "What is the dispense as written code on this claim?",
        "Is this prescription DAW or was substitution allowed?",
        "Show the DAW indicator for this fill.",
        "Was the brand name required by the prescriber?",
        "DAW code for this claim.",
        "Is this a DAW 1 or DAW 0 claim?",
        "Was generic substitution blocked on this claim?",
        "DAW? Tell me the substitution status.",
    ],
    "government_claim_type": [
        "Was this claim processed under a government program?",
        "Is this a Medicare or Medicaid claim?",
        "What federal program does this claim fall under?",
        "Government claim classification for this claim.",
        "Is this a commercial or government-sponsored claim?",
        "Govt claim type?",
        "Was this filled under a state or federal plan?",
        "What program type was this claim processed under?",
    ],
    "network_info": [
        "Which pharmacy network processed this claim?",
        "Is this an in-network or out-of-network claim?",
        "Network details for this prescription fill.",
        "What paying network was used on this claim?",
        "Was this claim filled through the preferred pharmacy network?",
        "Network? Which network tier applies?",
        "What is the network status of the pharmacy on this claim?",
        "Show the paying network details for this prescription.",
    ],
    "multi_claim_summary": [
        "Can you give me a summary of all claims on file for this member?",
        "I need a complete overview of every claim submitted.",
        "Show me all the claims this member has had processed.",
        "Full claims history overview for this member please.",
        "Pull up a consolidated summary of the entire claims list.",
        "How many total claims does this member have and what are they?",
        "Multi-claim summary for this patient.",
        "Aggregate all claims and show me the highlights.",
    ],
    "cob_info": [
        "Is there coordination of benefits on this claim?",
        "Show COB details for this claim.",
        "Was a secondary payer involved in this claim?",
        "Coordination of benefits summary for this claim.",
        "Did another insurance pay before this plan?",
        "Other payer information on this fill.",
        "COB breakdown for this claim.",
        "Is this member dual-covered for this claim?",
        # ── From failure: "other insurance payment" on THIS claim ───────
        "What was the other insurance payment on this prescription?",
        "How much did the other insurer pay on this specific claim?",
        "Secondary payer payment amount on this claim.",
        "What did the primary insurance pay before this plan on this fill?",
    ],
    "rx_details": [
        "What is the prescription number on this claim?",
        "How many days supply was dispensed on this fill?",
        "Show the RX number and quantity for this claim.",
        "What fill number is this?",
        "RX details for this specific claim.",
        "Drug strength and dosage form on this claim.",
        "Quantity dispensed and days supply for this prescription.",
        "Tell me the prescription specifics — RX number and quantity.",
        # ── From failure: fill number / original vs refill for ONE claim ─
        "What fill number is this — is it original or a refill?",
        "Is this a new fill or a continuation?",
        "Is this an original fill or a subsequent refill on this claim?",
        "What is the fill sequence number on this specific claim?",
        "Is this the first dispense or a refill on this prescription?",
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
        "Total out-of-pocket for a specific drug across all fills.",
        "Cost per fill for OMEPRAZOLE over the past year.",
        "Average copay across all fills of this medication.",
    ],
    "Settlement": [
        "Show claims with settlement code 358.",
        "Filter by settlement code 001 across all claims.",
        "Which claims returned settlement code 425?",
        "List fills that received settlement code 310.",
        "Retrieve claims with pharmacy settlement response 200.",
        "Show all claims with settlement code 2.",
        "Search claims by settlement response code number.",
        "Filter claims by a specific settlement code.",
    ],
    "Pharmacy": [
        "Show claims filled at CVS PHARMACY 00610.",
        "List fills dispensed by WALGREENS 04528.",
        "Retrieve claims from RITE AID 11237.",
        "Which fills were filled at TARGET PHARMACY 01893?",
        "Give me claims processed at WALMART PHARMACY 10340.",
        "Filter the member's claim history by pharmacy name.",
        "Show all claims from this pharmacy location.",
        "List claims by dispensing pharmacy name.",
        "Show fills at Costco pharmacy for this member.",
        "Which stores filled prescriptions for this member?",
    ],
    "Prescriber": [
        "Show claims by prescriber NOEUV.",
        "List claims written by Dr. PATEL.",
        "Retrieve fills prescribed by NPI 1234567890.",
        "Which claims were ordered by prescriber SMITH?",
        "Display claims from prescriber Dr. JOHNSON.",
        "Filter this member's claims by the prescribing doctor.",
        "Search all claims by this specific prescriber NPI.",
        "Show all fills written by this prescriber across history.",
        "How many claims has this prescriber written for this member?",
        "Which prescribers wrote for this member?",
    ],
    "Status": [
        "Show all rejected claims for this member.",
        "List claims in paid status this year.",
        "Which claims are currently pending?",
        "Display all denied claims across all drugs.",
        "Give me claims in reversed status.",
        "Filter claims by status — show pending only.",
        "Show all paid claims for this member.",
        "List claims that were denied.",
    ],
    "RejectCode": [
        "Show claims with reject code 79.",
        "Filter claims by rejection code 75.",
        "Which claims have NCPDP reject code MR?",
        "List claims rejected under code 76.",
        "Retrieve claims with reject code 70.",
        "Search claims by reject code for this member.",
        "How many claims were rejected with code 75?",
        "Filter claims by specific rejection code.",
    ],
    "PriorAuth": [
        "Which claims required prior authorization?",
        "Show fills that went through a PA process.",
        "List claims where PA was approved.",
        "Retrieve prescriptions with an active prior auth on file.",
        "Display PA-approved claims for specialty drugs.",
        "Filter claims by prior authorization requirement.",
        "Show all PA-required claims for this member.",
        "How many claims needed PA approval in this history?",
    ],
    "PharmType": [
        "Show claims filled at specialty pharmacies only.",
        "Filter by pharmacy type — retail versus mail order.",
        "Which claims came from a mail-service pharmacy?",
        "List claims by pharmacy channel for this member.",
        "Show only retail pharmacy fills.",
        "Filter claims by pharmacy type code.",
        "Were any claims processed at a long-term care pharmacy?",
        "Which fills came from a specialty pharmacy?",
        "Show claims by dispensing pharmacy type.",
        "Pull claims from retail locations only.",
    ],
    "Generic": [
        "Show only generic drug claims for this member.",
        "Filter for generics only.",
        "Which claims were dispensed as generic medications?",
        "List all generic fills.",
        "How many generic prescriptions has this member had?",
        "Filter by generic indicator.",
        "Pull generic drug fills for this member.",
        "Show claims where a generic was dispensed.",
    ],
    "Brand": [
        "Show only brand name drug claims for this member.",
        "Filter for brand-name fills only.",
        "Which claims were dispensed as brand medications?",
        "List all brand drug fills.",
        "How many brand prescriptions does this member have?",
        "Filter by brand indicator.",
        "Pull brand drug fills for this member.",
        "Show claims where the brand was dispensed instead of generic.",
    ],
    "DaysSupply": [
        "Filter claims by days supply — show only 90-day fills.",
        "Which prescriptions had 30 days supply or less?",
        "Show all 60-day supply fills for this member.",
        "List claims where days supply exceeds 30.",
        "How many claims were dispensed as 90-day supplies?",
        "Days supply breakdown across this member's fills.",
    ],
    "Refills": [
        "How many refills remain on this member's prescriptions?",
        "Show refill history for this member.",
        "Which prescriptions have been refilled more than 3 times?",
        "List claims by refill count.",
        "Are there any prescriptions eligible for refill?",
        "Refill count across this member's medications.",
    ],
    "Diagnosis": [
        "Filter claims by ICD-10 diagnosis code.",
        "Show claims associated with diagnosis J45.",
        "Which claims have a diabetes-related diagnosis code?",
        "List claims by diagnosis for this member.",
        "What diagnosis codes appear on this member's claims?",
        "Filter by diagnosis code E11 for this member.",
    ],
    "Month": [
        "Show all claims from January.",
        "Claims processed in March for this member.",
        "List fills from October.",
        "Filter claims by month — show February only.",
        "Which claims were processed in December?",
        "April claims for this member.",
    ],
    "NDC": [
        "Search claims by NDC 00071-0155-23.",
        "Show fills for NDC 33342-0395-44.",
        "Which claims used NDC 00093-7180-98?",
        "Filter by national drug code.",
        "List claims with this NDC number.",
        "Pull claims for NDC 16729-0044-01.",
    ],
    "Manufacturer": [
        "Filter claims by drug manufacturer — show Teva only.",
        "Which manufacturers appear in this member's history?",
        "Show claims for drugs made by Mylan.",
        "List claims by manufacturer name.",
        "Claims from Pfizer products for this member.",
        "Pull claims for medications made by Novartis.",
    ],
    "Plan": [
        "Filter claims by plan code for this member.",
        "Show claims under plan ABC123.",
        "Which claims were processed under this insurance plan?",
        "List claims by benefit plan.",
        "Show all fills under the current plan ID.",
        "Filter by insurance plan code.",
    ],
    "ClaimNum": [
        "Look up claim number 132435151040074.",
        "Find claim 220133725669000 in the system.",
        "Pull up claim 201503823714118.",
        "Search for claim number 242905816136000.",
        "Can you pull up claim number 201752592251000?",
        "Locate claim 180571470939000 for this member.",
    ],
    "date_range_claims": [
        "Show me all claims from March through June.",
        "Claims within the last 90 days for this member.",
        "Pull claims that contributed to the deductible this year.",
        "What claims were processed between January 1 and April 30?",
        "List claims from the current benefit year.",
        "Pull the claim history for the last 12 months.",
    ],
    "drug_info": [
        "What drug was dispensed on the claims for this member?",
        "Show me the NDC and GPI for drugs in this history.",
        "Drug name and therapeutic class for claims on this member.",
        "GPI lookup across this member's prescriptions.",
        "What therapeutic classes are represented in this member's fills?",
        "Which drugs has this member been prescribed?",
    ],
    "drug_interaction_info": [
        "Were there any DUR edits triggered across this member's claims?",
        "Show drug interaction alerts for this member's prescription history.",
        "DUR edit details across all fills for this member.",
        "Are there drug-drug interaction warnings in the claim history?",
        "Drug utilization review alerts across this member's history.",
        "What clinical edits were triggered for this member's medications?",
    ],
    "fill_date_info": [
        "When was each prescription filled for this member?",
        "Show fill dates across all claims for this member.",
        "What was the date of service for the most recent fill?",
        "List claims sorted by fill date.",
        "Dispensing dates for all prescriptions on file.",
        "Fill date timeline for this member's prescriptions.",
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
        # ── From failure: TF status on member eligibility (not claim TF) ─
        "What is the TF status on this member's eligibility record?",
        "Is the member flagged as a transition fill member on their eligibility?",
        "Show the TF status and start date from the member eligibility.",
        "What is the member's transition fill eligibility status?",
        "Is this member currently in a transition fill eligibility window?",
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
    # ── family_type (80%): confused with beneficiary_info ──────────────
    # Key: plan CLASSIFICATION (individual vs family), NOT benefit accumulations
    "family_type": [
        "Is this member on a family or individual plan?",
        "Show the coverage tier designation for this member.",
        "What is the family coverage classification for this member?",
        "Employee-only or employee-plus-spouse?",
        "Coverage tier for this member.",
        "Is this an individual or family enrollment?",
        "What plan tier is this member enrolled in?",
        "Show the family status indicator for this member.",
        "Is this member's plan single or family coverage?",
        "Family or individual designation for this enrollment.",
        "What type of coverage tier does this member have?",
        "Is this an individual, family, or employee-plus coverage?",
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
        # ── From failure: PA overview with PA number → pa_summary ───────
        "PA overview for this prior authorization.",
        "Give me an overview of PA JW012726LC.",
        "PA overview for PA JW012726LC.",
        "High-level PA overview for this PA record.",
        "Overview of this prior authorization record.",
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
        # ── From failure: short "PA follow me?" query ────────────────────
        "PA follow me?",
        "Follow me logic?",
        "PA follow me logic?",
        "Does this PA follow me?",
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
        # ── From failure: plan name/effective date and formulary queries ──
        "What is the plan name and effective date for this member?",
        "What formulary is associated with this member's current plan?",
        "Give me the benefit plan name for this member.",
        "What plan is this member currently enrolled in?",
        "Show the formulary tier structure for this member's plan.",
        "What is the member's active benefit plan name?",
        "Which plans offer mail-order benefits in this group?",
        "Find plans with mail-order benefits for this member's group.",
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
