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
    ],

    # ── greeting: confused with out_of_scope ───────────────────────────
    # Added: repetitive/emphatic forms (Group 12)
    "greeting": [
        "Hello", "Hi there", "Welcome", "Hiya",
        "Hello, how are you?", "Hi, good to see you",
        "Hey", "What's up", "Good morning", "Good afternoon",
        "Yo!", "Greetings", "Howdy",
        "Good evening", "Good day", "Morning", "Afternoon",
        "Evening", "Hey there", "Hello there",
        # Group 12: repetitive/emphatic greeting forms (previously confused → out_of_scope)
        "Hey hey hey",
        "Hello hello",
        "Hello hello hello",
        "Hiii",
        "Heyyyy",
        "Heyyy there",
        "Hiiii",
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

    # ── mail_order_info: single-claim pharmacy channel lookup ─────────
    # Key: "was THIS specific claim filled via mail order" (one claim)
    # NOT: "filter claims by pharmacy type" (PharmType = history search)
    # Added: NCPDP pharmacy type / dispensing channel phrasing (Group 3A)
    "mail_order_info": [
        # Original anchors
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
        # Group 3A: NCPDP pharmacy type / dispensing channel on a specific claim
        "Was claim 243122443413000 sequence 001 filled at a mail order pharmacy or retail?",
        "Dispensing channel and NCPDP pharmacy type for claim 900112233440000 seq 001.",
        "Show the pharmacy dispensing type for claim 201533994612000 sequence 003 — retail, mail, or specialty.",
        "What NCPDP pharmacy type and dispensing channel are on claim 243122443413000 sequence 001?",
        "Provide the dispensing channel information for claim 412369874100000 sequence 001.",
        "What is the NCPDP pharmacy type on claim 900112233440000 seq 001?",
        "Was this claim dispensed via mail order or retail? Claim 201533994612000 sequence 003.",
        "Show the dispensing channel recorded on claim 412369874100000.",
        "What pharmacy channel — retail, mail, specialty — is on claim 243122443413000 seq 001?",
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
    ],

    # ── DrugLast (80%): needs "when was X LAST filled" phrasing ────────
    # Includes "drug" placeholder and claim-number suffix to match inference path.
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
        # Exact user query patterns (drug=placeholder, real claim number for normalization)
        "When was drug taken last for the member with claim ?",
        "What was the last fill date for drug for the member with claim ?",
        "Show the latest claim for drug for the member with claim including status and fill details.",
        "When was drug most recently dispensed for the member with claim ?",
        "What is the most recent drug fill date for the member with claim ?",
        "Show me the last time drug was filled for the member with claim.",
        "When did the member with claim last receive drug?",
    ],

    # ── prior_auth_info: for ONE SPECIFIC claim ───────────────────────
    # Distinguishing signals:
    #   - Always references a specific claim number + sequence
    #   - Asks for PA fields/metadata ON that claim (not history search)
    #   - Includes approval-outcome language (previously confused → approval_info)
    #   - Includes "Smart PA / Member PA / plan PA" (previously confused → PriorAuth)
    "prior_auth_info": [
        # Original anchors
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
        # Group 1A: approval-outcome phrasing (previously confused → approval_info)
        "Provide the PA number and approval status recorded on claim 198374652901000 seq 001.",
        "What PA type is recorded on claim 317469028345000 sequence 002 and was it approved prior to dispensing?",
        "For claim 428019365784000 sequence 001, did a Smart PA override or a Member PA govern the adjudication, and what was the result?",
        "Retrieve prior authorization status and approval details for claim 153902847631000 sequence 001.",
        "Was a member PA or plan PA used on claim 310928475630000 seq 001, and is the PA currently active?",
        "What was the PA approval outcome on claim 198374652901000 seq 001?",
        "Was this claim approved via PA — what type and was it active at fill time?",
        "Fetch the PA reference number and its current approval status for claim 198374652901000.",
        "Did the plan PA or member PA apply to this claim, and what result did it produce?",
        "Show me the authorization outcome and PA type for claim 317469028345000 sequence 002.",
        # Group 1B: single-claim PA field lookup (previously confused → PriorAuth)
        "Retrieve the prior authorization details — PA number, type, and approved/denied status — for claim 267489013456000 sequence 001.",
        "Show whether Smart PA was applied on claim 430981276543000 seq 001 and what the authorization outcome was.",
        "Display the Member PA information embedded in claim 375916284073000 sequence 001.",
        "Give me the prior auth fields from claim 621847390125000 seq 002, including the PA reference number.",
        "Pull prior auth status and number for claim 643091827465000 seq 001.",
        "PA status? Claim 509234781650000 seq 001.",
        "What PA fields are embedded in claim 267489013456000 seq 001?",
        "Show the Smart PA or Member PA data stored on claim 430981276543000 sequence 001.",
        "What prior auth number is associated with claim 621847390125000 seq 002?",
        "Is there an active PA governing claim 375916284073000 sequence 001?",
    ],

    # ── help: general guidance, not a specific-claim question ─────────
    # Added: "why do claims get denied" general question (Group 13, was confused → rejection_reasons)
    "help": [
        # Original anchors
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
        # Group 13: general denial-reason guidance (previously confused → rejection_reasons)
        "What are the common reasons my claims get denied?",
        "Why do pharmacy claims typically get rejected?",
        "What causes claim denials in general?",
        "How do I prevent my claims from being rejected?",
        "What should I know about common claim rejection reasons?",
        "What are the most frequent reasons a pharmacy claim fails?",
    ],

    # ══════════════════════════════════════════════════════════════════════
    # 2. CAP_API ANCHORS (single-claim operations)
    # ══════════════════════════════════════════════════════════════════════

    # ── generic_availability: confused with Generic, daw_info & Brand ──
    # Added: DAW+generic combo queries and BPG/interchange queries (Group 5)
    "generic_availability": [
        # Original anchors
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
        # Group 5: DAW+generic combo and brand/formulary queries
        "DAW code and generic availability — claim 243122443413000 sequence 001.",
        "I want to check DAW and generic availability flags for claim 100045678900000 sequence 001.",
        "Is claim 755400123600000 sequence 001 dispensed as a brand when a generic is on formulary?",
        "Lookup: does the drug on claim 600012398700000 seq 001 have a BPG or generic interchange?",
        "Are there generic alternatives available for the drug on claim 100045678900000 seq 001?",
        "Was a generic dispensed or was brand required — check generic availability on claim 243122443413000?",
        "Show both the DAW code and generic availability status for claim 600012398700000 sequence 001.",
        "Is there a therapeutic equivalent for the drug on this claim?",
        "Does this drug have an available generic alternative on the formulary?",
        "What BPG or generic interchange options exist for the drug on this claim?",
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
    # ── claim_status: adjudication outcome for ONE claim ─────────────
    # Added: "claim summary" phrasing (Group 4, previously confused → ClaimNum)
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
        # Group 4: "claim summary" language (previously confused → ClaimNum)
        "Provide a full claim summary for claim 198734561209000 seq 002.",
        "Show claim summary for 123456789012000 seq 001.",
        "Give me a full adjudication summary for claim 198734561209000 sequence 002.",
        "What is the overall claim status and summary for claim 123456789012000 seq 001?",
        "Full processing summary for this claim — what happened during adjudication?",
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
    # ── multi_claim_summary: ALL claims for a member, no filter ───────────
    # Distinguishing signal: no date/drug/status filter — full aggregate.
    # Confused with date_range_claims (date filter), Pharmacy, Month, etc.
    "multi_claim_summary": [
        # Original anchors
        "Can you give me a summary of all claims on file for this member?",
        "I need a complete overview of every claim submitted.",
        "Show me all the claims this member has had processed.",
        "Full claims history overview for this member please.",
        "Pull up a consolidated summary of the entire claims list.",
        "How many total claims does this member have and what are they?",
        "Multi-claim summary for this patient.",
        "Aggregate all claims and show me the highlights.",
        # Group 2: misclassified queries (previously confused → date_range_claims, etc.)
        "What is the total number of claims and overall spend for this member across all their prescriptions?",
        "Summarize the entire claim history for this member, including all drugs, dates, and costs.",
        "Can you aggregate all claims for this member and show me the breakdown by drug?",
        "Member's entire claim history — all claims, not just one — for the current benefit year.",
        "Pull all claims for this member and give me a summary of their prescription utilization pattern over the past year.",
        "Give me a summary of all prescriptions filled by member 987654321 this year.",
        "How many claims did member 321654987 have this quarter and what was the total member pay?",
        "How many pharmacy claims does this member have on file and what do they total in cost?",
        "Give me an overview of all claims submitted for member 741852963 since plan inception.",
        "What is the total claim volume for member 963741852 and which drugs appear most frequently?",
        # Paraphrase variants reinforcing "no filter, aggregate"
        "Aggregate all claims for this member with no filter — full history summary.",
        "Show member's complete prescription utilization: total count, total spend, drug breakdown.",
        "Full claim count and cost summary for this member across all time.",
        "Prescription utilization overview — all claims ever filed for this member.",
        "Break down this member's entire claim history by drug, date, and cost.",
        "How many fills total has this member had and what have they paid in aggregate?",
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
    ],
    # ── rx_details: prescription specifics for ONE claim ─────────────
    # Added: days supply with claim number anchors (Group 6, was confused → DaysSupply)
    "rx_details": [
        # Original anchors
        "What is the prescription number on this claim?",
        "How many days supply was dispensed on this fill?",
        "Show the RX number and quantity for this claim.",
        "What fill number is this?",
        "RX details for this specific claim.",
        "Drug strength and dosage form on this claim.",
        "Quantity dispensed and days supply for this prescription.",
        "Tell me the prescription specifics — RX number and quantity.",
        # Group 6: days supply on a specific claim (previously confused → DaysSupply)
        "I want to confirm the supply duration — how many days supply is on claim 132435151040074?",
        "Was this a 30-day or 90-day supply? Check claim 180571470939000 seq 001.",
        "How many days supply was dispensed on claim 132435151040074?",
        "Is this a 30-day or 90-day fill? Claim 180571470939000 sequence 001.",
        "What is the days supply recorded for claim 132435151040074?",
        "Supply duration for claim 180571470939000 seq 001 — 30 or 90 days?",
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
        # Exact user query patterns (drug=placeholder, real claim number for normalization)
        "How much did the member with claim   pay out of pocket for drug on the last fill?",
        "What has the member with claim   spent on prescriptions year-to-date?",
        "Show me all claims having deductible or out of pocket amount greater than zero for the member with claim  .",
        "Show me all claims having a TrOOP amount for the member with claim  .",
        "Total out-of-pocket costs across all fills for the member with claim  .",
        "Year-to-date spending on prescriptions for the member with claim  .",
        "What is the TrOOP amount across claims for the member with claim  ?",
        "Show claims with deductible amounts greater than zero for the member with claim  .",
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
        # Exact user query patterns (drug=placeholder, real claim number for normalization)
        "Are there any GLP-1 claims rejected for the member with claim ?",
        "Are there any claims with a transition fill (TF) tag this year for the member with claim ?",
        "Was the drug claim on 03/15/2024 paid or rejected for the member with claim ?",
        "Was another claim ever submitted for the member with claim after this rejection?",
        "Show all rejected or paid drug claims for the member with claim.",
        "Any claims with a TF status for the member with claim ?",
        "Which claims were rejected for the member with claim this year?",
    ],
    # ── RejectCode: filter history by specific NCPDP reject code value ─
    # Added: named reject codes (DG, M1, FQ) in history-search context (Group 8)
    "RejectCode": [
        # Original anchors
        "Show claims with reject code 79.",
        "Filter claims by rejection code 75.",
        "Which claims have NCPDP reject code MR?",
        "List claims rejected under code 76.",
        "Retrieve claims with reject code 70.",
        "Search claims by reject code for this member.",
        "How many claims were rejected with code 75?",
        "Filter claims by specific rejection code.",
        # Exact user query patterns
        "Show me all rejected claims with reject code 75 for the member with claim   in the last 6 months.",
        "Identify claims rejected for refill too soon or quantity limits or eligibility issues for the member with claim  .",
        "Which claims for the member with claim   have reject code 75?",
        "List all claims rejected with code 76 for the member with claim  .",
        "Claims rejected for quantity limit exceeded for the member with claim  .",
        "Show claims rejected due to eligibility issues for the member with claim  .",
        "Filter by NCPDP reject code 75 for the member with claim  .",
        # Group 8: named alphanumeric codes in history filter context
        "Retrieve all claims that hit reject code DG across this member's history.",
        "Pull up all fills that returned reject code M1 in the adjudication response.",
        "Display fills across the full history that were turned away with reject FQ.",
        "Break down the member's rejected claims by NCPDP reject code.",
        "Show member's history filtered to reject code 75 only.",
        "How many claims in the history hit reject code 76?",
        "List all claim history entries where the NCPDP reject code was DG.",
        "Filter the member's history by reject code M1 — how many claims?",
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
        # Exact user query patterns
        "Any claim that was paid using Prior Authorization for drug for the member with claim   in the last 30 days?",
        "Show claims paid using prior authorization for the member with claim  .",
        "Which drug claims used PA approval for the member with claim  ?",
        "List claims that went through a PA process for the member with claim  .",
    ],
    # ── PharmType: filter member HISTORY by pharmacy channel/type ─────
    # Key: searching/filtering MULTIPLE claims by pharmacy type
    # NOT: "what type of pharmacy filled THIS claim" (mail_order_info)
    # Added: mail-order fills in history (Group 3B), compound history (Group 3C)
    "PharmType": [
        # Original anchors
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
        # Group 3B: mail-order history filter (previously confused → mail_order_info)
        "Filter the claim history to only mail-order fills.",
        "Which of this member's prescriptions were filled via home delivery?",
        "I want to isolate just the mail order fills this member has had since January.",
        "Does this member have any home delivery claims on file?",
        "Show only home delivery fills from this member's claim history.",
        "Pull the member's history — filter by mail order pharmacy type only.",
        "How many mail-order fills does this member have in their history?",
        "List all specialty pharmacy fills across the member's claim history.",
        "Filter claim history to pharmacy type: home delivery.",
        # Group 3C: compound history filter (previously confused → compound_info)
        "Are there any compound medication claims in the history for this member?",
        "Member's compounding claims history, any MIC breakdowns available?",
        "Filter claim history to compound fills only.",
        "Show all compound drug claims across this member's history.",
        "How many compound medication fills has this member had?",
        "Pull only compounded prescriptions from this member's claim history.",
    ],
    # ── Generic: filter member history for generic-dispensed claims ───
    # Added: non-branded, DAW+generic, tier-1 generic phrasing (Group 10)
    "Generic": [
        # Original anchors
        "Show only generic drug claims for this member.",
        "Filter for generics only.",
        "Which claims were dispensed as generic medications?",
        "List all generic fills.",
        "How many generic prescriptions has this member had?",
        "Filter by generic indicator.",
        "Pull generic drug fills for this member.",
        "Show claims where a generic was dispensed.",
        # Group 10: misclassified queries (previously confused → Brand, daw_info, date_range_claims)
        "I want to see every non-branded fill across the member's entire claim history.",
        "Give me a list of all DAW code 1 overrides or generic claims for member 200456789.",
        "Run a claim history search for all tier-1 generic utilization for member 400223344.",
        "Which claims in the member's history had a DAW code indicating a generic was dispensed?",
        "Show all generic drug fills in this member's claim history.",
        "Filter the history to only generic fills — no brand name drugs.",
        "How many generic claims has this member had vs brand?",
        "List all DAW-0 fills in the member's claim history.",
        "Pull only generic drug dispensings from this member's history.",
    ],
    # ── Brand: filter member history for brand-dispensed claims ───────
    # Added: NDA-approved brand, specialty brand, brand with cost phrasing (Group 10)
    "Brand": [
        # Original anchors
        "Show only brand name drug claims for this member.",
        "Filter for brand-name fills only.",
        "Which claims were dispensed as brand medications?",
        "List all brand drug fills.",
        "How many brand prescriptions does this member have?",
        "Filter by brand indicator.",
        "Pull brand drug fills for this member.",
        "Show claims where the brand was dispensed instead of generic.",
        # Group 10: misclassified queries (previously confused → Settlement, Generic, daw_info, Prescriber)
        "Narrow the claim history search to NDA-approved brand drugs for member 300112233.",
        "Which of this member's claims were for a brand drug that had a generic available?",
        "I want a count of specialty brand drug fills in member 400223344's claim records.",
        "Show member 200456789 all brand fills — include cost share, date of service, and prescriber NPI.",
        "List all brand name drug claims in this member's history.",
        "How many brand fills has this member had? Filter out generics.",
        "Show brand-only claim history for member, including pricing and prescriber.",
        "Pull all DAW-1 or DAW-2 fills from this member's claim history.",
        "Filter history to only brand-dispensed prescriptions for this member.",
    ],
    "DaysSupply": [
        "Filter claims by days supply — show only 90-day fills.",
        "Which prescriptions had 30 days supply or less?",
        "Show all 60-day supply fills for this member.",
        "List claims where days supply exceeds 30.",
        "How many claims were dispensed as 90-day supplies?",
        "Days supply breakdown across this member's fills.",
    ],
    # ── Refills: filter history by fill count/sequence ────────────────
    # Added: fill-number comparison queries (Group 7, was confused → Status)
    "Refills": [
        # Original anchors
        "How many refills remain on this member's prescriptions?",
        "Show refill history for this member.",
        "Which prescriptions have been refilled more than 3 times?",
        "List claims by refill count.",
        "Are there any prescriptions eligible for refill?",
        "Refill count across this member's medications.",
        # Exact user query patterns
        "Did the member with claim   get their drug refill this month?",
        "Has the member with claim   refilled their drug prescription recently?",
        "Show the refill history for drug for the member with claim  .",
        "Did this member get a refill on drug this month?",
        "Has drug been refilled for the member with claim  ?",
        # Group 7: fill number / second-fill-or-later phrasing
        "Pull up every claim in the history that was a second fill or later.",
        "Across all claims in this member's history, how many were fill number 1 versus fill number 2 or above?",
        "Show all refill claims — fills number 2 and beyond — in the member's history.",
        "How many of this member's claims were first fills versus refills?",
        "List claims by fill sequence number across the full history.",
        "Which claims in the history were a third fill or later?",
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
    # ── NDC: filter member history by national drug code value ────────
    # Added: "adjudicated claims where NDC equals", "find any claim with NDC" (Group 9)
    "NDC": [
        # Original anchors
        "Search claims by NDC 00071-0155-23.",
        "Show fills for NDC 33342-0395-44.",
        "Which claims used NDC 00093-7180-98?",
        "Filter by national drug code.",
        "List claims with this NDC number.",
        "Pull claims for NDC 16729-0044-01.",
        # Group 9: misclassified queries (previously confused → Settlement, Status, etc.)
        "Look through member 200456789's history and find any claim with national drug code 68462-0168-01.",
        "I want the full list of adjudicated claims where NDC equals 00003-0895-20 for member 300112233.",
        "Gimme all claims where the NDC was 00085-0019-01 for member number 400223344.",
        "I have an NDC: 43598-0290-05. Can you find all the claims for this member that used it?",
        "Cross-reference this member's PDE records against NDC 00169-4175-11.",
        "Filter member's claim history to NDC 00003-0895-20.",
        "Show all fills for member where national drug code matches 68462-0168-01.",
        "Pull history records for NDC 43598-0290-05 across all of member's claims.",
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
    # ── date_range_claims: claims within a specific date window ────────
    # Added: PDE reconciliation, cross-claim, "adjudicated between" (Group 11)
    "date_range_claims": [
        # Original anchors
        "Show me all claims from March through June.",
        "Claims within the last 90 days for this member.",
        "Pull claims that contributed to the deductible this year.",
        "What claims were processed between January 1 and April 30?",
        "List claims from the current benefit year.",
        "Pull the claim history for the last 12 months.",
        # Group 11: misclassified queries (previously confused → fill_date_info, multi_claim_summary, etc.)
        "Show a cross-claim analysis of all fills from July through December 2024 for this member.",
        "I'm doing a PDE reconciliation — need all claims between 01/01/2024 and 06/30/2024 for this member.",
        "What's the full list of adjudicated claims for member 200456789 between Q3 and Q4 last year?",
        "List all claims for this member filed between April 2025 and the current date.",
        "All adjudicated claims in the history between 01/01/2024 and 06/30/2024 for this member.",
        "Pull claims from Q1 this year through Q2 for member 200456789.",
        "Show me every claim processed from July to December for this member.",
        "Date range search: all claims between Q3 and Q4 2024 for this member.",
    ],

    # ── DateRange: date-range filtered multi-claim search ─────────────────
    # Examples use "drug" placeholder (matches sanitize_for_embedding output)
    # and real claim numbers (normalize_query replaces them with claim_id).
    "DateRange": [
        "Give me claims for the member with claim   for the last 2 months.",
        "Give me claims for the member with claim   in the last 90 days for drug.",
        "Show all claims in the last 90 days for the member with claim  .",
        "Claims from the past 30 days for the member with claim  .",
        "Get claim history for the past 6 months for the member with claim  .",
        "Summarize the claim history for the member with claim   for the last 90 days.",
        "Show me all fills from the last 60 days for this member.",
        "Pull claims within the last 45 days for the member with claim  .",
        "What claims were processed in the last 3 months for this member?",
        "Retrieve fills from the past 2 weeks for the member with claim  .",
        "List all claims filed within the last 180 days for this member.",
        "Claims history for the last 1 year for the member with claim  .",
        "Summary of last 90 days claims including rejected and PA-related for this member.",
        "Show the claim history for the member with claim   for the last 2 months including drug details.",
    ],

    # ── DrugList: all claims for a specific drug or medication list ────────
    # Use "drug" placeholder so embedding matches inference path.
    "DrugList": [
        "Give me all claims for the member with claim   for drug.",
        "Show all claims for drug for the member with claim  .",
        "List every drug this member with claim   has been prescribed.",
        "Retrieve all fills for drug for the member with claim  .",
        "Give me the complete drug claim history for the member with claim  .",
        "Show all prescriptions filled for drug for the member with claim  .",
        "How many drug claims does the member with claim   have?",
        "What drug claims are on file for the member with claim  ?",
        "Pull up every claim for drug for this member.",
        "Show me a list of all drug-related claims for the member with claim  .",
        "Give me every fill for this drug for the member.",
        "All fills of drug across the full claim history for this member.",
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
    #    Real-world conversational phrasing — informal, problem-focused,
    #    action-oriented language that mirrors how business users actually
    #    talk about PA overrides in daily operations.
    # ══════════════════════════════════════════════════════════════════════

    "pa_summary": [
        "Give me a summary of this prior authorization.",
        "Summarize the key details of this PA.",
        "Show the most important fields on this PA.",
        "Display a high-level overview of this prior authorization.",
        "Provide the PA summary including effective dates and drug coverage.",
        # Real-world conversational additions
        "Can you give me a quick rundown of what this PA does?",
        "I need to understand the key points on this PA — what's the summary?",
        "Run through this PA for me — what's important to know?",
        "What's the bottom line on this prior authorization?",
        "Summarize what this PA is supposed to accomplish.",
        "I need a one-minute overview of this PA.",
        "What should I know about this PA?",
        "Break down what this PA does for me.",
        "Can you sum this up? I need the PA highlights.",
        "Quick summary of this override record please.",
        "Just the important stuff on this PA — not every field.",
        "What does this PA do? Give me the TLDR.",
        "Tell me about this PA — what are the key settings?",
        "Walk me through this PA override configuration.",
        "I just pulled up this PA — what am I looking at?",
    ],
    "pa_override_reject": [
        "Will this PA override a reject 75 PA required?",
        "Does this PA handle reject code 75?",
        "Will this prior authorization bypass a reject 70 non-formulary?",
        "Does this PA override reject 70 plan exclusion?",
        "Show me which reject codes this PA can override.",
        # Real-world conversational additions — reject code + description patterns
        "Will this PA fix the reject 75?",
        "Can this PA bypass reject 70 for non-formulary issues?",
        "Does this PA handle reject 76 — plan limits exceeded?",
        "What if the reject is 70 plan exclusion, will this PA help?",
        "Show me: if reject is 75 (PA required), does this PA fix it?",
        "The claim hit reject 75, will this PA clear it?",
        "Will this PA override the 70 non-formulary?",
        "Can I use this PA to get past reject 76?",
        "Does this PA cover reject 70 when it's plan exclusion?",
        "I got a 75 reject on the claim — does this PA handle that?",
        "Will this PA resolve a 76 plan limits issue?",
        "Is reject 75 PA required covered by this override?",
        "The pharmacy is getting reject 70, will this PA take care of it?",
        "Can this PA clear a reject 75 or 76?",
        "Does this prior auth address the 70 reject?",
    ],
    "pa_field_help": [
        "What does the PA type field do?",
        "Explain the purpose of the effective date field on a PA.",
        "What is the GPI override field used for?",
        "Describe what the PA status indicator means.",
        "What does the quantity limit override field do on this PA?",
        # Real-world additions — why/when questions, casual tone
        "Help me understand what this PA field means.",
        "I don't get what this field is for — can you explain?",
        "Why would I set this field on a PA?",
        "When do I use this field on a PA override?",
        "What's the purpose of this field on the override?",
        "Can you explain what this field does?",
        "I'm not sure what this field is — break it down for me.",
        "What happens if I change this field on the PA?",
        "Is this field important? What does it control?",
        "Walk me through what this PA field means.",
        "What is the point of this field on the PA record?",
        "I see this field on the PA but I don't know what it does.",
        "Explain this PA field in simple terms.",
        "What is the impact of this field on claim processing?",
        "Why would someone populate this field on a PA?",
    ],
    "pa_copay_pricing": [
        "Does this copay override influence the price?",
        "How does the copay on this PA affect pricing?",
        "Will the PA copay change the member's out-of-pocket cost?",
        "Show me how the copay override impacts the final price.",
        "Does the copay field on this PA modify the claim price?",
        # Real-world additions — conditional patterns, casual tone
        "Does this copay affect the price, and if yes, how?",
        "If the copay changes the cost, show me the impact.",
        "Will the copay override influence the patient pay? Show how.",
        "How does the copay on this PA interact with the plan pricing?",
        "Does the PA copay actually change what the member pays?",
        "Is the copay override going to lower the member's cost?",
        "Tell me how this copay override impacts the final price.",
        "Will this PA copay affect the dispensing fee or ingredient cost?",
        "Does this copay override influence the coinsurance?",
        "I need to know if this PA copay changes the claim payment.",
    ],
    "pa_drug_coverage": [
        "What drugs will this PA cover?",
        "Show me the drug list covered by this prior authorization.",
        "Which medications are included under this PA?",
        "List the drugs that this PA authorizes.",
        "Display the GPI range covered by this PA.",
        # Real-world additions — specific drug queries, casual tone
        "Does this PA cover LISINOPRIL?",
        "Is METFORMIN on the drug list for this PA?",
        "Can I use this PA for ATORVASTATIN?",
        "What meds does this PA apply to?",
        "Drugs covered — give me the full list.",
        "Show me every medication this PA authorizes.",
        "Does this PA cover this specific drug?",
        "What's the drug coverage scope for this PA?",
        "Which NDCs or GPIs are on this PA?",
        "I need to know what drugs fall under this PA.",
    ],
    "pa_claim_usage": [
        "How many claims used this PA?",
        "Show the claim count for this prior authorization.",
        "How many times has this PA been applied to claims?",
        "Display the number of claims processed under this PA.",
        "Retrieve the claim usage count for this PA.",
        # Real-world additions
        "Is this PA heavily utilized?",
        "How often has this PA been used?",
        "What's the total utilization count for this PA?",
        "How many fills hit this PA?",
        "Has this PA been used a lot?",
        "Tell me how many claims went through this PA.",
        "PA usage count — how many claims?",
        "How frequently is this PA being applied?",
        "Total claims for this prior auth?",
        "What's the volume of claims using this PA?",
    ],
    "pa_reason_code": [
        "What is the reason code on this PA?",
        "Show the reason code assigned to this prior authorization.",
        "Is the reason code U1 or LC on this PA?",
        "Which PAs have reason code OD?",
        "Display the reason code for this PA.",
        # Real-world additions
        "What reason code is set on this PA?",
        "Tell me the override reason on this PA.",
        "Is this PA set to OD or OA?",
        "What's the reason code?",
        "Show me the reason code — is it U1?",
        "Why was this PA created? What's the reason code?",
        "PA reason code for this override?",
        "Which reason code is on this prior auth?",
        "Is the reason code MB or ES on this PA?",
        "What override reason was used when this PA was set up?",
    ],
    "pa_effective_dates": [
        "What are the effective dates for this PA?",
        "Show the start and end dates on this prior authorization.",
        "When does this PA expire?",
        "Is this PA currently within its effective period?",
        "Display the dateBegin and dateEnd for this PA.",
        # Real-world additions
        "When does this PA start and end?",
        "Is this PA still active?",
        "Has this PA expired yet?",
        "What's the date range on this PA?",
        "Show me when this PA is valid.",
        "PA effective period — is it still good?",
        "When was this PA activated and when does it end?",
        "Is this PA current or has it lapsed?",
        "Tell me the effective dates — start to finish.",
        "How long is this PA valid for?",
    ],
    "pa_agent_code": [
        "What is the agent code on this PA?",
        "Show the agent code assigned to this prior authorization.",
        "Who created this PA based on the agent code?",
        "Which PAs were created by agent code C?",
        "Display the agent code for each PA on this member.",
        # Real-world additions
        "Who set up this PA?",
        "Was this PA created automatically or manually?",
        "Is the agent code A or C on this PA?",
        "Tell me who entered this PA.",
        "Is this an auto-generated PA or custom?",
        "Agent code — was it system-created?",
        "Who made this PA? Check the agent code.",
        "Was this PA set up by the system or a user?",
        "Show me the source code — who created this override?",
        "Is this PA agent code A (auto) or C (custom)?",
    ],
    "pa_ignore_status": [
        "What is the ignore status code on this PA?",
        "Show the ignore status for this prior authorization.",
        "Is the ignore status code set to Y on this PA?",
        "Which PAs have ignore status P?",
        "Display the ignoreStatusCode for this PA.",
        # Real-world additions
        "What does Ignore PA Status do?",
        "Is ignore status turned on for this PA?",
        "What happens when ignore status is Y?",
        "Tell me the ignore status setting.",
        "Is ignore status active on this PA?",
        "What does setting ignore status to Y mean?",
        "Show me the ignore status flag.",
        "Is the PA ignoring the drug status?",
        "Ignore status — is it set to Y or N?",
        "What is the purpose of the ignore status flag?",
    ],
    "pa_specialty_rx_override": [
        "Does this PA override the specialty prescription reject?",
        "Show the specialty Rx override indicator for this PA.",
        "Is the specialty Rx reject indicator enabled on this PA?",
        "Which PAs have the specialty prescription override?",
        "Display the overrideSpecialtyPrescriptionRejectIndicator.",
        # Real-world additions
        "Is the specialty override turned on?",
        "Can this PA bypass specialty Rx rejection?",
        "Will the specialty reject be overridden by this PA?",
        "Is specialty Rx override set to true?",
        "Does this PA handle specialty prescription rejects?",
        "Check if specialty override is enabled.",
        "Is the specialty reject indicator on?",
        "Will this PA let specialty scripts process?",
        "Is the specialty prescription override active?",
        "Show me if the specialty Rx flag is set.",
    ],
    "pa_clinical_admin_code": [
        "What is the clinical administration code on this PA?",
        "Show the clinical admin code for this PA.",
        "Is there a clinical administration code set on this PA?",
        "Which PAs have clinical admin code C?",
        "Display the clinicalAdministrationCode for this PA.",
        # Real-world additions
        "What's the clinical admin code?",
        "Is the clinical admin code blank or set to A?",
        "Tell me the clinical admin code on this PA.",
        "Why would I set a clinical admin code?",
        "When do I use clinical admin code A vs C?",
        "Is clinical admin code populated on this PA?",
        "What does the clinical admin code do?",
        "Check the clinical admin code setting.",
        "Does this PA have a clinical admin code?",
        "What is clinical admin code used for on a PA?",
    ],
    "pa_transform_care": [
        "What is the transform care type on this PA?",
        "Show the transform care setting for this PA.",
        "Is there a transform care type configured on this PA?",
        "Display the transformCare type for this prior authorization.",
        "Which PAs have a transform care type assigned?",
        # Real-world additions
        "What's the transform care setting?",
        "Is transform care set up on this PA?",
        "Tell me about the transform care type.",
        "Is there a care program on this PA?",
        "What does transform care do on a PA?",
        "Is transform care blank or populated?",
        "Check the transform care type.",
        "Does this PA have a transform care designation?",
        "What is the transform care program on this override?",
        "When do I use the transform care field?",
    ],
    "pa_follow_me_logic": [
        "Is follow me logic enabled on this PA?",
        "Show the follow me logic indicator for this PA.",
        "Does this PA use follow me logic?",
        "Which PAs have the follow me indicator set to true?",
        "Display the followMeLogicIndicator for this PA.",
        # Real-world additions
        "Does this PA follow the member to a new plan?",
        "Will this PA transfer if the member changes plans?",
        "Is follow me turned on?",
        "Does the PA persist across plan transitions?",
        "Does this authorization carry over to a new plan?",
        "Will this PA stick with the member if they switch plans?",
        "Is the follow me flag set on this PA?",
        "Check if this PA follows the member.",
        "Is portability enabled for this PA?",
        "Does this PA carry over during a plan change?",
    ],
    "pa_drug_type_indicator": [
        "What is the authorized drug type on this PA?",
        "Show the drug type indicator for this PA.",
        "Is the drug type set to G for GPI or N for NDC?",
        "Which PAs use NDC-based drug matching?",
        "Display the authorizedDrugType for this PA.",
        # Real-world additions
        "Is this PA matching by GPI or NDC?",
        "Drug type — GPI or NDC?",
        "How does this PA match drugs?",
        "Is this PA set up for GPI matching or NDC?",
        "What type of drug matching is configured?",
        "Tell me if this PA uses GPI or NDC.",
        "Is the drug type G or N?",
        "How are drugs being matched on this PA?",
        "Check the drug matching method on this PA.",
        "Is this PA using NDC-level or GPI-level matching?",
    ],
    "pa_modification_history": [
        "When was this PA last modified?",
        "Show the modification date and time for this PA.",
        "Which PA was most recently modified?",
        "Display the modifyDateTime for this prior authorization.",
        "How recently was this PA updated?",
        # Real-world additions
        "When was this PA last changed?",
        "Has this PA been modified recently?",
        "What's the last update date on this PA?",
        "When did someone last touch this PA?",
        "Show me when this PA was last edited.",
        "How old is this PA configuration?",
        "Was this PA changed recently?",
        "Tell me the last modified timestamp.",
        "When was this PA record last updated?",
        "What's the modification date?",
    ],

    # ── pa_contingent_therapy_override: bypass contingent therapy ────────
    "pa_contingent_therapy_override": [
        "How do I bypass contingent therapy on this PA override?",
        "Can I flip the contingent therapy flag to skip step therapy?",
        "I need to turn off the contingent therapy check for this member's PA.",
        "Where is the bypass contingent therapy indicator on this PA?",
        "Override step therapy requirement for this PA record.",
        "Disable contingent therapy validation so the claim can process.",
        "What flag do I set to waive contingent therapy on a PA?",
        "The claim is rejecting for contingent therapy — how do I override it via PA?",
        "Contingent therapy bypass — set to Y on PA override.",
        "I need to enable the bypass contingent therapy indicator.",
        "Turn off the step therapy edit using this PA.",
        "How to skip the contingent therapy step on this override?",
        "Set bypass contingent therapy flag for this PA record.",
        "Can a PA override waive the contingent therapy requirement?",
        "Steps to disable contingent therapy on PA.",
        # Additional real-world conversational
        "The member failed contingent therapy, how do I get around it with a PA?",
        "Step therapy is blocking the claim — can the PA bypass it?",
        "I need to override the step therapy edit, what do I do on the PA?",
        "Member's claim is rejecting because of step therapy, how to use PA to fix it?",
        "Contingent therapy is preventing the fill — PA override steps?",
    ],

    # ── pa_smart_pa_override: bypass Smart PA processing ────────────────
    "pa_smart_pa_override": [
        "How do I bypass the Smart PA edit using this PA override?",
        "I need to flip the Smart PA bypass flag to Y.",
        "Enter the Smart PA criteria number on this override record.",
        "Where do I set the bypass Smart PA indicator?",
        "Override Smart PA processing for this PA.",
        "Can I skip Smart PA by entering the criteria number?",
        "Set bypass Smart PA to Y on this PA record.",
        "How to turn off Smart PA validation on this override?",
        "Smart PA bypass — what fields do I configure?",
        "Disable Smart PA check for this prior authorization.",
        "The claim needs Smart PA criteria — how do I enter it on the PA?",
        "I want to bypass the Smart PA edit for this member.",
        "What is the process for overriding Smart PA on a PA?",
        "How do I configure the Smart PA criteria number on the override?",
        "Set Smart PA bypass flag and criteria number.",
        # Additional real-world conversational
        "Smart PA is blocking the claim — can I override it with the PA?",
        "The claim needs Smart PA but we want to bypass it, how?",
        "I need to get past the Smart PA edit on this claim.",
        "Member's claim is failing Smart PA — what do I set on the PA?",
        "Smart PA criteria number — where do I put it on the override?",
    ],

    # ── pa_part_b_override: route claim to Medicare Part-B ──────────────
    "pa_part_b_override": [
        "How do I route this claim to Part B using a PA?",
        "Set override reason MB so the claim pays as Part-B.",
        "I need to make the pharmacy claim process as medical Part B.",
        "Which reason code makes a claim pay under Medicare Part B?",
        "PA override to process as Part-B — what reason code?",
        "Use reason code MB on the PA for Part B payment.",
        "How to set up a PA so the claim routes to Medicare Part B?",
        "Enter MB as the override reason for Part-B processing.",
        "Can I use a PA to change claim payment to Part B?",
        "Override to Part B — steps for PA reason code MB.",
        "How do I make this claim pay as Part-B through a PA override?",
        "Configure PA for Medicare Part B claim processing.",
        "Steps to route a pharmacy claim to Part B using PA reason MB.",
        "Part B override setup — which reason code?",
        "Set up PA with reason code MB for Part B.",
        # Additional real-world conversational
        "This claim should be Part B not Part D — how do I fix it with a PA?",
        "The claim needs to process under Medicare Part B, what PA reason code?",
        "Member needs this to go through Part B — what do I enter?",
        "Part B override — just tell me the reason code to use.",
        "How do I get this pharmacy claim to pay under Part B?",
    ],

    # ── pa_esrd_override: override ESRD reject ─────────────────────────
    "pa_esrd_override": [
        "How do I override the ESRD rejection using a PA?",
        "Set PA reason code ES to bypass the ESRD edit.",
        "The claim is rejecting for ESRD — what PA override do I use?",
        "Enter reason code ES on the PA for ESRD override.",
        "How to bypass End-Stage Renal Disease reject via PA?",
        "Which reason code overrides the ESRD edit?",
        "PA override for ESRD rejection — what do I enter?",
        "Use override reason ES for ESRD on this PA record.",
        "Can I use a PA to bypass the ESRD reject code?",
        "Steps to override ESRD using PA reason code ES.",
        "ESRD reject bypass — set up PA with reason ES.",
        "How to waive the ESRD edit for this member using PA?",
        "Configure PA to override ESRD rejection.",
        "The ESRD reject is blocking the claim — PA override steps?",
        "Override End-Stage Renal Disease using reason code ES.",
        # Additional real-world conversational
        "Member has ESRD and the claim is rejecting — how do I fix it?",
        "ESRD edit is stopping the claim — what PA do I need?",
        "The claim failed because of ESRD — what override reason do I use?",
        "How do I get past the ESRD edit? What reason code on the PA?",
        "Claim rejected for ESRD — PA override steps?",
    ],

    # ── pa_skip_deductible: waive deductible via PA ─────────────────────
    "pa_skip_deductible": [
        "How to set the skip deductible flag on this PA?",
        "Flip the skip DED indicator to Y on this override.",
        "I need to waive the member's deductible using this PA.",
        "Where is the skip deductible flag on the PA?",
        "Can I bypass the deductible through a PA override?",
        "Set skip DED to Y so the deductible is waived.",
        "Override the deductible using this PA.",
        "How do I enable skip deductible on the PA?",
        "Deductible waiver — what PA flag do I set?",
        "Turn on the skip DED flag on this prior authorization.",
        "I want to exempt this member from deductible via PA.",
        "Steps to skip the deductible on a PA override.",
        "Is there a PA flag that bypasses the deductible?",
        "Skip deductible indicator — how to set it on PA?",
        "Bypass deductible using PA skip DED flag.",
        # Additional real-world conversational
        "Member shouldn't have to pay the deductible — can the PA waive it?",
        "The deductible is applying to this claim — how do I skip it with the PA?",
        "I need to remove the deductible from this claim using PA.",
        "Can I use a PA to zero out the deductible?",
        "Deductible is too high for this member — PA skip option?",
    ],

    # ── pa_send_expiration: send PA expiration date on claim ────────────
    "pa_send_expiration": [
        "How to enable sending the PA expiration date on a claim?",
        "Flip the send expiration date flag to Y on this PA.",
        "I need to include the PA end date when the claim processes.",
        "Set send expiration date to Y on the override.",
        "Where do I turn on the send expiration date indicator?",
        "Configure PA to transmit the expiration date to the claim.",
        "Can I send the PA termination date with the claim?",
        "Steps to include PA expiration date during adjudication.",
        "Enable the send expiration flag on this PA record.",
        "How do I pass the PA end date to the claim?",
        "Turn on send expiration date on the PA.",
        "I want the PA expiration to appear on the claim.",
        "Set the expiration date flag to Y on this PA.",
        "Send PA end date on claim — what flag to set?",
        "Include PA expiration in claim processing.",
        # Additional real-world conversational
        "The pharmacy needs the PA expiration date — how do I include it?",
        "How do I get the PA end date to show up on the claim?",
        "The claim should carry the PA expiration — what flag?",
        "PA expiration date isn't showing on the claim — how to fix?",
        "I need the end date from the PA to transmit with the claim.",
    ],

    # ── pa_tf_letter_setup: configure TF letter tag on PA ───────────────
    "pa_tf_letter_setup": [
        "How to configure a TF letter tag on this PA?",
        "I need to set up a Transition Fill letter on this override.",
        "Choose the TF letter type for this PA override reason.",
        "Which TF letter type do I assign to this PA?",
        "Steps to add a TF letter tag to the PA.",
        "Set up Transition Fill letter for this prior authorization.",
        "Where do I select the TF letter type on the PA?",
        "Configure TF letter type with TF override reason.",
        "TF letter setup on PA — what type should I use?",
        "How do I assign the correct TF letter type?",
        "I want to set up a TF letter for this PA.",
        "Add TF letter tag to PA override.",
        "Transition Fill letter configuration for PA.",
        "What TF letter types are available for this PA?",
        "Setup TF letter on the PA override record.",
        # Additional real-world conversational
        "I need to attach a transition fill letter to this PA.",
        "Which letter type goes with a TF override reason?",
        "Member needs a TF letter — where do I set that up on the PA?",
        "Transition fill — how do I pick the right letter type?",
        "The PA needs a TF letter tag — how do I add it?",
    ],

    # ── pa_copay_setup: configure copay schedule on PA ──────────────────
    "pa_copay_setup": [
        "How to set up a different copay schedule on this PA?",
        "Configure a custom copay on the PA override.",
        "I need to change the copay schedule for this PA.",
        "Steps to assign a specific copay schedule to the PA.",
        "Where do I set the copay schedule on the PA override?",
        "Set up a different copay using this PA.",
        "Configure copay override schedule on the PA record.",
        "How do I process a different copay through a PA?",
        "Change the copay schedule on this prior authorization.",
        "I want to process a custom copay using this PA override.",
        "Copay schedule assignment on PA — how to configure?",
        "Set a specific copay schedule on the PA.",
        "How to change copay processing via PA?",
        "Configure the PA to use a different copay.",
        "Steps to set up copay override on PA.",
        # Additional real-world conversational
        "The member should be paying a different copay — can the PA change it?",
        "How do I make the claim use a different copay through the PA?",
        "Copay is wrong on this claim — can I fix it with the PA override?",
        "I need to change what the member pays — copay override on PA?",
        "Member's copay should be lower — how to set up the PA copay schedule?",
    ],

    # ── pa_suggest_override: find/recommend the right PA ────────────────
    "pa_suggest_override": [
        "What PA should I enter to fix this reject?",
        "Suggest the right PA override for this rejection.",
        "Use the suggest override feature for this claim.",
        "Which PA do I need to override this reject code?",
        "Help me find the correct PA to fix this rejection.",
        "I need a PA recommendation for this reject situation.",
        "Suggest override — find the appropriate PA for this claim.",
        "What PA should be applied to resolve this reject?",
        "Recommend a PA to override this claim rejection.",
        "Find the right PA for overriding this reject.",
        "PA suggestion for this reject situation.",
        "Which PA override resolves this reject code?",
        "Guide me to the correct PA for this reject.",
        "Use suggest override to find the right PA.",
        "What PA should I create for this reject?",
        # Additional real-world conversational
        "The claim keeps rejecting — which PA do I need to enter?",
        "I don't know what PA to use for this reject — help me out.",
        "Claim is rejected and I need a PA — which one?",
        "Tell me what PA to create to fix this rejection.",
        "What override do I need to get this claim through?",
    ],

    # ── pa_reason_code_fields: fields required per reason code ──────────
    "pa_reason_code_fields": [
        "What fields do I need for reason code OD?",
        "Which PA fields are required for override reason U1?",
        "Show the applicable fields for reason code LC.",
        "What should I fill in for PA reason code OA?",
        "Fields required for reason code US on a PA.",
        "Which fields are needed for reason code MB on the override?",
        "What PA fields apply to reason code ES?",
        "Show me the required fields for reason code HS.",
        "What fields are used with reason code PN?",
        "Map the applicable fields for reason code 2A.",
        "Which override fields do I configure for reason 2B?",
        "What fields are relevant for PA reason code 2C?",
        "Fields needed for this override reason code.",
        "What do I need to set up for reason code U3?",
        "Which fields are active for this PA override reason?",
        # Additional real-world conversational
        "I'm setting up reason code OD — what fields do I need to fill out?",
        "Which fields are mandatory for this reason code on the PA?",
        "I picked reason code MB — now what fields do I configure?",
        "Setting up a PA with reason code ES — what fields apply?",
        "Help me understand which fields go with this override reason.",
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
