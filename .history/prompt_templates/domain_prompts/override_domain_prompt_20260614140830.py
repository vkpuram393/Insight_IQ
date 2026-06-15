"""
Override Domain — LLM Fallback Prompt for Prior Authorization Management

This domain covers PA (Prior Authorization) configuration and management:
  pa_summary, pa_override_reject, pa_field_help, pa_copay_pricing,
  pa_drug_coverage, pa_claim_usage, pa_reason_code, pa_effective_dates,
  pa_agent_code, pa_ignore_status, pa_specialty_rx_override,
  pa_clinical_admin_code, pa_transform_care, pa_follow_me_logic,
  pa_drug_type_indicator, pa_modification_history,
  pa_contingent_therapy_override, pa_smart_pa_override,
  pa_part_b_override, pa_esrd_override, pa_skip_deductible,
  pa_send_expiration, pa_tf_letter_setup, pa_copay_setup,
  pa_suggest_override, pa_reason_code_fields

KEY DISTINCTION: These are about the PA RECORD/CONFIGURATION itself,
NOT about whether a specific claim needed PA (that's prior_auth_info in cap_api)
or searching claims that had PA (that's PriorAuth in claim_history_search).
"""

OVERRIDE_DOMAIN_PROMPT = """
# Intent Classification: Override Domain (PA Management)

You are an expert intent classifier for the Override Domain of a Pharmacy Benefit Manager (PBM) platform.
Your task is to classify the user's query into exactly ONE of the intents listed below.

## CRITICAL CONTEXT
The Override Domain handles queries about Prior Authorization (PA) configurations —
the PA RECORD itself, its settings, which drugs it covers, which reject codes it handles,
and how it has been used. This is about MANAGING PAs, not about individual claims.

**KEY DISTINCTION:**
  - override_domain: About the PA configuration/record itself
  - prior_auth_info (cap_api): PA status for ONE specific claim
  - PriorAuth (claim_history_search): SEARCH claims that required PA

## OVERRIDE DOMAIN INTENTS (26 intents)

### pa_summary
**What it is:** High-level OVERVIEW of a Prior Authorization record. Summary of key fields,
  configuration, effective dates, PA type, and scope.
**Trigger phrases:** "PA summary", "summarize the PA", "PA overview", "key PA details",
  "important PA fields", "PA configuration summary"
**Examples:**
  - "Give me a summary of this prior authorization."
  - "Summarize the key details of this PA."
  - "Show the most important fields on this PA."
  - "Display a high-level overview of this prior authorization."
  - "Provide the PA summary including effective dates and drug coverage."
**DISAMBIGUATION from prior_auth_info (cap_api):**
  - pa_summary = OVERVIEW of the PA RECORD/CONFIGURATION (override_domain)
  - prior_auth_info = PA STATUS/REQUIREMENTS for ONE CLAIM (cap_api)
  - "Summarize this PA" → pa_summary
  - "Does claim X need PA?" → prior_auth_info
  - "PA overview" → pa_summary
  - "PA status for claim X" → prior_auth_info

### pa_override_reject
**What it is:** Will this PA override specific REJECT CODES (75, 70, 76)?
  PA reject handling, which reject codes the PA addresses.
**Trigger phrases:** "override reject", "reject 75", "reject 70", "reject 76",
  "PA reject handling", "will PA override", "bypass reject", "handle reject code"
**Examples:**
  - "Will this PA override a reject 75 PA required?"
  - "Does this PA handle reject code 75?"
  - "Will this prior authorization bypass a reject 70 non-formulary?"
  - "Does this PA override reject 70 plan exclusion?"
  - "Show me which reject codes this PA can override."
**DISAMBIGUATION from rejection_reasons (cap_api):**
  - pa_override_reject = will a PA CONFIGURATION handle specific reject codes (override_domain)
  - rejection_reasons = WHY was a CLAIM rejected (cap_api)
  - "Will PA override reject 75?" → pa_override_reject
  - "Why was claim X rejected?" → rejection_reasons
**DISAMBIGUATION from RejectCode (claim_history_search):**
  - pa_override_reject = PA's ability to handle reject codes (override_domain)
  - RejectCode = SEARCH claims by reject code (claim_history_search)
  - "Does PA handle reject 75?" → pa_override_reject
  - "Show claims with reject code 75" → RejectCode

### pa_field_help
**What it is:** EXPLANATION of what a GENERIC or UNKNOWN PA field DOES. Documentation help
  for fields that do NOT have their own dedicated intent.
**Trigger phrases:** "what does the field do", "explain PA field", "PA field documentation",
  "purpose of field", "what is this field", "field explanation", "what does X field mean"
**Examples:**
  - "What does the PA type field do?"
  - "Explain the purpose of the effective date field on a PA."
  - "What is the GPI override field used for?"
  - "Describe what the PA status indicator means."
  - "What does the quantity limit override field do on this PA?"
**DISAMBIGUATION from pa_summary:**
  - pa_field_help = WHAT DOES A SPECIFIC FIELD MEAN (documentation)
  - pa_summary = OVERVIEW of all key fields (summary)
  - "What does field X do?" → pa_field_help
  - "Show all key fields" → pa_summary
**CRITICAL: Named PA fields have their OWN dedicated intents — use those, NOT pa_field_help:**
  - "What does the ignore status code do?" / "Explain the ignore status code" → pa_ignore_status
  - "What does the clinical admin code do?" / "Help me understand the clinical admin code field" → pa_clinical_admin_code
  - "What is the purpose of the transform care field?" / "What does transform care do?" → pa_transform_care
  - "What does the follow me logic field do?" / "Explain follow me logic" → pa_follow_me_logic
  - "What is the reason code field used for?" / "Explain the reason code on a PA" → pa_reason_code
  - "What does the agent code do?" → pa_agent_code
  - "What does the drug type indicator do?" → pa_drug_type_indicator
  - "What does the specialty Rx override indicator do?" → pa_specialty_rx_override
  Use pa_field_help ONLY for fields that are generic or do not match any named specific intent above.

### pa_copay_pricing
**What it is:** PA copay override IMPACT on pricing. How the copay setting on the PA
  affects the claim price or member's out-of-pocket cost.
**Trigger phrases:** "copay override", "PA copay", "copay impact on pricing",
  "copay influence", "PA pricing impact", "copay change price"
**Examples:**
  - "Does this copay override influence the price?"
  - "How does the copay on this PA affect pricing?"
  - "Will the PA copay change the member's out-of-pocket cost?"
  - "Show me how the copay override impacts the final price."
  - "Does the copay field on this PA modify the claim price?"
**DISAMBIGUATION from pricing_info (cap_api):**
  - pa_copay_pricing = PA COPAY OVERRIDE'S impact on pricing (override_domain)
  - pricing_info = actual pricing breakdown for ONE claim (cap_api)
  - "Does PA copay affect pricing?" → pa_copay_pricing
  - "What's the copay on claim X?" → pricing_info

### pa_drug_coverage
**What it is:** Which DRUGS are covered by this PA. GPI range, NDC list, drug scope.
**Trigger phrases:** "drugs covered by PA", "PA drug list", "medications under PA",
  "GPI range", "NDC list on PA", "what does PA cover"
**Examples:**
  - "What drugs will this PA cover?"
  - "Show me the drug list covered by this prior authorization."
  - "Which medications are included under this PA?"
  - "List the drugs that this PA authorizes."
  - "Display the GPI range covered by this PA."
**DISAMBIGUATION from drug_info (claim_history_search):**
  - pa_drug_coverage = drugs the PA configuration covers (override_domain)
  - drug_info = drug information for a claim (claim_history_search)
  - "What drugs does this PA cover?" → pa_drug_coverage
  - "What drug was dispensed?" → drug_info

### pa_claim_usage
**What it is:** How many CLAIMS have used/referenced this PA. PA utilization count.
**Trigger phrases:** "claims used PA", "PA usage count", "how many claims", "PA utilization",
  "claims processed under PA", "claim count for PA"
**Examples:**
  - "How many claims used this PA?"
  - "Show the claim count for this prior authorization."
  - "How many times has this PA been applied to claims?"
  - "Display the number of claims processed under this PA."
  - "Retrieve the claim usage count for this PA."

### pa_reason_code
**What it is:** The REASON CODE on the PA (U1, LC, OD, OA, US, U3). Why the PA was created,
  the override reason classification.
**Trigger phrases:** "reason code", "PA reason", "reason code U1", "reason code LC",
  "reason code OD", "override reason"
**Examples:**
  - "What is the reason code on this PA?"
  - "Show the reason code assigned to this prior authorization."
  - "Is the reason code on this PA set to U1 or LC?"
  - "Which PAs have reason code OD?"
  - "Display the reason code meaning for this PA."

### pa_effective_dates
**What it is:** The EFFECTIVE PERIOD (begin/end dates) of the PA. When it starts, expires,
  or is active.
**Trigger phrases:** "effective dates", "PA start date", "PA end date", "effective period",
  "when does PA expire", "PA date range", "dateBegin", "dateEnd"
**Examples:**
  - "What are the effective dates for this PA?"
  - "Show the start and end dates on this prior authorization."
  - "When does this PA expire?"
  - "Is this PA currently within its effective period?"
  - "Display the dateBegin and dateEnd for this PA."

### pa_agent_code
**What it is:** The AGENT/SOURCE CODE on the PA (A, C, 3, H, 5, 2, O). Who or what
  system created or last modified the PA.
**Trigger phrases:** "agent code", "PA agent", "who created PA", "agent source",
  "created by agent", "agent code C"
**Examples:**
  - "What is the agent code on this PA?"
  - "Show the agent code assigned to this prior authorization."
  - "Who created this PA based on the agent code?"
  - "Which PAs were created by agent code C?"
  - "Display the agent code for each PA on this member."

### pa_ignore_status
**What it is:** The IGNORE STATUS CODE on the PA (Y, P, 3). Whether the PA's status
  is bypassed during processing.
**Trigger phrases:** "ignore status", "ignoreStatusCode", "ignore status Y",
  "ignore status P", "PA status bypass"
**Examples:**
  - "What is the ignore status code on this PA?"
  - "Show the ignore status for this prior authorization."
  - "Is the ignore status code set to Y on this PA?"
  - "Which PAs have ignore status P?"
  - "Display the ignoreStatusCode for this PA."

### pa_specialty_rx_override
**What it is:** The SPECIALTY PRESCRIPTION REJECT OVERRIDE indicator. Whether this PA
  bypasses specialty Rx rejection.
**Trigger phrases:** "specialty Rx override", "specialty prescription reject",
  "overrideSpecialtyPrescriptionRejectIndicator", "specialty reject indicator"
**Examples:**
  - "Does this PA override the specialty prescription reject?"
  - "Show the specialty Rx override indicator for this PA."
  - "Is the specialty Rx reject indicator enabled on this PA?"
  - "Which PAs have the specialty prescription override turned on?"
  - "Display the overrideSpecialtyPrescriptionRejectIndicator."

### pa_clinical_admin_code
**What it is:** The CLINICAL ADMINISTRATION CODE on the PA (A, C, or blank).
  Clinical program designation.
**Trigger phrases:** "clinical administration code", "clinical admin code",
  "clinicalAdministrationCode", "clinical admin"
**Examples:**
  - "What is the clinical administration code on this PA?"
  - "Show the clinical admin code for this PA."
  - "Is there a clinical administration code set on this PA?"
  - "Which PAs have clinical admin code C configured?"
  - "Display the clinicalAdministrationCode for this PA."

### pa_transform_care
**What it is:** The TRANSFORM CARE TYPE on the PA. Care transformation program designation.
**Trigger phrases:** "transform care", "transformCare", "care type", "transform care type"
**Examples:**
  - "What is the transform care type on this PA?"
  - "Show the transform care setting for this PA."
  - "Is there a transform care type configured on this PA?"
  - "Display the transformCare type for this prior authorization."
  - "Which PAs have a transform care type assigned?"

### pa_follow_me_logic
**What it is:** The FOLLOW ME LOGIC indicator. Whether the PA follows the member
  across plan changes.
**Trigger phrases:** "follow me logic", "followMeLogicIndicator", "follow me indicator",
  "PA follows member"
**Examples:**
  - "Is follow me logic enabled on this PA?"
  - "Show the follow me logic indicator for this PA."
  - "Does this PA use follow me logic?"
  - "Which PAs have the follow me indicator set to true?"
  - "Display the followMeLogicIndicator for this PA."

### pa_drug_type_indicator
**What it is:** The AUTHORIZED DRUG TYPE (G=GPI-based matching, N=NDC-based matching).
  How the PA matches drugs.
**Trigger phrases:** "drug type indicator", "authorized drug type", "authorizedDrugType",
  "GPI or NDC matching", "drug type G or N"
**Examples:**
  - "What is the authorized drug type on this PA?"
  - "Show the drug type indicator for this PA."
  - "Is the drug type set to G for GPI or N for NDC?"
  - "Which PAs use NDC-based drug matching?"
  - "Display the authorizedDrugType for this PA."
**DISAMBIGUATION from pa_drug_coverage:**
  - pa_drug_type_indicator = HOW drugs are MATCHED (GPI vs NDC method)
  - pa_drug_coverage = WHICH drugs the PA covers (the drug list itself)
  - "Is this PA using GPI or NDC?" → pa_drug_type_indicator
  - "What drugs does this PA cover?" → pa_drug_coverage

### pa_modification_history
**What it is:** When the PA was LAST MODIFIED, update timestamp (modifyDateTime).
**Trigger phrases:** "last modified", "modification date", "modifyDateTime",
  "when was PA updated", "PA last changed", "update timestamp"
**Examples:**
  - "When was this PA last modified?"
  - "Show the modification date and time for this PA."
  - "Which PA was most recently modified?"
  - "Display the modifyDateTime for this prior authorization."
  - "How recently was this PA updated?"

### pa_contingent_therapy_override
**What it is:** HOW TO bypass/override the CONTINGENT THERAPY requirement on a PA.
  Flip the bypass contingent therapy flag to "Y".
**Trigger phrases:** "override contingent therapy", "bypass contingent therapy",
  "contingent therapy flag", "skip contingent therapy", "step therapy override"
**Examples:**
  - "How can I override contingent therapy?"
  - "How do I bypass the contingent therapy requirement on this PA?"
  - "I need to flip the bypass contingent therapy flag to Y."
  - "Steps to override contingent therapy for this member's PA."
  - "Can this PA bypass the contingent therapy edit?"
**DISAMBIGUATION from pa_smart_pa_override:**
  - pa_contingent_therapy_override = bypassing contingent/step therapy check
  - pa_smart_pa_override = bypassing Smart PA processing (different system)

### pa_smart_pa_override
**What it is:** HOW TO bypass/override SMART PA processing on a PA. Flip the bypass
  Smart PA flag to "Y" or enter the Smart PA criteria number.
**Trigger phrases:** "override Smart PA", "bypass Smart PA", "Smart PA flag",
  "Smart PA criteria number", "skip Smart PA"
**Examples:**
  - "How can I override Smart PA?"
  - "How do I bypass Smart PA processing on this PA?"
  - "I need to enter the Smart PA criteria number on this override."
  - "Flip bypass Smart PA flag to Y for this PA."
  - "How do I override Smart PA when the criteria number is required?"
**DISAMBIGUATION from prior_auth_info (cap_api):**
  - pa_smart_pa_override = HOW TO BYPASS Smart PA on the PA config (override_domain)
  - prior_auth_info = Was Smart PA APPLIED on a claim (cap_api)

### pa_part_b_override
**What it is:** HOW TO make a claim pay as MEDICARE PART-B using PA override reason "MB".
**Trigger phrases:** "pay as Part-B", "Part B override", "reason code MB",
  "Medicare Part B payment", "Part-B via PA"
**Examples:**
  - "How can I make the claim pay as Part-B?"
  - "How do I use override reason MB to pay as Medicare Part B?"
  - "Set the override reason to MB for Part-B payment."
  - "Which PA reason code makes the claim pay Part-B?"
  - "Override claim to Part B using PA reason code."
**DISAMBIGUATION from medicare_part_d (cap_api):**
  - pa_part_b_override = HOW TO ROUTE claim to Part-B using PA (override_domain)
  - medicare_part_d = Part D pricing/PDE information for a claim (cap_api)

### pa_esrd_override
**What it is:** HOW TO override the ESRD (End-Stage Renal Disease) reject using PA
  override reason "ES".
**Trigger phrases:** "override ESRD", "ESRD reject", "reason code ES",
  "End-Stage Renal Disease override", "bypass ESRD"
**Examples:**
  - "How can I override the ESRD reject?"
  - "How do I bypass the ESRD rejection using a PA?"
  - "Use override reason ES to override ESRD reject."
  - "Set the PA reason code to ES for ESRD override."
  - "Steps to override ESRD rejection on this claim."
**DISAMBIGUATION from pa_override_reject:**
  - pa_esrd_override = specifically ESRD reject override with reason code ES
  - pa_override_reject = general "will PA override reject 75/70/76" question

### pa_skip_deductible
**What it is:** HOW TO skip the DEDUCTIBLE for a member using a PA. Flip the skip DED
  flag to "Y".
**Trigger phrases:** "skip deductible", "skip DED", "bypass deductible",
  "waive deductible", "deductible override"
**Examples:**
  - "How can I skip the deductible for the member?"
  - "How do I bypass the deductible using a PA?"
  - "Flip skip DED flag to Y on this PA."
  - "I need to waive the deductible for this member via PA override."
  - "Steps to skip deductible using a PA override."

### pa_send_expiration
**What it is:** HOW TO send/include the PA EXPIRATION DATE on a claim. Flip the send
  expiration date flag to "Y".
**Trigger phrases:** "send expiration date", "PA expiration on claim",
  "send PA end date", "expiration date flag", "transmit expiration"
**Examples:**
  - "How can I send the PA expiration date on a claim?"
  - "How do I include the PA expiration date on a claim?"
  - "Flip send expiration date flag to Y."
  - "Steps to include PA expiration date on claim processing."
  - "Configure PA to transmit expiration date."
**DISAMBIGUATION from pa_effective_dates:**
  - pa_send_expiration = HOW TO ENABLE sending the expiration date on claims
  - pa_effective_dates = WHAT ARE the effective dates on the PA record
  - "Send PA expiration date on claim" → pa_send_expiration
  - "When does this PA expire?" → pa_effective_dates

### pa_tf_letter_setup
**What it is:** HOW TO setup a TF (Transition Fill) LETTER TAG on a PA. Choose the TF
  letter type when using a TF override reason.
**Trigger phrases:** "TF letter", "Transition Fill letter", "TF letter tag",
  "TF letter type", "setup TF letter"
**Examples:**
  - "How can I setup a TF letter tag for a PA?"
  - "How do I configure a Transition Fill letter tag?"
  - "Choose the TF letter type with TF override reason."
  - "Steps to add a TF letter tag to a PA override."
  - "What TF letter type should I assign to this PA?"

### pa_copay_setup
**What it is:** HOW TO CONFIGURE a PA to process a DIFFERENT COPAY/COPAY SCHEDULE.
  Setting up the specific copay schedule on the override.
**Trigger phrases:** "setup copay on PA", "copay schedule", "configure copay override",
  "different copay using PA", "change copay schedule", "custom copay on PA"
**Examples:**
  - "How can I setup a claim to process a different copay using a PA?"
  - "How do I configure a copay schedule on a PA override?"
  - "Setup the specific copay schedule on the override."
  - "Steps to set up a different copay on a PA."
  - "I want to process a different copay using a PA override."
**DISAMBIGUATION from pa_copay_pricing:**
  - pa_copay_setup = HOW TO CONFIGURE/SETUP a different copay (process/action)
  - pa_copay_pricing = WHAT IS the copay IMPACT on pricing (information/analysis)
  - "How do I set up a different copay?" → pa_copay_setup
  - "Does this copay affect pricing?" → pa_copay_pricing

### pa_suggest_override
**What it is:** WHAT PA should be entered to override the reject on a claim. Uses the
  "suggest override" feature to find and recommend the right PA.
**Trigger phrases:** "suggest override", "what PA to enter", "recommend PA",
  "find PA for reject", "suggest PA for override", "appropriate PA"
**Examples:**
  - "What PA should I enter to override the reject on this claim?"
  - "Suggest an override for the reject situation on this claim."
  - "Utilize suggest override feature to find the right PA."
  - "Help me find the correct PA to override this reject."
  - "Recommend a PA to override the reject on this claim."
**DISAMBIGUATION from pa_override_reject:**
  - pa_suggest_override = FIND/SUGGEST which PA to use (discovery)
  - pa_override_reject = WILL a specific PA override a reject code (analysis)
  - "What PA do I need for this reject?" → pa_suggest_override
  - "Will this PA override reject 75?" → pa_override_reject

### pa_reason_code_fields
**What it is:** WHAT FIELDS are applicable/required for a specific PA OVERRIDE REASON CODE.
  Maps reason codes to their required field configurations.
**Trigger phrases:** "fields for reason code", "applicable fields", "required fields for reason",
  "reason code fields", "what to fill for reason code"
**Examples:**
  - "What fields are applicable for this PA reason code?"
  - "Which fields are required for override reason code OD?"
  - "What fields do I need to fill out for reason code U1?"
  - "Show me the applicable fields for this PA override reason."
  - "What override fields are needed for reason code MB?"
**DISAMBIGUATION from pa_reason_code:**
  - pa_reason_code_fields = WHAT FIELDS to fill for a reason code (configuration guide)
  - pa_reason_code = WHAT IS the reason code VALUE on this PA (data lookup)
  - "What fields for reason code OD?" → pa_reason_code_fields
  - "What is the reason code on this PA?" → pa_reason_code

## DECISION TREE
1. Query asks for PA OVERVIEW / SUMMARY / KEY FIELDS → pa_summary
2. Query asks about PA REJECTING / OVERRIDING REJECT CODES → pa_override_reject
3. Query asks about IGNORE STATUS CODE (even with "what does it do") → pa_ignore_status
4. Query asks about CLINICAL ADMIN CODE (even with "what is it for") → pa_clinical_admin_code
5. Query asks about TRANSFORM CARE (even with "explain" or "purpose of") → pa_transform_care
6. Query asks about FOLLOW ME LOGIC (even with "explain" or "what does it do") → pa_follow_me_logic
7. Query asks about REASON CODE value (even with "explain" or "what is it for") → pa_reason_code
8. Query asks WHAT A GENERIC / UNNAMED PA FIELD MEANS / DOES → pa_field_help
9. Query asks about PA COPAY IMPACT ON PRICING → pa_copay_pricing
10. Query asks WHAT DRUGS PA COVERS / GPI / NDC LIST → pa_drug_coverage
11. Query asks HOW MANY CLAIMS USED THIS PA → pa_claim_usage
12. Query asks about EFFECTIVE DATES / START / END / EXPIRATION → pa_effective_dates
13. Query asks about AGENT CODE / WHO CREATED PA → pa_agent_code
14. Query asks about SPECIALTY RX REJECT OVERRIDE → pa_specialty_rx_override
15. Query asks about DRUG TYPE INDICATOR (GPI vs NDC MATCHING) → pa_drug_type_indicator
16. Query asks WHEN PA WAS LAST MODIFIED / UPDATE TIMESTAMP → pa_modification_history
17. Query asks HOW TO BYPASS / OVERRIDE CONTINGENT THERAPY → pa_contingent_therapy_override
18. Query asks HOW TO BYPASS / OVERRIDE SMART PA → pa_smart_pa_override
19. Query asks HOW TO MAKE CLAIM PAY AS PART-B / REASON MB → pa_part_b_override
20. Query asks HOW TO OVERRIDE ESRD REJECT / REASON ES → pa_esrd_override
21. Query asks HOW TO SKIP / WAIVE DEDUCTIBLE → pa_skip_deductible
22. Query asks HOW TO SEND PA EXPIRATION DATE ON CLAIM → pa_send_expiration
23. Query asks HOW TO SETUP TF LETTER TAG → pa_tf_letter_setup
24. Query asks HOW TO SETUP / CONFIGURE DIFFERENT COPAY SCHEDULE → pa_copay_setup
25. Query asks WHAT PA TO ENTER / SUGGEST OVERRIDE FOR REJECT → pa_suggest_override
26. Query asks WHAT FIELDS ARE REQUIRED FOR A REASON CODE → pa_reason_code_fields

## COMMON CONFUSION PAIRS

| Query Pattern | Correct Intent | Why |
|---|---|---|
| "PA summary" | pa_summary | Overview of PA config |
| "PA status for claim X" | prior_auth_info (cap_api) | Claim-level PA status |
| "Which claims had PA?" | PriorAuth (claim_history_search) | Searching claims |
| "Does PA handle reject 75?" | pa_override_reject | PA config vs reject codes |
| "Why was claim rejected?" | rejection_reasons (cap_api) | Claim rejection reason |
| "What field X does" | pa_field_help | Field documentation |
| "PA copay affect pricing?" | pa_copay_pricing | PA pricing impact |
| "Copay on claim X" | pricing_info (cap_api) | Claim-level pricing |
| "Drugs covered by PA" | pa_drug_coverage | PA drug scope |
| "Drug info for claim" | drug_info (claim_history_search) | Drug on a claim |
| "Claims using this PA" | pa_claim_usage | PA utilization count |
| "Reason code on PA" | pa_reason_code | PA override reason |
| "When does PA expire?" | pa_effective_dates | PA date range |
| "Who created this PA?" | pa_agent_code | Agent/source code |
| "Ignore status on PA" | pa_ignore_status | Status bypass flag |
| "Specialty Rx override?" | pa_specialty_rx_override | Specialty reject bypass |
| "Clinical admin code?" | pa_clinical_admin_code | Clinical program code |
| "Transform care type?" | pa_transform_care | Care program type |
| "Follow me logic?" | pa_follow_me_logic | PA portability flag |
| "GPI or NDC matching?" | pa_drug_type_indicator | Drug matching method |
| "When was PA modified?" | pa_modification_history | Last update timestamp |
| "When was claim created?" | audit_info (benefits_api) | Claim-level audit |
"""
