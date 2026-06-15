# Override Domain Enhancement Prompt for VS Code Copilot

> **Copy everything below this line and paste it as a prompt in VS Code Copilot Chat (Agent mode recommended)**

---

## Task: Strengthen Override Domain Intent Detection in the Multidomain Intent Detection System

### Objective
Enhance the `override_domain` in the multidomain intent detection system to classify Prior Authorization (PA) override prompts with high confidence. Add new intents derived from real business user prompts (documented in `response.txt`), add comprehensive training examples, update confusion pairs, and ensure zero regression on `claims` (`cap_api`), `claim_history_search`, and all other existing domains.

---

### Context: Codebase Structure

The multidomain intent detection system lives in `multidomain_intent_detection/` with these key files:

| File | Purpose |
|------|---------|
| `config.py` | `INTENT_TO_DOMAIN`, `INTENT_DESCRIPTIONS`, `DOMAIN_ENDPOINTS`, `DOMAIN_NAMES` mappings |
| `intents_mapping.py` | `embeddingVars.CVS_INTENT_EXAMPLES` — base training examples per intent (~20 examples each) |
| `augmented_examples.py` | `AUGMENTED_EXAMPLES` — real-world phrasing variations per intent |
| `pipeline.py` | `CONFUSION_PAIRS` dict + 4-classifier ensemble (SVM, LogReg, kNN, ExtraTrees) |
| `training.py` | Training workflow: loads embeddings, augments, builds X/y, PCA search, trains pipeline. Has `DISABLED_DOMAINS` set |
| `normalizer.py` | Query normalization (strips claim/PA numbers, extracts entities via regex) |
| `llm_fallback.py` | LLM fallback classifier for low-confidence predictions |
| `classifier.py` | Production classifier API |
| `batch_test.py` | Dual-pass evaluation (ensemble-only + LLM hybrid) |
| `tuning_config.json` | All hyperparameters (PCA, classifiers, gating thresholds, confidence penalties) |

Supporting files:
- `config/api_routing_config.py` — intent-to-API endpoint routing
- `Overrides_api/intent_router.py` — 4-tier PA routing logic
- `response.txt` — Business user prompts and PA API response schema

---

### Existing Override Domain (16 intents)

Currently registered in `config.py` under `override_domain`:
`pa_summary`, `pa_override_reject`, `pa_field_help`, `pa_copay_pricing`, `pa_drug_coverage`, `pa_claim_usage`, `pa_reason_code`, `pa_effective_dates`, `pa_agent_code`, `pa_ignore_status`, `pa_specialty_rx_override`, `pa_clinical_admin_code`, `pa_transform_care`, `pa_follow_me_logic`, `pa_drug_type_indicator`, `pa_modification_history`

---

### Step 1: Add New Override Domain Intents

Based on the operational "How can I...?" questions in `response.txt` (Section B), add these **new intents** to the `override_domain`. These are how-to/action-oriented intents that DO NOT overlap with existing informational pa_* intents:

#### New Intents to Add:

1. **`pa_contingent_therapy_override`** — How to override/bypass contingent therapy requirement on a PA (flip bypass contingent therapy flag to "Y")
2. **`pa_smart_pa_override`** — How to override/bypass Smart PA processing (flip bypass Smart PA flag or enter criteria number)
3. **`pa_part_b_override`** — How to make a claim pay as Medicare Part-B using PA override reason "MB"
4. **`pa_esrd_override`** — How to override ESRD (End-Stage Renal Disease) reject using override reason "ES"
5. **`pa_skip_deductible`** — How to skip deductible for a member using PA (flip skip DED flag to "Y")
6. **`pa_send_expiration`** — How to send/include PA expiration date on a claim (flip send expiration date flag to "Y")
7. **`pa_tf_letter_setup`** — How to setup a TF (Transition Fill) letter tag on a PA (choose TF letter type with TF override reason)
8. **`pa_copay_setup`** — How to configure a PA to process a different copay/copay schedule (setup specific copay schedule on the override). NOTE: This is DISTINCT from `pa_copay_pricing` which asks about copay IMPACT on price — this intent is about HOW TO SET UP a copay override.
9. **`pa_suggest_override`** — What PA should be entered to override a reject on a claim? Uses the "suggest override" feature to guide the user. This is about finding/suggesting the right PA, not about a specific reject code.
10. **`pa_reason_code_fields`** — What fields are applicable/required for a specific PA override reason code? Maps reason codes to their field requirements.

---

### Step 2: Update `config.py`

#### 2a. Add new intents to `INTENT_TO_DOMAIN`:
```python
# ── override_domain (new operational how-to intents) ─────────────────
"pa_contingent_therapy_override": "override_domain",
"pa_smart_pa_override": "override_domain",
"pa_part_b_override": "override_domain",
"pa_esrd_override": "override_domain",
"pa_skip_deductible": "override_domain",
"pa_send_expiration": "override_domain",
"pa_tf_letter_setup": "override_domain",
"pa_copay_setup": "override_domain",
"pa_suggest_override": "override_domain",
"pa_reason_code_fields": "override_domain",
```

#### 2b. Add descriptions to `INTENT_DESCRIPTIONS`:
```python
"pa_contingent_therapy_override": "How to bypass/override contingent therapy on a PA, flip contingent therapy flag",
"pa_smart_pa_override": "How to bypass/override Smart PA processing, enter Smart PA criteria number",
"pa_part_b_override": "How to make a claim pay as Medicare Part-B using PA override reason MB",
"pa_esrd_override": "How to override ESRD reject using PA override reason ES",
"pa_skip_deductible": "How to skip deductible for a member using PA, flip skip DED flag",
"pa_send_expiration": "How to send PA expiration date on a claim, flip send expiration date flag",
"pa_tf_letter_setup": "How to setup TF (Transition Fill) letter tag on a PA, TF letter type configuration",
"pa_copay_setup": "How to configure a different copay schedule on a PA override, copay setup process",
"pa_suggest_override": "What PA to enter for overriding a reject, suggest override feature, find appropriate PA",
"pa_reason_code_fields": "What fields are applicable or required for a specific PA override reason code",
```

---

### Step 3: Add Training Examples to `intents_mapping.py`

Add to `embeddingVars.CVS_INTENT_EXAMPLES` — at least **20 diverse examples per new intent**. Examples should cover:
- Direct questions
- Conversational phrasing
- Variations with PA numbers (e.g., "PA 100000001")
- Variations with/without member context
- Imperative ("override contingent therapy") and interrogative ("how do I override contingent therapy?") forms

**Use these seed phrasings from `response.txt` as a starting point and generate 20+ variations for each:**

```python
"pa_contingent_therapy_override": [
    "How can I override contingent therapy?",
    "How do I bypass the contingent therapy requirement on this PA?",
    "I need to flip the bypass contingent therapy flag to Y.",
    "Override contingent therapy on PA 100000001.",
    "How to turn off contingent therapy check for this PA?",
    "Set bypass contingent therapy to Y on this prior authorization.",
    "Can I skip the contingent therapy step using a PA override?",
    "Steps to override contingent therapy for this member's PA.",
    "What do I need to do to bypass contingent therapy on this override?",
    "Where do I set the contingent therapy bypass on this PA?",
    "How do I disable contingent therapy validation on this PA?",
    "Enable contingent therapy bypass for PA override.",
    "Turn on bypass contingent therapy flag.",
    "I want to bypass contingent therapy for this claim.",
    "Contingent therapy override setup for PA.",
    "Can this PA bypass the contingent therapy edit?",
    "How to waive contingent therapy using a PA override?",
    "Override the step therapy requirement on this PA.",
    "Flip the contingent therapy flag for this override record.",
    "Steps to set bypass contingent therapy indicator.",
],

"pa_smart_pa_override": [
    "How can I override Smart PA?",
    "How do I bypass Smart PA processing on this PA?",
    "I need to enter the Smart PA criteria number on this override.",
    "Flip bypass Smart PA flag to Y for PA 100000001.",
    "How to turn off Smart PA for this prior authorization?",
    "Override Smart PA on this override record.",
    "Set bypass Smart PA to Y on this PA.",
    "How do I enter the criteria number for Smart PA override?",
    "Steps to bypass Smart PA processing on a PA.",
    "Where do I configure the Smart PA bypass flag?",
    "Disable Smart PA check on this PA override.",
    "Can I skip Smart PA using a PA override?",
    "Enter Smart PA criteria number on this PA.",
    "How to waive Smart PA processing for this member?",
    "Turn on bypass Smart PA flag for this override.",
    "Smart PA bypass configuration on PA.",
    "I want to override the Smart PA edit on this claim.",
    "What is the process to bypass Smart PA?",
    "Set the Smart PA criteria number on the override.",
    "How do I override Smart PA when the criteria number is required?",
],

"pa_part_b_override": [
    "How can I make the claim pay as Part-B?",
    "How do I use override reason MB to pay as Medicare Part B?",
    "I need to make this claim process as Part-B using a PA.",
    "Set the override reason to MB for Part-B payment.",
    "How to configure PA for Medicare Part-B payment?",
    "Use override reason MB on this PA to pay as Part-B.",
    "Make this claim pay under Medicare Part B.",
    "What PA reason code do I use for Part-B override?",
    "How do I set up a PA to process the claim as Part B?",
    "Override to Part-B using reason code MB.",
    "Steps to make a claim pay as Medicare Part B using PA.",
    "Which PA reason code makes the claim pay Part-B?",
    "I need to override this claim to process under Part B.",
    "How can I change this claim to pay as Part-B via PA?",
    "Enter reason code MB for Part-B override.",
    "Part-B payment override setup on PA.",
    "How do I make a pharmacy claim pay as medical Part B?",
    "Override claim to Part B using PA reason code.",
    "Set up PA override for Medicare Part B processing.",
    "Use MB reason code to route claim to Part B.",
],

"pa_esrd_override": [
    "How can I override the ESRD reject?",
    "How do I bypass the ESRD rejection using a PA?",
    "Use override reason ES to override ESRD reject.",
    "I need to override the End-Stage Renal Disease reject on this claim.",
    "Set the PA reason code to ES for ESRD override.",
    "What PA reason code overrides the ESRD edit?",
    "How to bypass the ESRD reject using a PA override?",
    "ESRD reject override using reason code ES.",
    "Steps to override ESRD rejection on this claim.",
    "Which reason code overrides ESRD?",
    "Enter reason code ES on the PA to override ESRD.",
    "I need to bypass the ESRD edit for this member.",
    "Override End-Stage Renal Disease reject.",
    "How do I configure a PA for ESRD override?",
    "PA setup for ESRD reject bypass.",
    "What do I need to enter to override the ESRD reject?",
    "Can I use a PA to bypass the ESRD rejection code?",
    "Override reason ES for ESRD on this PA.",
    "How to waive the ESRD edit using a PA override?",
    "Set up PA override for End-Stage Renal Disease.",
],

"pa_skip_deductible": [
    "How can I skip the deductible for the member?",
    "How do I bypass the deductible using a PA?",
    "Flip skip DED flag to Y on this PA.",
    "I need to waive the deductible for this member via PA override.",
    "Set skip deductible flag to Y.",
    "How to override the deductible on this PA?",
    "Skip deductible configuration on PA override.",
    "Where do I set the skip DED indicator?",
    "Can this PA bypass the deductible?",
    "Steps to skip deductible using a PA override.",
    "Override the member deductible using this PA.",
    "How do I turn on skip deductible on this override?",
    "Enable skip DED flag on PA.",
    "Waive deductible for this member using PA.",
    "I want to skip the deductible on this claim using a PA.",
    "How to set up a PA to skip deductible?",
    "Flip the deductible skip flag to Y.",
    "Is there a PA flag to bypass the deductible?",
    "How to exempt a member from deductible via PA?",
    "Set skip deductible indicator on this prior authorization.",
],

"pa_send_expiration": [
    "How can I send the PA expiration date on a claim?",
    "How do I include the PA expiration date on a claim?",
    "Flip send expiration date flag to Y.",
    "I need to send the PA expiration date with the claim.",
    "Set send expiration date to Y on this PA.",
    "How to configure PA to send the expiration date?",
    "Where do I enable the send expiration date flag?",
    "Turn on send expiration date on this PA override.",
    "Steps to include PA expiration date on claim processing.",
    "Enable the PA expiration date flag on this override.",
    "Can I send the PA end date on the claim?",
    "How to set the send expiration date indicator?",
    "Flip the expiration date flag to Y on this PA.",
    "Send PA termination date on the claim.",
    "How do I include the PA end date during claim adjudication?",
    "I want to send the expiration date from the PA on the claim.",
    "Configure PA to transmit expiration date.",
    "Set the send expiration flag on this prior authorization.",
    "How to enable PA expiration date on claim response?",
    "Turn on the send expiration date indicator for this PA.",
],

"pa_tf_letter_setup": [
    "How can I setup a TF letter tag for a PA?",
    "How do I configure a Transition Fill letter tag?",
    "Choose the TF letter type with TF override reason.",
    "I need to set up a TF letter on this PA.",
    "Setup Transition Fill letter tag on this override.",
    "How to configure TF letter type on a PA?",
    "Which TF letter type should I use on this PA?",
    "Steps to add a TF letter tag to a PA override.",
    "Set up Transition Fill letter on this prior authorization.",
    "How do I select the TF letter type for this PA?",
    "TF letter configuration for PA override.",
    "I need to set up a transition fill letter tag.",
    "Where do I configure the TF letter type on the PA?",
    "How to choose the correct TF letter type?",
    "Configure TF override reason with letter type.",
    "Set TF letter tag on PA for transition fill.",
    "What TF letter type should I assign to this PA?",
    "Add transition fill letter to this PA override.",
    "Setup TF letter for the override reason.",
    "How to assign a TF letter type to this PA record?",
],

"pa_copay_setup": [
    "How can I setup a claim to process a different copay using a PA?",
    "How do I configure a copay schedule on a PA override?",
    "Setup the specific copay schedule on the override.",
    "I need to change the copay on this PA override.",
    "How to set up a different copay schedule using PA?",
    "Configure copay override schedule on this PA.",
    "Steps to set up a different copay on a PA.",
    "Set the copay schedule on this prior authorization override.",
    "How do I assign a specific copay schedule to this PA?",
    "I want to process a different copay using a PA override.",
    "Where do I configure the copay schedule on the PA?",
    "Copay schedule setup for PA override.",
    "How to set up a custom copay on the PA?",
    "Change copay schedule on this PA override.",
    "Steps to assign a copay schedule to this PA.",
    "How do I process a different copay using this PA?",
    "Set a custom copay on the PA override record.",
    "Configure copay override on this prior authorization.",
    "I need to set up a specific copay schedule for this PA.",
    "How to change copay processing via PA override?",
],

"pa_suggest_override": [
    "What PA should I enter to override the reject on this claim?",
    "Suggest an override for the reject situation on this claim.",
    "Utilize suggest override feature to find the right PA.",
    "Which PA do I need to override this reject?",
    "I need to find the right PA to override this rejection.",
    "Use suggest override to guide me for an appropriate PA.",
    "What PA should I use for overriding this reject code?",
    "Suggest the correct PA for this claim rejection.",
    "How do I find the right PA to fix this reject?",
    "PA suggestion for overriding the reject on claim.",
    "Which PA override should I apply to this claim?",
    "Help me find the correct PA to override this reject.",
    "Suggest override for this reject situation.",
    "I need a PA recommendation to override this rejection.",
    "What PA should be applied to fix this reject code?",
    "Find the appropriate PA to override this claim reject.",
    "Suggest the best PA to override the rejection on this claim.",
    "What PA override resolves this reject?",
    "Use suggest override to find PA for this reject.",
    "Recommend a PA to override the reject on this claim.",
],

"pa_reason_code_fields": [
    "What fields are applicable for this PA reason code?",
    "Which fields are required for override reason code OD?",
    "What fields do I need to fill out for reason code U1?",
    "Show me the applicable fields for this PA override reason.",
    "What fields are relevant for reason code LC?",
    "Which PA fields apply to override reason OA?",
    "What should I fill in for PA reason code US?",
    "Fields applicable for reason code U3 on a PA.",
    "What override fields are needed for reason code MB?",
    "Show the required fields for PA reason code ES.",
    "Which fields are active for this override reason?",
    "What fields should I configure for this PA reason?",
    "Map the fields for this PA override reason code.",
    "Which PA fields are relevant to reason code HS?",
    "What are the required fields for reason code PN?",
    "Show me which fields apply for reason code 2A.",
    "Fields needed for PA reason code 2B.",
    "What fields are used with reason code 2C?",
    "Which override fields are applicable for this reason?",
    "What do I need to set up for this PA reason code?",
],
```

---

### Step 4: Add Augmented Examples to `augmented_examples.py`

Add to `AUGMENTED_EXAMPLES` dict — at least **15 additional real-world phrasing variations per new intent**. These should be:
- More conversational / natural language
- Include typos or informal speech patterns users actually use
- Include domain-specific jargon ("flip the flag", "set the indicator", "waive the edit")
- Include references to PA numbers, member IDs, claim numbers where appropriate
- Distinctly different from the base examples in `intents_mapping.py`

Generate these following the style of existing augmented examples in the file.

---

### Step 5: Update Confusion Pairs in `pipeline.py`

Add these new confusion pairs to `CONFUSION_PAIRS` dict. Focus on REAL confusability between new and existing intents:

```python
# New override how-to intent confusion pairs
"pa_contingent_therapy_override": {"pa_smart_pa_override", "pa_field_help", "pa_override_reject"},
"pa_smart_pa_override": {"pa_contingent_therapy_override", "pa_field_help", "prior_auth_info"},
"pa_part_b_override": {"pa_reason_code", "pa_esrd_override", "medicare_part_d"},
"pa_esrd_override": {"pa_part_b_override", "pa_reason_code", "pa_override_reject"},
"pa_skip_deductible": {"pa_copay_pricing", "pa_copay_setup", "pa_field_help"},
"pa_send_expiration": {"pa_effective_dates", "pa_field_help"},
"pa_tf_letter_setup": {"pa_copay_setup", "pa_reason_code", "approval_info"},
"pa_copay_setup": {"pa_copay_pricing", "pa_skip_deductible", "pa_tf_letter_setup"},
"pa_suggest_override": {"pa_override_reject", "pa_reason_code", "rejection_reasons"},
"pa_reason_code_fields": {"pa_reason_code", "pa_field_help", "pa_suggest_override"},
```

Also update EXISTING confusion pairs that now have new confusable neighbors:
- Add `"pa_suggest_override"` to `rejection_reasons` confusion set
- Add `"pa_part_b_override"` to `medicare_part_d` confusion set
- Add `"pa_copay_setup"` to `pa_copay_pricing`'s confusion set (if not already present or create one)
- Add `"pa_send_expiration"` to `pa_effective_dates`'s confusion set (if not already present or create one)

---

### Step 6: Update `normalizer.py` Entity Extraction

Ensure the normalizer correctly handles PA-specific patterns in user queries:
- PA numbers (already handled: `PA JW012726LC`, numeric PA numbers)
- Reason codes (OD, MB, ES, U1, LC, OA, US, U3, HS, PN, 2A, 2B, 2C)  
- Flag values (Y/N/P)
- Criteria numbers for Smart PA

Add a regex pattern to extract `reason_code` from queries like "reason code MB" or "override reason ES" if not already present.

---

### Step 7: Update `config/api_routing_config.py`

Add routing entries for each new intent:
```python
"pa_contingent_therapy_override": {
    "api_endpoint": "/myclaims/overrides/v1/pa",
    "method": "GET",
    "required_entities": ["pa_number"],
    "optional_entities": ["member_id"],
    "response_fields": ["bypassContingentTherapy"],
    "description": "Override contingent therapy on PA"
},
# ... similar entries for all 10 new intents
```

---

### Step 8: Update LLM Fallback in `llm_fallback.py`

If the LLM fallback has domain-specific decision trees for `override_domain`, add the new intents to that tree. The LLM should understand:
- **Informational intents** (pa_summary, pa_field_help, etc.) = "What is/does..."
- **Operational/How-To intents** (new ones) = "How can I...", "Steps to...", "Setup..."
- **Discovery intents** (pa_suggest_override) = "What PA should I...", "Suggest..."

---

### Step 9: Add Test Cases

Add labeled test rows to `testingFinalDataset.csv` (and optionally `Intent_detection_system/Testdata.csv`) for each new intent — at least **5 test prompts per new intent** covering:
- Clean direct question
- Conversational with PA number
- Edge case with overlapping language (e.g., "override" appearing in cap_api context)
- Short form (3-5 words)
- Long form with context

---

### Step 10: Verify Training Pipeline Compatibility

In `training.py`:
1. Make sure `override_domain` is **NOT** in `DISABLED_DOMAINS` (currently only `benefits_api` and `member_domain` are disabled — verify this remains unchanged)
2. Verify the `build_Xy()` function picks up all new intents from both `CVS_INTENT_EXAMPLES` and `AUGMENTED_EXAMPLES`
3. Verify `load_embeddings()` handles the new intent keys properly

---

### Critical Constraints (MUST follow)

1. **No overlap with `cap_api` intents**: The cap_api intent `prior_auth_info` is about PA status/requirements on a SINGLE CLAIM. The new `pa_*` intents are about PA CONFIGURATION and OPERATIONAL how-to questions. Keep this boundary clear in examples — never use "claim 12345 sequence 001" phrasing in override examples.

2. **No overlap with `claim_history_search`**: The `PriorAuth` intent searches claims that used PA. The new `pa_*` intents configure the PA itself. Keep this boundary clear.

3. **No overlap with existing `pa_*` intents**: 
   - `pa_copay_setup` (HOW to set up copay) ≠ `pa_copay_pricing` (WHAT is the copay impact)
   - `pa_reason_code_fields` (WHAT fields for a reason code) ≠ `pa_reason_code` (WHAT is the reason code value)
   - `pa_suggest_override` (FIND the right PA) ≠ `pa_override_reject` (WILL this PA override a reject)

4. **No changes to `cap_api`, `claim_history_search`, `benefits_api`, or `member_domain` intents, examples, or mappings.** Only add/modify under `override_domain`.

5. **Maintain example balance**: Each new intent should have 20+ base examples + 15+ augmented examples to match the existing training distribution.

6. **Test with existing test data**: After changes, verify that running `batch_test.py` on existing test data shows no regression. The new intents should not "steal" classifications from existing intents.

---

### Files to Modify (in order)

1. `multidomain_intent_detection/config.py` — Add 10 new intents to `INTENT_TO_DOMAIN` and `INTENT_DESCRIPTIONS`
2. `multidomain_intent_detection/intents_mapping.py` — Add 20+ base examples per new intent to `CVS_INTENT_EXAMPLES`
3. `multidomain_intent_detection/augmented_examples.py` — Add 15+ augmented examples per new intent to `AUGMENTED_EXAMPLES`
4. `multidomain_intent_detection/pipeline.py` — Add new entries and update existing entries in `CONFUSION_PAIRS`
5. `multidomain_intent_detection/normalizer.py` — Add reason_code regex extraction if missing
6. `config/api_routing_config.py` — Add routing entries for new intents
7. `multidomain_intent_detection/llm_fallback.py` — Update override_domain decision tree with new intents
8. `multidomain_intent_detection/testingFinalDataset.csv` — Add 5+ test rows per new intent

### Files to READ but NOT modify (verify compatibility)

- `multidomain_intent_detection/training.py` — Confirm `DISABLED_DOMAINS` does not include `override_domain`
- `multidomain_intent_detection/classifier.py` — Confirm classifier loads all intents dynamically
- `multidomain_intent_detection/embeddings.py` — No changes needed
- `Overrides_api/intent_router.py` — Verify new intents are picked up by tier-1 routing

---

### Expected Outcome

After all changes:
- Total override_domain intents: **26** (16 existing + 10 new)
- Total system intents: **~126** (116 existing + 10 new)
- Each new intent has **35+ total training examples** (20 base + 15 augmented)
- Confusion pairs updated for bidirectional confusability
- Zero regression on cap_api, claim_history_search, benefits_api, member_domain
- High confidence (>0.85) on override domain operational prompts
- Clear semantic separation between informational (existing) and operational (new) PA intents
