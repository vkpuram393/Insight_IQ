# Flow Explanation - Intent Classification & Confidence Checking

## 🎯 Two Versions: Technical & Layman's Terms

---

## 📚 VERSION 1: Technical Explanation (For New Team Members)

### Overview
This system is a pharmacy claims assistant that uses AI to understand user questions and provide answers. The flow involves multiple decision points where the system decides how confident it is about understanding the user's intent.

### Complete Flow with Decision Points

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER SENDS QUERY                              │
│              "What's the status of claim 12345?"                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. ORCHESTRATOR                                                 │
│    - Normalizes text (removes punctuation, fixes typos)         │
│    - Generates request UUID                                     │
│    - Prepares state for processing                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. SAFETY PRECHECK                                              │
│    - Checks for harmful content                                 │
│    - Masks PII (Personal Identifiable Information)             │
│    - Validates safety with Gemini API                           │
│    Decision: BLOCKED → END | PASSED → Continue                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. CHECK CACHE                                                  │
│    - Looks for similar previous queries                         │
│    - Checks if answer exists in cache                          │
│    Decision: HIT → (future: return cached) | MISS → Continue     │
│    Note: Currently always continues to intent_agent             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. INTENT AGENT (First Classification)                         │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ DECISION POINT 1: Which Classifier to Use?              │  │
│    │                                                          │  │
│    │ Based on config flag:                                   │  │
│    │ • use_embedding_classifier = True  → Embedding Classifier│  │
│    │ • use_embedding_classifier = False → Keyword Classifier │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│    Classifier Returns:                                          │
│    • intent: "claim_status"                                     │
│    • confidence: 0.65 (65% sure)                               │
│    • entities: {"claim_number": "12345"}                       │
│    • missing_slots: [] (all required info present)             │
│    • is_complex: False (simple query)                          │
│                                                                  │
│    Sets: intent_reclassified = False (initial classification)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. CONFIDENCE CHECKER (First Decision Point)                   │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ DECISION POINT 2: Is Confidence High Enough?            │  │
│    │                                                          │  │
│    │ Checks:                                                  │  │
│    │ • intent_reclassified == False? (Yes - initial result) │  │
│    │ • confidence < threshold (0.7)? (Yes - 0.65 < 0.7)    │  │
│    │ • is_complex == True? (No)                              │  │
│    │                                                          │  │
│    │ DECISION TREE:                                           │  │
│    │                                                          │  │
│    │ IF intent_reclassified == False:                        │  │
│    │   IF confidence < 0.7 OR is_complex:                   │  │
│    │     → Route to: safety_precheck → llm_judge            │  │
│    │   ELSE IF confidence >= 0.7 AND entities present:       │  │
│    │     → Route to: build_context (API call)                │  │
│    │                                                          │  │
│    │ IF intent_reclassified == True:                         │  │
│    │   IF confidence >= 0.7 AND entities present:            │  │
│    │     → Route to: build_context (API call)                │  │
│    │   ELSE IF missing entities OR confidence < 0.7:         │  │
│    │     → Route to: clarification (template)                │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│    Current State:                                               │
│    • intent_reclassified = False                               │
│    • confidence = 0.65 (< 0.7 threshold)                      │
│    • Decision: Route to LLM Judge                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. SAFETY PRECHECK (Before LLM Judge)                          │
│    - Re-checks safety (required before LLM call)              │
│    - Masks PII again                                            │
│    Decision: BLOCKED → END | PASSED → Continue                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. LLM JUDGE (Second Classification)                           │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ DECISION POINT 3: Re-classify Intent with LLM           │  │
│    │                                                          │  │
│    │ Mock Implementation (for testing):                       │  │
│    │ • Reads config: llm_judge_mock_high_confidence           │  │
│    │ • If true: Returns confidence = 0.95 (high)            │  │
│    │ • If false: Returns confidence = 0.3 (low)              │  │
│    │                                                          │  │
│    │ Real Implementation (future):                           │  │
│    │ • Calls Gemini LLM with full context                    │  │
│    │ • LLM analyzes query more deeply                        │  │
│    │ • Returns re-classified intent and confidence           │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│    Updates State:                                               │
│    • intent: "claim_status" (keeps original)                     │
│    • confidence: 0.95 (updated - high confidence)              │
│    • entities: {"claim_number": "12345"} (keeps original)       │
│    • intent_reclassified = True (KEY FLAG - prevents loop)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. CONFIDENCE CHECKER (Second Decision Point - Re-evaluation)   │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ DECISION POINT 4: Re-evaluate After LLM Judge           │  │
│    │                                                          │  │
│    │ Checks:                                                  │  │
│    │ • intent_reclassified == True? (Yes - LLM judge ran)    │  │
│    │ • confidence >= threshold (0.7)? (Yes - 0.95 >= 0.7)  │  │
│    │ • entities present? (Yes - claim_number exists)          │  │
│    │                                                          │  │
│    │ DECISION TREE (After LLM Judge):                         │  │
│    │                                                          │  │
│    │ IF intent_reclassified == True:                         │  │
│    │   IF confidence >= 0.7 AND entities present:             │  │
│    │     → Route to: build_context (API call)                │  │
│    │   ELSE IF missing entities OR confidence < 0.7:          │  │
│    │     → Use template from domain_config.json               │  │
│    │     → Route to: clarification → END                     │  │
│    │                                                          │  │
│    │ IMPORTANT: Never routes to llm_judge again!             │  │
│    │ (intent_reclassified == True prevents infinite loop)     │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                                                  │
│    Current State:                                               │
│    • intent_reclassified = True (LLM judge already ran)        │
│    • confidence = 0.95 (>= 0.7 threshold)                      │
│    • entities: {"claim_number": "12345"} (present)              │
│    • Decision: Route to build_context (API call)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. BUILD CONTEXT                                                │
│    - Gathers conversation history                              │
│    - Prepares API call parameters                              │
│    - Extracts slots from conversation                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 10. CALL CLAIMS TOOL                                            │
│     - Makes real API call to CVS Claims API                    │
│     - Retrieves claim data                                      │
│     - Returns structured data                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 11. RESPONSE SAFETY PII PRECHECK                               │
│     - Masks PII before sending to LLM                          │
│     - Protects sensitive data                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 12. RESPONSE AGENT                                              │
│     - Uses Gemini LLM to generate natural language response    │
│     - Formats claim data into user-friendly answer             │
│     - Creates structured response                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 13. RESPONSE SAFETY PII POSTCHECK                               │
│     - Unmasks PII for user                                     │
│     - Checks for PII leakage                                   │
│     - Returns safe response                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 14. UPDATE MEMORY                                               │
│     - Stores conversation in memory                             │
│     - Saves for future context                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 15. CACHE RESPONSE                                              │
│     - Caches answer for similar future queries                 │
│     - Improves performance                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETURN RESPONSE TO USER                      │
│     "Your claim #12345 was approved on 05/15/2023..."          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Decision Points Explained

#### Decision Point 1: Which Classifier?
- **Embedding Classifier**: Uses AI embeddings to understand semantic meaning
- **Keyword Classifier**: Uses pattern matching on keywords
- **Decision**: Based on `use_embedding_classifier` config flag

#### Decision Point 2: First Confidence Check (After Initial Classifier)
- **Input**: `intent_reclassified = False`, `confidence = 0.65`, `entities = {"claim_number": "12345"}`
- **Logic**:
  ```
  IF intent_reclassified == False:
    IF confidence < 0.7 OR is_complex:
      → Route to LLM Judge (needs re-classification)
    ELSE IF confidence >= 0.7 AND entities present:
      → Route to API call (confident enough)
  ```
- **Decision**: Route to LLM Judge (0.65 < 0.7)

#### Decision Point 3: LLM Judge Re-classification
- **Input**: Original intent, low confidence
- **Process**: LLM analyzes query more deeply
- **Output**: Updated confidence (0.95), sets `intent_reclassified = True`
- **Why**: LLM has more context and reasoning ability than simple classifiers

#### Decision Point 4: Second Confidence Check (After LLM Judge)
- **Input**: `intent_reclassified = True`, `confidence = 0.95`, `entities = {"claim_number": "12345"}`
- **Logic**:
  ```
  IF intent_reclassified == True:
    IF confidence >= 0.7 AND entities present:
      → Route to API call (LLM is confident)
    ELSE IF missing entities OR confidence < 0.7:
      → Use template clarification (still uncertain)
    NEVER route to LLM judge again (flag prevents loop)
  ```
- **Decision**: Route to API call (0.95 >= 0.7, entities present)

### Infinite Loop Prevention

The `intent_reclassified` boolean flag prevents infinite loops:

```
Without Flag (BAD):
confidence_checker → llm_judge → confidence_checker → llm_judge → ... (infinite loop)

With Flag (GOOD):
confidence_checker (flag=False) → llm_judge (sets flag=True) → confidence_checker (flag=True, won't route to llm_judge again)
```

---

## 🗣️ VERSION 2: Layman's Terms (For Non-Technical Stakeholders)

### The Problem
Imagine you're a customer service agent helping people with pharmacy claims. Sometimes you're 100% sure what they're asking, sometimes you're only 60% sure. How do you handle uncertainty?

### The Solution: Multi-Stage Decision Making

Think of it like a **quality control process** with multiple checkpoints:

#### Stage 1: Initial Assessment (Fast but Less Accurate)
- **Like**: A quick glance at a document
- **What happens**: The system quickly reads the question and makes an initial guess
- **Example**: "What's my claim status?" → System thinks: "Probably asking about claim status, 65% sure"
- **Problem**: 65% isn't high enough confidence to proceed

#### Stage 2: Expert Review (Slower but More Accurate)
- **Like**: Asking a senior colleague to review
- **What happens**: When confidence is low, the system asks an AI "expert" (LLM Judge) to take a deeper look
- **Example**: LLM Judge analyzes: "User said 'claim status' and provided claim number 12345. This is definitely a claim status query. 95% confident."
- **Result**: Confidence increases from 65% to 95%

#### Stage 3: Final Decision (Based on Expert Review)
- **Like**: Making the final call based on expert's opinion
- **What happens**: System checks the expert's confidence level
- **If expert is confident (95%)**: Proceed with getting the answer
- **If expert is still uncertain (30%)**: Ask the user for clarification

### Real-World Analogy: Restaurant Order Taking

Imagine you're a waiter taking orders:

1. **Initial Assessment** (Fast Classifier):
   - Customer: "I'll have the... um... the thing with chicken"
   - You think: "Probably wants chicken dish, but not 100% sure" (65% confidence)
   - **Decision**: Ask the chef (LLM Judge) for help

2. **Expert Review** (LLM Judge):
   - Chef analyzes: "They said 'chicken' and we have 3 chicken dishes. Based on context, probably Chicken Parmesan" (95% confidence)
   - **Decision**: Proceed with Chicken Parmesan

3. **Final Check** (Second Confidence Check):
   - You check: "Chef is 95% sure, and we have all ingredients"
   - **Decision**: Confirm order and proceed

4. **If Still Uncertain**:
   - Chef says: "I'm only 30% sure, need more info" (low confidence)
   - **Decision**: Ask customer: "Which chicken dish would you like? We have Chicken Parmesan, Chicken Alfredo, or Grilled Chicken."

### The Multi-Decision Flow in Simple Terms

```
User asks: "What's my claim status?"

┌─────────────────────────────────────────┐
│ DECISION 1: Initial Guess               │
│ "I think they want claim status"         │
│ Confidence: 65% (not high enough)       │
│ Action: Ask expert for help              │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ DECISION 2: Expert Review                │
│ Expert analyzes more carefully          │
│ "Yes, definitely claim status"           │
│ Confidence: 95% (high enough)            │
│ Action: Proceed with answer              │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ DECISION 3: Final Check                 │
│ "Expert is 95% sure, we have claim #"   │
│ All requirements met                     │
│ Action: Get the answer and respond      │
└─────────────────────────────────────────┘
```

### Why Multiple Decisions?

**Single Decision Problem:**
- If we only had one decision point, we'd either:
  - Be too cautious (ask for clarification even when we're 95% sure)
  - Be too confident (proceed when only 60% sure, leading to wrong answers)

**Multi-Decision Solution:**
- **First decision**: Quick check (fast, but may be uncertain)
- **Second decision**: Expert review (slower, but more accurate)
- **Third decision**: Final validation (ensures we have everything needed)

### The "Flag" System (Preventing Confusion)

Imagine you're in a meeting:
- **Without flag**: You keep asking the same expert the same question over and over
- **With flag**: You ask once, get an answer, and mark it as "already asked" so you don't ask again

The `intent_reclassified` flag is like a sticky note that says:
- "Haven't asked expert yet" (False) → Can ask expert
- "Already asked expert" (True) → Don't ask again, use expert's answer

### Different Scenarios

#### Scenario 1: High Initial Confidence (No Expert Needed)
```
User: "What's the status of claim #12345?"
Initial Guess: 90% confident
Decision: Skip expert, proceed directly
Result: Fast response
```

#### Scenario 2: Low Initial Confidence → Expert Helps
```
User: "Tell me about my thing"
Initial Guess: 50% confident (too low)
Expert Review: 95% confident ("thing" = claim status)
Decision: Proceed with expert's answer
Result: Accurate response after expert review
```

#### Scenario 3: Expert Still Uncertain
```
User: "I need help"
Initial Guess: 40% confident
Expert Review: 30% confident (still too low)
Decision: Ask user for clarification
Result: "Could you provide your claim number?"
```

### Benefits of This Approach

1. **Speed**: Most queries get answered quickly (high initial confidence)
2. **Accuracy**: Uncertain queries get expert review (better answers)
3. **Efficiency**: Expert only consulted when needed (not for every query)
4. **Safety**: Never proceed when uncertain (prevents wrong answers)

---

## 🎯 Summary: The Multi-Decision System

### Technical Summary
- **3 Decision Points**: Initial classifier → LLM Judge → Final confidence check
- **2 Classification Stages**: Fast classifier → Expert LLM review
- **1 Safety Mechanism**: Boolean flag prevents infinite loops
- **Multiple Paths**: High confidence (direct), Low confidence (expert review), Still uncertain (clarification)

### Layman's Summary
- **Like a quality control process**: Quick check → Expert review → Final validation
- **Prevents mistakes**: Never proceed when uncertain
- **Efficient**: Only uses expert when needed
- **Smart routing**: Different paths based on confidence level

The system is designed to be both **fast** (for clear queries) and **accurate** (for uncertain queries), using a multi-stage decision process that ensures quality while maintaining efficiency.
