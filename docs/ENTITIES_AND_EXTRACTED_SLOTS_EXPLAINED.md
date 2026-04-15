# Entities and Extracted Slots - Who Populates Them and When

## 🎯 Quick Answer

**`entities`** - Populated by **Intent Agent Node** (from current message)  
**`extracted_slots`** - Populated by **Build Context Node** (from conversation history)

---

## 📊 The Flow: When Each Gets Populated

```
User Query: "What's my claim status?"
    ↓
┌─────────────────────────────────────────────────────────┐
│ 1. Intent Agent Node                                    │
│    (agents/intent_agent.py or                           │
│     agents/extended_intent_agent_node.py)               │
│                                                          │
│    ✅ POPULATES: state["entities"]                      │
│    📍 When: During intent classification                │
│    🔍 Source: CURRENT user message only                 │
│    📦 Example: {"claim_id": "12345"}                    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Confidence Checker → Router                          │
│    (nodes/confidence.py)                                │
│                                                          │
│    Routes to: build_context (if high confidence)        │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Build Context Node                                   │
│    (nodes/context.py)                                   │
│                                                          │
│    ✅ POPULATES: state["extracted_slots"]               │
│    📍 When: After intent classification, before API    │
│    🔍 Source: CONVERSATION HISTORY (previous messages) │
│    📦 Example: {"claim_id": "12345"} (from earlier msg)│
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Tool Call Agent Node                                 │
│    (tools/claims_api.py)                                │
│                                                          │
│    ✅ USES BOTH:                                        │
│       - state["entities"] (current message)             │
│       - state["extracted_slots"] (from history)        │
│    🔗 Merges them: {**extracted_slots, **entities}     │
│    💡 Why: Current entities take precedence            │
└─────────────────────────────────────────────────────────┘
```

---

## 1. `entities` - Populated by Intent Agent Node

### Who Populates It?

**Node:** `intent_agent_node` or `extended_intent_agent_node`  
**File:** `agents/intent_agent.py` or `agents/extended_intent_agent_node.py`

### When Does It Get Populated?

**Timing:** Early in the flow, during intent classification (right after the user sends a message)

### How Does It Work?

```python
# agents/extended_intent_agent_node.py - Lines 72-74

# Extract entities from the CURRENT user message
entity_result = extract_entities_unified(text, intent=intent)
entities = entity_result['entities']

# Return entities to state
result = {
    "intent": intent,
    "confidence": confidence,
    "entities": entities,  # ← Sets state["entities"]
    ...
}
```

**The extraction process:**
1. Takes the user's current message: `"What's the status of claim #12345?"`
2. Calls `extract_entities_unified()` which uses `EntityExtractor`
3. EntityExtractor uses **regex patterns** to find:
   - Claim IDs: `CLM12345` or `253152732536005` (15 digits)
   - Member IDs: `MEM123`
   - Prescription IDs: `RX123`
   - Dates, amounts, etc.
4. Returns entities found in **this message only**

### Example

**User Message:**
```
"What's the status of claim #12345?"
```

**After Intent Agent Node:**
```python
state = {
    "text": "What's the status of claim #12345?",
    "intent": "claim_status",
    "confidence": 0.95,
    "entities": {  # ← POPULATED HERE
        "claim_id": "12345"
    }
}
```

### Key Points

- ✅ **Only looks at current message** (not history)
- ✅ **Uses regex patterns** to extract structured data
- ✅ **Fast** (no API calls, just pattern matching)
- ✅ **Always populated** (even if empty `{}`)

---

## 2. `extracted_slots` - Populated by Build Context Node

### Who Populates It?

**Node:** `build_context_node`  
**File:** `nodes/context.py`

### When Does It Get Populated?

**Timing:** After intent classification, before the API call (if routed to `build_context`)

### How Does It Work?

```python
# nodes/context.py - Lines 97-101

# Get current slots from state (from intent classifier)
current_slots = state.get("slots") or {}

# Extract additional slots from conversation history
extracted_slots = _extract_slots_from_history(conversation_history, current_slots)

# Return extracted_slots to state
result = {
    "conversation_history": conversation_history,
    "relevant_facts": relevant_facts,
    "extracted_slots": extracted_slots,  # ← Sets state["extracted_slots"]
    "planner_context": planner_context
}
```

**The extraction process:**
1. Gets conversation history from memory store (previous messages in this session)
2. Calls `_extract_slots_from_history()` which delegates to `ConversationContextService`
3. `ConversationContextService.extract_entities_from_history()` uses **regex patterns** to scan **all previous messages** for:
   - Claim IDs: `"claim 12345"` or `"CLM12345"`
   - Member IDs: `"member ID ABC123"`
   - Dates: `"01/15/2024"`
   - Prescription IDs: `"prescription 12345"`
4. Returns entities found in **conversation history** (not current message)

### Example

**Conversation History:**
```
Message 1 (User): "What's the status of claim #12345?"
Message 2 (Assistant): "Your claim #12345 is paid."
Message 3 (User): "How much did I pay?"  ← Current message (no claim_id!)
```

**After Build Context Node:**
```python
state = {
    "text": "How much did I pay?",
    "intent": "pricing_info",
    "entities": {},  # ← Empty! (no claim_id in current message)
    "extracted_slots": {  # ← POPULATED HERE (from history!)
        "claim_number": "12345"  # Found in Message 1
    }
}
```

### Key Points

- ✅ **Looks at conversation history** (previous messages in the session)
- ✅ **Uses regex patterns** to extract from history
- ✅ **Only populated if routed to build_context** (not always present)
- ✅ **Enables follow-up questions** without re-asking for information

---

## 3. How They're Used Together in Tool Call Agent

### The Merge Logic

```python
# tools/claims_api.py - Lines 82-87

# CRITICAL: Merge extracted_slots (from conversation history) with current entities
# This enables follow-up questions without re-asking for claim_id
extracted_slots = state.get("extracted_slots", {})
current_entities = state.get("entities", {})

# Current entities take precedence over extracted ones
entities = {**extracted_slots, **current_entities}
```

**What this does:**
- Merges entities from **two sources**:
  1. `extracted_slots` - From conversation history (previous messages)
  2. `entities` - From current message
- **Priority:** Current entities **override** extracted slots (if both have the same key)

### Example Scenarios

#### Scenario 1: First Message (No History)

**User:** `"What's the status of claim #12345?"`

**State after Intent Agent:**
```python
{
    "entities": {"claim_id": "12345"}  # From current message
}
```

**State after Build Context:**
```python
{
    "entities": {"claim_id": "12345"},
    "extracted_slots": {}  # No history yet
}
```

**In Tool Call Agent:**
```python
extracted_slots = {}  # Empty (no history)
current_entities = {"claim_id": "12345"}
entities = {**{}, **{"claim_id": "12345"}}  # = {"claim_id": "12345"}
```

**Result:** Uses `claim_id` from current message ✅

---

#### Scenario 2: Follow-Up Question (Uses History)

**Message 1 (User):** `"What's the status of claim #12345?"`  
**Message 2 (Assistant):** `"Your claim #12345 is paid."`  
**Message 3 (User):** `"How much did I pay?"` ← Current message

**State after Intent Agent (Message 3):**
```python
{
    "text": "How much did I pay?",
    "entities": {}  # ← Empty! (no claim_id in current message)
}
```

**State after Build Context:**
```python
{
    "entities": {},
    "extracted_slots": {"claim_number": "12345"}  # ← Found in Message 1!
}
```

**In Tool Call Agent:**
```python
extracted_slots = {"claim_number": "12345"}  # From history
current_entities = {}  # Empty (current message has no entities)
entities = {**{"claim_number": "12345"}, **{}}  # = {"claim_number": "12345"}
```

**Result:** Uses `claim_number` from history! ✅  
**User doesn't need to repeat the claim ID!**

---

#### Scenario 3: User Changes Claim ID (Current Overrides History)

**Message 1 (User):** `"What's the status of claim #12345?"`  
**Message 2 (Assistant):** `"Your claim #12345 is paid."`  
**Message 3 (User):** `"What about claim #99999?"` ← Different claim!

**State after Intent Agent (Message 3):**
```python
{
    "text": "What about claim #99999?",
    "entities": {"claim_id": "99999"}  # ← New claim ID!
}
```

**State after Build Context:**
```python
{
    "entities": {"claim_id": "99999"},
    "extracted_slots": {"claim_number": "12345"}  # ← Old claim ID from history
}
```

**In Tool Call Agent:**
```python
extracted_slots = {"claim_number": "12345"}  # From history
current_entities = {"claim_id": "99999"}  # From current message
entities = {**{"claim_number": "12345"}, **{"claim_id": "99999"}}
# Result: {"claim_number": "12345", "claim_id": "99999"}
# But after normalization: {"claimId": "99999"} (current takes precedence)
```

**Result:** Uses `claim_id` from current message (overrides history) ✅  
**User can change the claim ID in follow-up questions!**

---

## 🔍 Code References

### Where `entities` is Set

```python
# agents/extended_intent_agent_node.py - Lines 72-128

# Extract entities from current message
entity_result = extract_entities_unified(text, intent=intent)
entities = entity_result['entities']

# Return to state
result = {
    "entities": entities,  # ← Sets state["entities"]
    ...
}
```

### Where `extracted_slots` is Set

```python
# nodes/context.py - Lines 97-149

# Extract slots from conversation history
extracted_slots = _extract_slots_from_history(conversation_history, current_slots)

# Return to state
result = {
    "extracted_slots": extracted_slots,  # ← Sets state["extracted_slots"]
    ...
}
```

### Where They're Used Together

```python
# tools/claims_api.py - Lines 82-87

# Merge extracted_slots (from history) with current entities
extracted_slots = state.get("extracted_slots", {})
current_entities = state.get("entities", {})

# Current entities take precedence
entities = {**extracted_slots, **current_entities}
```

---

## 📋 Summary Table

| Field | Populated By | When | Source | Example |
|-------|--------------|------|--------|---------|
| **`entities`** | Intent Agent Node | During intent classification | **Current message only** | `{"claim_id": "12345"}` |
| **`extracted_slots`** | Build Context Node | After intent, before API call | **Conversation history** | `{"claim_number": "12345"}` (from earlier message) |

### Key Differences

| Aspect | `entities` | `extracted_slots` |
|--------|------------|-------------------|
| **Source** | Current user message | Previous messages in session |
| **When** | Always (every request) | Only if routed to `build_context` |
| **Purpose** | Extract info from current query | Remember info from earlier messages |
| **Example Use** | "What's claim #12345 status?" → extracts `12345` | "How much did I pay?" → remembers `12345` from earlier |

---

## 🎯 Why This Design?

**Problem:** Users don't want to repeat information in follow-up questions.

**Example:**
```
User: "What's the status of claim #12345?"
Assistant: "Your claim #12345 is paid."
User: "How much did I pay?"  ← Doesn't mention claim ID again!
```

**Solution:**
1. **Intent Agent** extracts entities from current message → `entities`
2. **Build Context** extracts entities from history → `extracted_slots`
3. **Tool Call Agent** merges both → Uses claim ID from history if not in current message

**Result:** User can ask follow-up questions without repeating the claim ID! ✅

---

## 🔄 Complete Flow Example

### Request 1: First Message

```
User: "What's the status of claim #12345?"
    ↓
Intent Agent Node:
  - Extracts: entities = {"claim_id": "12345"}
  - Sets: state["entities"] = {"claim_id": "12345"}
    ↓
Build Context Node:
  - History: [] (empty, first message)
  - Extracts: extracted_slots = {}
  - Sets: state["extracted_slots"] = {}
    ↓
Tool Call Agent:
  - extracted_slots = {}
  - current_entities = {"claim_id": "12345"}
  - entities = {**{}, **{"claim_id": "12345"}} = {"claim_id": "12345"}
  - Uses: claim_id = "12345" ✅
```

### Request 2: Follow-Up Question

```
User: "How much did I pay?"
    ↓
Intent Agent Node:
  - Extracts: entities = {} (no claim_id in current message!)
  - Sets: state["entities"] = {}
    ↓
Build Context Node:
  - History: [
      {"role": "user", "content": "What's the status of claim #12345?"},
      {"role": "assistant", "content": "Your claim #12345 is paid."}
    ]
  - Scans history for "claim 12345"
  - Extracts: extracted_slots = {"claim_number": "12345"}
  - Sets: state["extracted_slots"] = {"claim_number": "12345"}
    ↓
Tool Call Agent:
  - extracted_slots = {"claim_number": "12345"}
  - current_entities = {}
  - entities = {**{"claim_number": "12345"}, **{}} = {"claim_number": "12345"}
  - Uses: claim_number = "12345" ✅ (from history!)
```

**User didn't need to repeat the claim ID!** 🎉

---

## 🛠️ Technical Details

### Entity Extraction (Current Message)

**Function:** `extract_entities_unified()`  
**File:** `classifiers/intent_classifier_wrapper.py`  
**Uses:** `EntityExtractor` from `utils/entity_extractor.py`

**Patterns:**
- Claim ID: `r'\b(CLM\d{3,10}|\d{15})\b'`
- Member ID: `r'\b(MEM\d{3,4})\b'`
- Prescription ID: `r'\b(RX\d{3,10})\b'`
- Dates, amounts, etc.

### Slot Extraction (History)

**Function:** `_extract_slots_from_history()`  
**File:** `nodes/context.py`  
**Delegates to:** `ConversationContextService.extract_entities_from_history()`  
**File:** `services/conversation_context.py`

**Patterns:**
- Claim ID: `r'(?:claim|clm)\s*(?:number|id|#)?\s*:?\s*(\d+)\b'`
- Member ID: `r'(?:member|patient)\s*(?:id|number)?\s*:?\s*([A-Z0-9\-]{6,20})\b'`
- Dates: `r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b'`

**Key Difference:**
- History extraction uses **keyword-aware patterns** (e.g., "claim 12345" not just "12345")
- This prevents false positives (won't match random numbers)

---

## ✅ Summary

1. **`entities`** = Entities from **current message** (populated by Intent Agent)
2. **`extracted_slots`** = Entities from **conversation history** (populated by Build Context)
3. **Tool Call Agent merges both** (current takes precedence)
4. **Enables follow-up questions** without repeating information

**That's how entities and extracted_slots work together!** 🎉

