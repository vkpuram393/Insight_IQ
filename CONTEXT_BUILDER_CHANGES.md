# Context Builder Implementation - Changes Summary

## Overview
Enhanced the context builder to create a comprehensive planner context object that contains all information needed for planner/executor to determine which API to call and what parameters to use.

## Changes Made

### 1. **State Schema Updates** (`state/schema.py`)
**Added new fields to `AgentState`:**
- `slots: Optional[Dict[str, Any]]` - API parameters from intent classifier
- `required_slots: Optional[List[str]]` - Required slots for the intent (from intent classifier)
- `missing_slots: Optional[List[str]]` - Required slots that are missing (from intent classifier)
- `extracted_slots: Optional[Dict[str, Any]]` - Slots extracted from conversation history (from context builder)
- `planner_context: Optional[Dict[str, Any]]` - Complete context object for planner/executor

**Why:** These fields allow the context builder to receive slots from the intent classifier and build a comprehensive context object for the planner/executor.

### 2. **Configuration Updates** (`config/domain_config.json`)
**Added:**
- `conversation_history_window: 5` - Configurable number of messages to include in context
- `required_slots_by_intent: {...}` - Mapping of intents to their required slots

**Why:** 
- Makes conversation history window configurable (not hardcoded)
- Centralizes required slots definition (scalable for multiple domains/APIs)
- Domain-agnostic - can be extended for any domain without code changes

### 3. **Context Builder Enhancement** (`nodes/context.py`)

#### New Function: `_extract_slots_from_history()`
- Extracts slots/entities from conversation history using regex patterns
- Generic patterns for numeric IDs, alphanumeric IDs, and dates
- Tries to infer slot names from context (e.g., "claim" → "claim_number", "prescription" → "prescription_number")
- Merges with current slots (current slots take precedence)

**Why:** Allows the system to find entities mentioned in previous messages, not just the current one.

#### Enhanced `build_context_node()`:
- Loads `conversation_history_window` from config
- Limits conversation history to last N messages (configurable)
- Extracts additional slots from conversation history
- Builds comprehensive `planner_context` object with:
  - Request metadata (request_id, session_id, timestamp, domain)
  - User information (user_id, profile placeholder)
  - Intent information (intent, confidence, original_text)
  - Entities and slots (filled, missing, required, extracted_from_history)
  - Conversation context (history, relevant_facts, counts)
- Logs context builder output and full planner context to SQLite

**Returns:**
- `conversation_history` - Last N messages
- `relevant_facts` - All facts from session
- `extracted_slots` - All slots (current + extracted from history)
- `planner_context` - Complete context object for planner/executor

**Why:** Provides planner/executor with all available information to make decisions about which API to call and what parameters to use.

### 4. **Confidence Checker Updates** (`nodes/confidence.py`)
**Changed:**
- Removed hardcoded logic for `claim_rejection_reason` requiring `claim_number`
- Now uses `missing_slots` from state (determined by intent classifier)
- Passes `slots`, `required_slots`, and `missing_slots` to context builder

**Why:** 
- Domain-agnostic - no hardcoded intent-specific logic
- Scalable - works for any intent/domain without code changes
- Separation of concerns - intent classifier determines required slots, confidence checker just routes

### 5. **Test Endpoint Updates** (`utils/test_endpoints.py`)
**Enhanced `/utils/test-context-building`:**
- New `ContextBuilderTestRequest` model that accepts full payload from confidence checker
- Accepts: intent, confidence, entities, slots, required_slots, missing_slots, domain, uuid, user_info
- Returns: conversation_history, relevant_facts, extracted_slots, planner_context

**Why:** Allows testing the context builder with the exact payload structure that confidence checker would pass.

## Domain-Agnostic Design

### ✅ Domain-Agnostic Elements:
1. **Slot extraction patterns** - Generic regex patterns that work for any domain (numeric IDs, alphanumeric IDs, dates)
2. **Required slots mapping** - Stored in config, not hardcoded in code
3. **Domain field** - Passed from orchestrator, no default value
4. **Planner context structure** - Generic structure that works for any domain

### ⚠️ Partially Domain-Specific (but configurable):
1. **Slot name inference** - Currently infers slot names from keywords ("claim" → "claim_number", "prescription" → "prescription_number")
   - **Note:** This is a fallback. The intent classifier should provide proper slot names.
   - **Future:** Could be moved to domain-specific config or removed if intent classifier always provides slots

2. **Relevant facts extraction** - In `update_memory_node`, there's a check for "claim" in text
   - **Note:** This is in the memory update node, not the context builder
   - **Future:** Should be made configurable or moved to domain-specific logic

### 🔴 Hardcoded Domain-Specific Logic (to be addressed):
1. **Clarification node** (`nodes/clarification.py`):
   - Hardcoded questions for specific intents ("claim_status", "claim_rejection_reason")
   - **Recommendation:** Move to config or make generic

2. **Memory update node** (`nodes/context.py` - `update_memory_node`):
   - Checks for "claim" in text to extract facts
   - **Recommendation:** Make this configurable or domain-specific

## Testing

### All Endpoints Tested: ✅ 16/16 Passed
- Health check
- Intent classification
- Cache operations
- Persistence/telemetry
- Session memory
- Context building (new enhanced version)
- Safety checks
- Clarification
- Confidence checker
- Claims API
- Response agent

### Context Builder Test Endpoint:
**Endpoint:** `POST /utils/test-context-building`

**Request Example:**
```json
{
  "text": "what is the status of claim 12345678",
  "intent": "claim_status",
  "confidence": 0.92,
  "entities": {"claim_number": "12345678"},
  "slots": {"claim_number": "12345678"},
  "required_slots": ["claim_number"],
  "missing_slots": [],
  "session_id": "test-session",
  "uuid": "req-uuid-123",
  "domain": "prescriptions",
  "user_info": {"user_id": "member_222"}
}
```

**Response Example:**
```json
{
  "session_id": "test-session",
  "conversation_history": [],
  "relevant_facts": [],
  "extracted_slots": {
    "claim_number": "12345678"
  },
  "planner_context": {
    "request_metadata": {
      "request_id": "req-uuid-123",
      "session_id": "test-session",
      "timestamp": "2025-11-16T17:17:52.393115",
      "domain": "prescriptions"
    },
    "user": {
      "user_id": "member_222",
      "profile": {}
    },
    "intent": {
      "intent": "claim_status",
      "confidence": 0.92,
      "original_text": "what is the status of claim 12345678"
    },
    "entities": {"claim_number": "12345678"},
    "slots": {
      "filled": {"claim_number": "12345678"},
      "missing": [],
      "required": ["claim_number"],
      "extracted_from_history": {}
    },
    "conversation": {
      "history": [],
      "relevant_facts": [],
      "history_length": 0,
      "facts_count": 0
    }
  },
  "timestamp": "2025-11-16T17:17:52.397463"
}
```

## Integration Points

### Input (from Confidence Checker):
- Intent, confidence, entities, slots from intent classifier
- Required slots and missing slots (determined by intent classifier)
- Domain, UUID, user info from orchestrator
- Conversation history from memory store

### Output (to Planner/Executor):
- Complete `planner_context` object with all available information
- Extracted slots from conversation history
- Conversation history and relevant facts

## Logging

The context builder logs:
1. **Context builder output** - Summary (history_length, facts_count, slots_count, missing_slots, intent)
2. **Planner context** - Full planner context object (for debugging/audit)

Both are logged to SQLite `logs` table with event types:
- `context_builder_output`
- `planner_context`

## Hardcoded Domain-Specific Logic Analysis

### ✅ Domain-Agnostic (No Hardcoded Logic):
1. **Context Builder** - Uses domain from state, no defaults
2. **Confidence Checker** - Uses slots from intent classifier, no hardcoded intent checks
3. **Required Slots** - Stored in config, not hardcoded
4. **Conversation History Window** - Configurable in config

### ⚠️ Partially Domain-Specific (Needs Attention):
1. **Slot Name Inference** (`nodes/context.py` - `_extract_slots_from_history()`):
   - Line 66-72: Infers slot names from keywords ("claim" → "claim_number", "prescription" → "prescription_number")
   - **Impact:** Low - This is a fallback. Intent classifier should provide proper slot names.
   - **Recommendation:** Keep as-is for now, or move to domain-specific config if needed

2. **Memory Update Fact Extraction** (`nodes/context.py` - `update_memory_node()`):
   - Line 285: Checks for "claim" in text to extract facts
   - **Impact:** Medium - Only affects fact extraction, not context building
   - **Recommendation:** Make configurable or move to domain-specific module

### 🔴 Hardcoded Domain-Specific Logic (Outside Context Builder):
1. **Clarification Node** (`nodes/clarification.py`):
   - Lines 39-41: Hardcoded questions for "claim_status", "claim_rejection_reason"
   - **Impact:** High - Not domain-agnostic
   - **Recommendation:** Move to config or make generic

**Note:** These are in other nodes, not in the context builder implementation.

## Next Steps

1. **Intent Classifier Update** (by teammate):
   - Extract slots (API parameters)
   - Determine required slots for intent
   - Return slots, required_slots, missing_slots in state

2. **Future Enhancements**:
   - Move slot name inference to domain-specific config (optional)
   - Make fact extraction in memory update configurable
   - Move clarification questions to config
   - Add domain-specific slot extraction patterns to config

## Files Modified

1. `state/schema.py` - Added new state fields
2. `config/domain_config.json` - Added conversation_history_window and required_slots_by_intent
3. `nodes/context.py` - Enhanced context builder with planner context creation
4. `nodes/confidence.py` - Removed hardcoded logic, uses slots from state
5. `utils/test_endpoints.py` - Enhanced test endpoint for context builder

## Files NOT Modified (as requested)

- `agents/intent_agent.py` - Reverted, teammate will update

