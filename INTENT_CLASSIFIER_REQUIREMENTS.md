# Intent Classifier Requirements

**Version**: 1.0  
**Purpose**: Define the requirements and implementation details for the CVS-specific intent classification system

---

## Table of Contents

1. [Overview](#overview)
2. [Intent Classification System](#intent-classification-system)
3. [Entity Extraction](#entity-extraction)
4. [Slot Validation & Management](#slot-validation--management)
5. [Confidence Scoring & Routing](#confidence-scoring--routing)
6. [Multi-Turn Conversations](#multi-turn-conversations)
7. [Integration Points](#integration-points)
8. [Best Practices](#best-practices)

---

## Overview

### Purpose
The intent classifier identifies user intent from natural language queries and extracts relevant entities to route requests to appropriate handlers within the PBM system.

### Core Components
1. **Intent Classifier** - Keyword-weighted classification with 28+ intents
2. **Entity Extractor** - Regex and LLM-based entity extraction
3. **Slot Validator** - Ensures required information is present
4. **Confidence Router** - Routes based on classification confidence
5. **Multi-Turn Handler** - Manages conversation continuity

### Data Models
Define Pydantic models in `state/schema.py`:
- `IntentResult` - Classification result with intent and confidence
- `EntityResult` - Extracted entities with validation status
- `SlotValidation` - Required vs. filled slots
- `SuggestedLink` - Follow-up action suggestions

---

## Intent Classification System

### 1. Core Intent Classifier

**Location**: `agents/intent_classifier.py`

**Supported Intents** (28+):
- **Claim Intents**: claim_status, rejection_reasons, date_range_claims, multi_claim_summary, audit_info
- **Drug Intents**: drug_info, compound_info, pricing_info, daw_info, dur_info
- **Coverage Intents**: coverage_info, prior_auth_info, network_info
- **Service Intents**: pharmacy_info, prescriber_info, mail_order_info
- **Member Intents**: member_info, prescription_history
- **Process Intents**: appeal_info
- **Meta Intents**: general_question, greeting, help, out_of_scope

### 2. Implementation Requirements

#### Keyword-Weighted Classification
```python
# Assign domain-specific keywords with confidence weights
keywords = {
    'claim_status': {
        'status': 0.9,      # High confidence indicator
        'where': 0.7,       # Location-specific
        'track': 0.6,       # Action-specific
        'check': 0.6
    },
    'rejection_reasons': {
        'rejected': 1.0,    # Strongest indicator
        'denied': 0.9,
        'refused': 0.8
    }
}
```

#### Text Normalization
```python
# Strip punctuation before matching
query_cleaned = re.sub(r'[^\w\s]', ' ', query_lower)

# Use whole-word matching to avoid false positives
pattern = r'\b' + keyword + r'\b'
```

#### Confidence Calculation
```python
# IMPORTANT: Normalize by matched keywords, not total words
score = sum(keyword_weights) / len(matched_keywords)

# NOT: sum(keyword_weights) / len(query_words)  # ❌ Wrong!
```

#### Threshold-Based Routing
- **Confidence ≥ 0.50**: Direct routing to intent handler
- **Confidence < 0.50**: Route to `out_of_scope` handler

---

## Entity Extraction

### 1. Regex-Based Extraction

**Location**: `agents/intent_agent.py` or separate `services/entity_extractor.py`

**Entity Types & Patterns**:

```python
ENTITY_PATTERNS = {
    # Claim IDs
    'claim_id': [
        r'CLM\d+',           # CLM12345678
        r'\b\d{8,12}\b'      # 8-12 digit numbers
    ],
    
    # Member IDs
    'member_id': [
        r'MBR\d+',           # MBR123456789
        r'\b\d{9,11}\b'      # 9-11 digit numbers
    ],
    
    # Prescription IDs
    'prescription_id': [
        r'RX\d+',            # RX12345
        r'[Pp]rescription\s+#?\s*\d+'
    ],
    
    # Phone Numbers
    'phone': [
        r'\d{3}-\d{3}-\d{4}',
        r'\(\d{3}\)\s*\d{3}-\d{4}'
    ],
    
    # Dollar Amounts
    'amount': [
        r'\$\d+\.?\d*',
        r'\d+\.?\d*\s*dollars?'
    ],
    
    # Dates
    'date': [
        r'\d{1,2}/\d{1,2}/\d{4}',        # MM/DD/YYYY
        r'\d{4}-\d{2}-\d{2}',             # YYYY-MM-DD
        r'(January|February|...|December)\s+\d{1,2},?\s+\d{4}',
        r'(last|this|next)\s+(week|month|year)',
        r'yesterday|today|tomorrow',
        r'Q[1-4]\s+\d{4}'                 # Q1 2025
    ]
}
```

**Features**:
- Support multiple entities per query (e.g., multiple claim IDs)
- Validate extracted entities (format checks, checksums)
- Return extraction method (`'regex'`) with results

### 2. LLM-Based Extraction (Fallback)

**Location**: `nodes/entity_extraction.py` (new node)

**When to Use**: When regex extraction fails to find required slots

**Implementation**:
```python
async def llm_entity_extraction_node(state: AgentState):
    """Extract entities using LLM when regex fails"""
    
    # Build prompt with conversation history
    prompt = f"""
    Based on the conversation history below, extract the following entities:
    {missing_slots}
    
    Conversation:
    {last_5_messages}
    
    Return JSON format: {{"claim_id": "...", "member_id": "..."}}
    """
    
    # Get LLM response
    result = await llm.ainvoke(prompt)
    
    # Parse and validate
    entities = parse_json(result)
    
    return {
        "entities": entities,
        "extraction_method": "llm"
    }
```

**Fallback Chain**:
1. Regex extraction (fast, accurate for structured data)
2. LLM extraction from history (slower, handles context)
3. Ask user directly (last resort)

---

## Slot Validation & Management

### 1. Required Slots by Intent

**Location**: `core/config.py` or `services/template_registry.py`

```python
REQUIRED_SLOTS = {
    'claim_status': ['claim_number'],
    'rejection_reasons': ['claim_number'],
    'rx_details': ['prescription_id'],
    'member_info': ['member_id'],
    'drug_info': ['drug_name'],
    'date_range_claims': ['start_date', 'end_date'],
    'multi_claim_summary': ['member_id']
}

OPTIONAL_SLOTS = {
    'claim_status': ['member_id'],  # Improves accuracy
    'drug_info': ['member_id']       # For personalized info
}
```

### 2. Slot Validation Node

**Location**: `nodes/slot_validation.py` (new node)

```python
async def slot_validation_node(state: AgentState) -> Dict[str, Any]:
    """Validate that all required slots are filled"""
    
    intent = state["intent"]
    entities = state.get("entities", {})
    
    required = REQUIRED_SLOTS.get(intent, [])
    missing = [slot for slot in required if slot not in entities]
    
    if missing:
        return {
            "missing_slots": missing,
            "slot_validation_passed": False,
            "next_action": "extract_from_llm"  # or "ask_user"
        }
    
    return {
        "missing_slots": [],
        "slot_validation_passed": True,
        "next_action": "proceed"
    }
```

### 3. Routing Logic

```python
def route_after_slot_validation(state: AgentState) -> str:
    """Route based on slot validation result"""
    
    if state.get("missing_slots"):
        # Try LLM extraction first
        if not state.get("llm_extraction_attempted"):
            return "llm_entity_extraction"
        
        # If LLM failed, ask user
        return "ask_user_for_slots"
    
    # All slots filled, proceed
    return "data_cache_lookup"
```

---

## Confidence Scoring & Routing

### 1. Confidence Thresholds

```python
CONFIDENCE_THRESHOLDS = {
    'HIGH': 0.75,      # Direct routing, confident response
    'MEDIUM': 0.60,    # Proceed with hedging language
    'LOW': 0.50        # Minimum for routing, else out_of_scope
}
```

### 2. Routing Rules

**High Confidence (≥ 0.75)**:
- Route directly to intent handler
- Use confident language in responses
- No additional validation needed

**Medium Confidence (0.60 - 0.75)**:
- Proceed to intent handler
- Add hedging language: "It looks like...", "This might be..."
- Log for confidence calibration

**Low Confidence (< 0.60)**:
- Route to LLM double-check node
- Or route to out_of_scope handler
- Ask clarifying questions if needed

### 3. Ambiguity Detection

```python
def detect_ambiguity(intents_with_scores):
    """Detect when top 2 intents are too close"""
    
    if len(intents_with_scores) < 2:
        return False
    
    top1_score = intents_with_scores[0]['confidence']
    top2_score = intents_with_scores[1]['confidence']
    
    # If difference is less than 0.1, it's ambiguous
    if abs(top1_score - top2_score) < 0.1:
        return True, intents_with_scores[0], intents_with_scores[1]
    
    return False
```

**Clarification Response**:
```
"I'm not sure if you want to:
1. Check your claim status
2. Understand rejection reasons

Which one would you like help with?"
```

---

## Multi-Turn Conversations

### 1. Conversation State Management

**State Fields** (in `AgentState`):
```python
class AgentState(TypedDict):
    # ...existing fields...
    waiting_for_slot: Optional[str]      # Which slot we're waiting for
    pending_intent: Optional[str]        # Intent to resume after slot fill
    continuation_context: Optional[Dict] # Full context for resumption
```

### 2. Slot Filling Flow

**Turn 1 - Missing Slot**:
```
User: "What's my claim status?"
Bot: "I need your claim number to check the status. What's your claim number?"
State: waiting_for_slot='claim_number', pending_intent='claim_status'
```

**Turn 2 - Slot Provided**:
```
User: "12345678"
Bot: (Skip intent classification, use pending_intent='claim_status')
     (Extract claim_number from "12345678")
     (Proceed with claim_status handler)
State: Clear waiting_for_slot and pending_intent
```

### 3. Intent Change Detection (Critical!)

**Problem**: User changes their mind mid-conversation

```python
async def check_intent_change(state: AgentState, new_query: str):
    """Detect if user is changing intent while waiting for slot"""
    
    # Only check if we're waiting for a slot
    if not state.get("waiting_for_slot"):
        return False
    
    # Classify the new query
    new_intent_result = await classify_intent(new_query, {})
    
    # If new intent has high confidence and is different
    if (new_intent_result['confidence'] >= 0.70 and 
        new_intent_result['intent'] != state['pending_intent']):
        
        # Clear continuation state
        return True, new_intent_result['intent']
    
    return False, None
```

**Example**:
```
Turn 1: "What's my claim status?"
Bot: "What's your claim number?"
State: waiting_for_slot='claim_number'

Turn 2: "Never mind, how do I appeal?" (High confidence different intent)
Bot: (Clear continuation, honor new intent='appeal_info')
     "Here's how to file an appeal..."
```

### 4. Continuation State Persistence

**Save State**:
```python
# In api/routes.py
session.metadata = {
    'waiting_for_slot': state.get('waiting_for_slot'),
    'pending_intent': state.get('pending_intent'),
    'continuation_context': state.get('continuation_context')
}
```

**Restore State**:
```python
# In next request
if session.metadata.get('waiting_for_slot'):
    state['waiting_for_slot'] = session.metadata['waiting_for_slot']
    state['pending_intent'] = session.metadata['pending_intent']
```

### 5. Clearing Continuation State

Clear when:
1. ✅ API call succeeds and response is delivered
2. ✅ User changes intent (high confidence different intent)
3. ✅ RAG path is taken (general question)
4. ✅ User says "cancel", "nevermind", "stop"
5. ✅ More than 5 turns without success (prevent infinite loops)

---

## Integration Points

### 1. Data Cache Integration

**Node**: `nodes/data_cache.py` (new)

```python
async def data_cache_lookup_node(state: AgentState):
    """Check if we have cached API response"""
    
    cache_key = f"{state['session_id']}:{state['entities']['claim_id']}:{state['intent']}"
    
    cached_data = await cache.get(cache_key)
    
    if cached_data:
        return {
            "tool_result": cached_data,
            "cache_hit": True,
            "next_action": "render_template"
        }
    
    return {
        "cache_hit": False,
        "next_action": "call_api"
    }
```

**TTL**: 5-10 minutes per cache entry

### 2. Query Focus Detection

**Node**: `nodes/query_focus.py` (new)

```python
async def query_focus_detection_node(state: AgentState):
    """Detect what aspect of the claim user is asking about"""
    
    query = state["text"].lower()
    
    focus_keywords = {
        'drug': ['drug', 'medication', 'prescription', 'pill'],
        'pricing': ['cost', 'price', 'pay', 'charge', 'copay'],
        'status': ['status', 'where', 'when', 'track'],
        'reason': ['why', 'reason', 'rejected', 'denied']
    }
    
    # Score each focus
    focus_scores = {}
    for focus, keywords in focus_keywords.items():
        score = sum(1 for kw in keywords if kw in query)
        focus_scores[focus] = score
    
    # Get top focus
    top_focus = max(focus_scores, key=focus_scores.get)
    
    return {
        "query_focus": top_focus,
        "focus_scores": focus_scores
    }
```

**Template Selection**:
```python
# intent=claim_status + focus=drug → drug_details_template
# intent=claim_status + focus=pricing → pricing_details_template
```

### 3. Suggested Links Generation

**Location**: `agents/intent_classifier.py` or separate service

```python
def generate_suggested_links(intent: str, user_history: List) -> List[SuggestedLink]:
    """Generate 3-5 follow-up action suggestions"""
    
    # Define common follow-ups per intent
    FOLLOW_UPS = {
        'claim_status': ['rejection_reasons', 'drug_info', 'appeal_info'],
        'rejection_reasons': ['appeal_info', 'prior_auth_info', 'coverage_info'],
        'drug_info': ['pricing_info', 'pharmacy_info', 'coverage_info']
    }
    
    suggestions = []
    
    # Add pre-defined follow-ups
    for follow_up_intent in FOLLOW_UPS.get(intent, [])[:2]:
        suggestions.append(SuggestedLink(
            intent=follow_up_intent,
            label=get_intent_label(follow_up_intent),
            relevance_score=0.8
        ))
    
    # Add popular follow-ups from memory (MongoDB)
    popular = query_follow_up_memory(intent, limit=2)
    suggestions.extend(popular)
    
    # Add relevant FAQs (if available)
    faqs = query_faq_repository(intent, limit=2)
    suggestions.extend(faqs)
    
    # Deduplicate and limit to 5
    unique_suggestions = deduplicate(suggestions, key='intent')
    return unique_suggestions[:5]
```

### 4. PII Handling

**Mask PII** before logging:
```python
def mask_pii(text: str) -> str:
    """Mask sensitive information"""
    
    # Mask claim IDs
    text = re.sub(r'\b\d{8,12}\b', 'CLM***', text)
    
    # Mask member IDs
    text = re.sub(r'\bMBR\d+', 'MBR***', text)
    
    # Mask phone numbers
    text = re.sub(r'\d{3}-\d{3}-\d{4}', '***-***-****', text)
    
    return text
```

---

## Best Practices

### 1. Keyword Weight Tuning

#### ❌ Don't Do This:
```python
# Using generic keywords with low weights
'rejection_reasons': {
    'claim': 0.8,    # Too generic!
    'why': 0.3,      # Too generic!
    'rejected': 1.0
}
```

**Problem**: Generic keywords dilute the average score
- Query: "why was my claim rejected"
- Matched: 'why' (0.3) + 'claim' (0.8) + 'rejected' (1.0) = 2.1 / 3 = **0.70**
- But 'claim_status' with just 'claim' (0.8) = **0.80** wins! ❌

#### ✅ Do This Instead:
```python
# Use specific, high-weight keywords only
'rejection_reasons': {
    'rejected': 1.0,
    'denied': 0.9,
    'refused': 0.8
}

'claim_status': {
    'status': 0.9,
    'where': 0.7,
    'track': 0.6
}
```

**Result**: Clear separation between intents
- "why was my claim rejected" → matches 'rejected' (1.0) in rejection_reasons ✅
- "where is my claim" → matches 'where' (0.7) in claim_status ✅

### 2. Avoid Keyword Overlap

#### ❌ Don't Share Keywords:
```python
'claim_status': ['claim', 'status'],
'rejection_reasons': ['claim', 'rejected']  # 'claim' appears in both!
```

#### ✅ Use Differentiators:
```python
'claim_status': ['status', 'where', 'track', 'check'],
'rejection_reasons': ['rejected', 'denied', 'refused', 'why']
```

### 3. Test for False Positives

After adding new keywords, test with queries that shouldn't match:
```python
test_cases = [
    ("I want to exclaim something", should_not_match='claim_status'),
    ("claim my rewards", should_not_match='claim_status'),
    ("proclaimed winner", should_not_match='claim_status')
]
```

### 4. Monitor Confidence Distribution

Track confidence scores to identify issues:
- **Too many low scores** (< 0.6): Keywords too weak or generic
- **Scores cluster at threshold**: Need better separation
- **Always high scores**: Might be overfitting on keywords

### 5. Conversation Continuity Best Practices

✅ **Always detect intent changes** to prevent infinite loops
✅ **Clear continuation state** after success
✅ **Limit slot-filling attempts** (max 3-5 turns)
✅ **Persist state** to survive page refreshes
✅ **Provide escape hatches** ("cancel", "start over")

---

## Implementation Checklist

### Phase 1: Core Classification
- [ ] Implement keyword-weighted classifier
- [ ] Define 28+ intents with weights
- [ ] Add whole-word matching (regex boundaries)
- [ ] Implement confidence calculation
- [ ] Add threshold-based routing

### Phase 2: Entity Extraction
- [ ] Implement regex patterns for all entity types
- [ ] Add entity validation logic
- [ ] Support multiple entities per query
- [ ] Create LLM fallback extraction node

### Phase 3: Slot Management
- [ ] Define required/optional slots per intent
- [ ] Create slot validation node
- [ ] Implement routing based on slot status
- [ ] Add "ask user" fallback flow

### Phase 4: Multi-Turn Support
- [ ] Add continuation state to AgentState
- [ ] Implement intent change detection
- [ ] Create state persistence logic
- [ ] Add continuation clearing rules

### Phase 5: Integration
- [ ] Add data cache lookup node
- [ ] Implement query focus detection
- [ ] Create suggested links generation
- [ ] Add PII masking for logs

### Phase 6: Testing & Tuning
- [ ] Test all 28+ intents with sample queries
- [ ] Tune keyword weights for accuracy
- [ ] Test multi-turn conversations
- [ ] Verify intent change detection
- [ ] Monitor confidence distribution

---

## Testing Examples

### Test Intent Classification
```bash
curl -X POST http://localhost:8000/utils/test-intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"why was my claim rejected"}'

# Expected: intent="rejection_reasons", confidence >= 0.75
```

### Test Entity Extraction
```bash
curl -X POST http://localhost:8000/utils/test-intent-agent \
  -H 'Content-Type: application/json' \
  -d '{"text":"Claim 12345678 status"}'

# Expected: entities={"claim_number": "12345678"}
```

### Test Multi-Turn
```bash
# Turn 1
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"text":"What is my claim status?","session_id":"test"}'

# Expected: Response asks for claim number

# Turn 2
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"text":"12345678","session_id":"test"}'

# Expected: Returns claim status for 12345678
```

---

## Summary

The intent classifier is the foundation of the chatbot's routing logic. It must:

1. **Accurately classify** user intent with ≥0.75 confidence for direct routing
2. **Extract entities** using regex first, LLM as fallback
3. **Validate slots** and handle missing information gracefully
4. **Manage multi-turn** conversations with proper state management
5. **Detect intent changes** to prevent infinite loops
6. **Integrate seamlessly** with caching, templates, and memory systems

Follow the best practices for keyword tuning, test thoroughly, and monitor confidence scores to maintain high accuracy.

**For detailed testing workflows, see**: `TEMP_ENDPOINTS.md`

