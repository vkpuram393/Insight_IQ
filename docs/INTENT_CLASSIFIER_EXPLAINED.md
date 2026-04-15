# Intent Classifier Explained

## 🎯 What is Intent Classification?

Think of intent classification like **finding the most similar example** in a library of pre-labeled questions!

> **📊 Visual Flow Diagram**: See `images/INTENT_CLASSIFIER_FLOW.png` for a complete visual representation of the entire intent classification and routing flow.

---

## 🔄 Complete Flow Overview

The intent classification is part of a larger workflow. Here's the complete flow from user query to response:

```
User Query
    ↓
Orchestrator Node (Normalize Text)
    ↓
Safety Precheck (Mask PII)
    ↓
Cache Check (Check if response cached)
    ↓
Intent Agent Node (extended_intent_agent_node)
    ├─ Generate Query Embedding
    ├─ Load 600 Pre-computed Embeddings (PKL cache)
    ├─ Cosine Similarity Calculation
    ├─ Match to 30 Intent Categories
    ├─ Confidence Score
    ├─ Regex Entity Extraction
    ├─ Complexity Detection
    └─ Set State Flags
    ↓
Confidence Checker Node (Process state & log to DB)
    ↓
Confidence Check Router (Route based on confidence, is_complex, needs_clarification)
    ├─→ LLM Judge (if is_complex=True OR confidence < 0.7)
    ├─→ Clarification (if needs_clarification=True OR missing entities)
    ├─→ Build Context (if high confidence + has entities)
    └─→ Direct Response (if greeting/help/out_of_scope)
    ↓
[Routing Paths Continue...]
    ├─→ Build Context → Call Claims API → Response Agent
    ├─→ LLM Judge → Response Agent
    └─→ Clarification → Response Agent
    ↓
Safety Postcheck (Unmask PII tokens)
    ↓
Update Memory (Store conversation)
    ↓
Cache Response
    ↓
Return Response to User
```

**Key Points:**
- Intent classification happens early in the flow (after safety precheck and cache check)
- The result influences routing decisions (LLM Judge, Clarification, or Build Context)
- Multiple paths can lead to the Response Agent, which generates the final answer

---

When a user types:
- "What's the status of my claim #12345?"
- "Why was my claim rejected?"
- "Find me a pharmacy near me"

The intent classifier figures out **what they want** by comparing their question to 600 pre-labeled examples:
- `claim_status` - They want to check claim status
- `rejection_reasons` - They want to know why claim was rejected
- `pharmacy_info` - They want to find a pharmacy

It's like a **semantic search engine** that understands meaning, not just keywords.

---

## 🤖 How It Works: The Big Picture

```
User Input: "What's my claim status?"
    ↓
Convert to Embedding (vector of numbers representing meaning)
    ↓
Compare with 600 Pre-computed Examples (using cosine similarity)
    ↓
Find Most Similar Intent
    ↓
Output: {
    "intent": "claim_status",
    "confidence": 0.88,
    "entities": {"claim_number": "12345"}
}
```

**In simple terms:**
1. User sends a message
2. We convert it to an embedding (a vector that captures meaning)
3. We compare it to 600 pre-computed example embeddings (20 per intent, 30 intents)
4. We find the most similar intent using cosine similarity
5. We return the intent with highest similarity score

**Key difference from LLM-based approach:**
- **No LLM calls** - Fast (~200ms vs 500ms+)
- **No training required** - Uses semantic similarity (zero-shot)
- **Cost-effective** - Only embedding API calls (cheaper than LLM)
- **30 CVS-specific intents** - Production-ready for pharmacy benefits domain

---

## 📝 Step-by-Step: How the Code Works

### Step 0: Pre-Intent Classification (Orchestrator & Safety)

Before intent classification, the query goes through:

**Orchestrator Node:**
- Normalizes the user's text
- Prepares the state for processing

**Safety Precheck:**
- Masks PII/PHI (Personally Identifiable Information)
- Checks for safety violations
- If blocked → Returns error and ENDs flow

**Cache Check:**
- Checks if a cached response exists for this query
- If cache hit → Returns cached response (skips intent classification)
- If cache miss → Continues to intent classification

---

### Step 1: The Function Gets Called

The intent classification happens in one of two ways:

**Option A: Via Intent Agent Node (LLM-based - currently in workflow)**
```python
# agents/intent_agent.py - Line 74
async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    """Classify user intent using LLM (Gemini)"""
```

**Option B: Via Extended Intent Agent Node (Embedding-based - recommended)**
```python
# agents/extended_intent_agent_node.py - Line 30
async def extended_intent_agent_node(state: AgentState) -> Dict[str, Any]:
    """Classify user intent using embedding classifier (NO LLM)"""
```

**What happens:**
- LangGraph automatically calls this function
- It passes the full `AgentState` (which contains the user's message)
- The function must return a dictionary with `intent`, `confidence`, and `entities`

**Think of it like:** A function that receives a question and returns an answer about what the question means.

---

### Step 2: Extract Input from State

```python
# agents/extended_intent_agent_node.py - Line 37

text = state["text"]  # User's message: "What's my claim status?"
```

**What this does:**
- Gets the user's current message from `state["text"]`
- Unlike LLM-based approach, we don't need conversation history for embeddings
- The embedding captures semantic meaning, which is context-independent

**Example:**
```python
# Input state:
state = {
    "text": "What's my claim status?",
    "session_id": "session_123"
}

# After extraction:
text = "What's my claim status?"
```

---

### Step 3: Call the Classifier Wrapper

```python
# agents/extended_intent_agent_node.py - Line 40

# Classify intent using wrapper (respects settings.use_cvs_intent_classifier)
intent_result = classify_intent_unified(text)
```

**What this does:**
- Calls the unified classifier wrapper
- The wrapper checks configuration and routes to the right classifier:
  - If `use_cvs_intent_classifier=True` and `use_embedding_classifier=True` → Uses `CVSIntentEmbedded` (embedding-based)
  - If `use_cvs_intent_classifier=True` and `use_embedding_classifier=False` → Uses keyword classifier
  - If `use_cvs_intent_classifier=False` → Uses original EDGAR classifier

**Configuration (config/config.py):**
```python
use_cvs_intent_classifier: bool = True  # Use CVS classifier (30+ intents)
use_embedding_classifier: bool = True   # Use embedding-based (semantic) vs keyword-based
```

---

### Step 4: Initialize the Embedded Classifier

```python
# classifiers/embedded_classifier.py - Lines 60-83

class CVSIntentEmbedded:
    def __init__(self):
        """Initialize with embedded intent examples"""
        
        # Cache the embeddings service (singleton)
        if EMBEDDINGS_AVAILABLE:
            self.embeddings_service = get_embeddings_service()
            # Uses Azure OpenAI or Google Cloud Vertex AI based on config
        else:
            self.embeddings_service = None
        
        # Build intent examples (600 examples embedded in this file)
        self.intent_examples = self._build_intent_examples()
        
        # Pre-compute embeddings for all examples (validates cache dimensions)
        self.intent_embeddings = self._embed_all_examples()
```

**What this does:**
1. **Selects embedding provider** based on `settings.use_google_embeddings`:
   - `True` → Google Cloud Vertex AI (`text-embedding-005`, 768 dimensions)
   - `False` → Azure OpenAI (`text-embedding-ada-002`, 1536 dimensions)

2. **Loads 600 intent examples** from `CVS_INTENT_EXAMPLES`:
   - 30 intents (e.g., `claim_status`, `rejection_reasons`, `pricing_info`)
   - 20 examples per intent = 600 total examples
   - Examples are LLM-generated from real CVS queries (no data leakage)

3. **Pre-computes embeddings** for all 600 examples:
   - Tries to load from cache first (`intent_embeddings_cache.pkl`)
   - If cache exists and dimensions match → Instant load (no API calls!)
   - If cache missing or dimension mismatch → Generates embeddings via API
   - Saves to cache for next time

**Why cache?**
- Generating 600 embeddings takes time and costs money
- Cache makes initialization instant on subsequent runs
- Automatically regenerates if embedding provider changes (Azure → Google or vice versa)

---

### Step 5: Convert User Query to Embedding

```python
# classifiers/embedded_classifier.py - Lines 268-277

# Get query embedding - FAIL if unavailable (no mock fallback)
if EMBEDDINGS_AVAILABLE and self.embeddings_service is not None:
    try:
        query_embedding = get_embedding(query)
    except Exception as e:
        logger.error(f"❌ Failed to get query embedding: {e}")
        raise RuntimeError("Query embedding failed - routing to LLM fallback") from e
```

**What this does:**
- Converts the user's query to an embedding vector
- Uses the configured embedding service (Azure or Google)
- If embedding fails → Raises `RuntimeError` to route to LLM fallback

**What is an embedding?**
- A vector (list of numbers) that represents the **semantic meaning** of text
- Similar meanings → Similar vectors
- Example:
  - "What's my claim status?" → `[0.1, 0.3, -0.2, 0.5, ...]` (768 or 1536 numbers)
  - "Check claim status" → `[0.12, 0.28, -0.18, 0.52, ...]` (very similar!)
  - "Hello" → `[-0.5, 0.1, 0.8, -0.3, ...]` (very different!)

---

### Step 6: Calculate Similarity with All Intent Examples

```python
# classifiers/embedded_classifier.py - Lines 279-313

# Calculate similarity scores for all intents
intent_scores = {}

for intent, example_embeddings in self.intent_embeddings.items():
    # Calculate cosine similarity with all examples of this intent
    similarities = self.embeddings_service.batch_similarity(
        query_list, example_list
    )
    
    # Use the highest similarity as the score for this intent
    intent_scores[intent] = float(max(similarities))
```

**What this does:**
1. **For each intent** (30 intents total):
   - Gets all 20 example embeddings for that intent
   - Calculates cosine similarity between user query embedding and each example
   - Takes the **maximum similarity** as the score for that intent

2. **Cosine similarity explained:**
   - Measures how similar two vectors are
   - Range: -1.0 (opposite) to 1.0 (identical)
   - Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`
   - Higher score = more similar meaning

**Example:**
```python
# User query: "What's my claim status?"
# Intent: claim_status
# Example embeddings: [20 examples]

similarities = [0.88, 0.85, 0.91, 0.79, ...]  # Similarity with each example
intent_scores["claim_status"] = 0.91  # Highest similarity

# Intent: rejection_reasons
similarities = [0.45, 0.52, 0.38, ...]  # Much lower
intent_scores["rejection_reasons"] = 0.52
```

---

### Step 7: Find Top Intent

```python
# classifiers/embedded_classifier.py - Lines 315-336

# Get top intent
if not intent_scores:
    return {
        'intent': 'out_of_scope',
        'confidence': 0.0,
        ...
    }

top_intent = max(intent_scores, key=intent_scores.get)
top_score = intent_scores[top_intent]

# Check if below threshold
if top_score < self.confidence_threshold:  # 0.50
    return {
        'intent': 'out_of_scope',
        'confidence': top_score,
        ...
    }
```

**What this does:**
1. Finds the intent with the highest similarity score
2. Checks if score is above threshold (0.50)
3. If below threshold → Returns `out_of_scope` (query doesn't match any intent)

**Example:**
```python
intent_scores = {
    "claim_status": 0.91,
    "rejection_reasons": 0.52,
    "pricing_info": 0.45,
    ...
}

top_intent = "claim_status"  # Highest score
top_score = 0.91  # Above threshold (0.50) ✅
```

---

### Step 8: Extract Entities

```python
# agents/extended_intent_agent_node.py - Lines 73-74

# Extract entities
entity_result = extract_entities_unified(text, intent=intent)
entities = entity_result['entities']
```

**What this does:**
- Uses the entity extractor to find specific information in the query
- Extracts: claim IDs, member IDs, prescription IDs, dates, amounts, etc.
- Validates required entities for the intent (e.g., `claim_status` requires `claim_id`)

**Example:**
```python
text = "What's the status of claim #12345?"
intent = "claim_status"

entities = {
    "claim_id": "12345"
}
```

---

### Step 9: Return the Result

```python
# agents/extended_intent_agent_node.py - Lines 125-149

result = {
    "intent": intent,
    "confidence": confidence,
    "entities": entities,
    "is_complex": is_complex,
    "needs_clarification": needs_clarification,
    "metadata": {
        "all_scores": intent_result.get('all_scores', {}),  # All intent similarities
        "classifier_type": "embedding",
        "intent_classification_metadata": {
            "top_intent": intent,
            "top_confidence": confidence,
            "num_intents_evaluated": len(intent_result.get('all_scores', {})),
        }
    }
}
```

**What this does:**
- Returns structured result with intent, confidence, entities
- Includes metadata for observability (all scores, classifier type)
- LangGraph automatically merges this into AgentState

---

### Step 10: Confidence Checker Node

After intent classification, the flow continues to the Confidence Checker:

```python
# nodes/confidence.py

async def confidence_checker_node(state: AgentState) -> Dict[str, Any]:
    """Process state and log to database"""
    # Logs the intent classification result
    # Prepares state for routing
    # Returns state for router
```

**What this does:**
- Processes the intent classification result
- Logs to database for telemetry
- Prepares state for the router

---

### Step 11: Confidence Check Router

The router decides the next step based on:
- `confidence`: Similarity score (0.0 to 1.0)
- `is_complex`: Whether query contains aggregations/comparisons
- `needs_clarification`: Whether required entities are missing
- `intent_reclassified`: Whether LLM Judge already ran
- `entities`: Extracted entities (claim_id, member_id, etc.)

**Routing Rules:**

```python
# nodes/confidence.py - confidence_check_router()

IF intent_reclassified == False (initial classifier result):
    1. If is_complex=True → llm_judge (needs expert review)
    2. If confidence < 0.7 → llm_judge (needs expert review)
    3. If greeting/help/out_of_scope → response_safety_pii_precheck (direct to LLM)
    4. If high confidence + has entities → build_context (proceed to API)

IF intent_reclassified == True (LLM judge already ran):
    1. If confidence >= 0.7 AND entities present → build_context
    2. If missing entities OR confidence < 0.7 → clarification
```

---

## 🎯 How Confidence Scores and Entities Influence Routing

This section explains in detail how **confidence scores** and **entities** work together to determine the flow path.

### Understanding Confidence Scores

**What is a confidence score?**
- A number between **0.0 and 1.0** representing how similar the user's query is to the best-matching intent
- Calculated using **cosine similarity** between the query embedding and intent example embeddings
- Higher score = More confident the intent is correct

**Confidence Threshold:**
- Default threshold: **0.7** (configurable in `domain_config.json`)
- Below 0.7 = Low confidence → Needs review (LLM Judge or Clarification)
- Above 0.7 = High confidence → Can proceed (if entities present)

**How confidence is calculated:**
```python
# Step 1: Compare query embedding to all 600 examples
intent_scores = {
    "claim_status": 0.91,      # Highest similarity
    "rejection_reasons": 0.52,
    "pricing_info": 0.45,
    ...
}

# Step 2: Take the highest score as confidence
confidence = max(intent_scores.values())  # 0.91
top_intent = max(intent_scores, key=intent_scores.get)  # "claim_status"
```

---

### Understanding Entity Extraction

**What are entities?**
- Structured information extracted from the user's query
- Required for API calls (e.g., `claim_id` needed to check claim status)
- Extracted using **regex patterns** and **validation rules**

**Common entities:**
- `claim_id`: Claim identifier (e.g., "CLM12345" or "253152732536005")
- `member_id`: Member identifier (e.g., "MEM123")
- `claim_sequence`: Sequence number (e.g., "001", "002")
- `prescription_id`: Prescription identifier (e.g., "RX123")
- `date_range`: Date ranges (e.g., "January to March")
- `amounts`: Monetary amounts (e.g., "$45.20")

**How entities are extracted:**
```python
# utils/entity_extractor.py

patterns = {
    'claim_id': r'\b(CLM\d{3,10}|\d{15})\b',  # CLM12345 or 15-digit number
    'member_id': r'\b(MEM\d{3,4})\b',         # MEM123
    'claim_sequence': r'\b(\d{3})\b',         # 001, 002, etc.
    # ... more patterns
}

# Extract from query
query = "What's the status of claim #12345?"
entities = {
    "claim_id": "12345"  # Extracted via regex
}
```

**Entity validation:**
- Each intent has **required entities** (defined in `config/api_routing_config.py`)
- Example: `claim_status` requires `claim_id`
- If required entity missing → `needs_clarification=True`

---

### How Confidence + Entities Determine Routing

The router uses **both** confidence and entities to make routing decisions. Here's how they work together:

#### Scenario 1: High Confidence + Has Entities ✅

**Example:**
```python
query = "What's the status of claim #12345?"
confidence = 0.91  # High (≥ 0.7)
entities = {"claim_id": "12345"}  # Required entity present
is_complex = False
```

**Routing Decision:**
```
✅ High confidence (0.91 ≥ 0.7)
✅ Has required entities (claim_id present)
→ Route to: build_context → Call Claims API
```

**What happens:**
1. Build Context gathers conversation history
2. Calls Claims API with `claim_id: "12345"`
3. Response Agent generates answer from API data
4. Returns response to user

---

#### Scenario 2: High Confidence + Missing Entities ⚠️

**Example:**
```python
query = "What's my claim status?"
confidence = 0.88  # High (≥ 0.7)
entities = {}  # Missing claim_id!
is_complex = False
```

**Routing Decision:**
```
✅ High confidence (0.88 ≥ 0.7)
❌ Missing required entities (claim_id missing)
→ Route to: clarification
```

**What happens:**
1. Clarification Node prepares question context
2. Response Agent generates: "I need your claim ID to look that up. Could you provide it?"
3. User provides claim_id in next message
4. Flow restarts with entities now present

---

#### Scenario 3: Low Confidence (Regardless of Entities) ⚠️

**Example:**
```python
query = "Help me with something"
confidence = 0.45  # Low (< 0.7)
entities = {}
is_complex = False
intent_reclassified = False  # LLM Judge hasn't run yet
```

**Routing Decision:**
```
❌ Low confidence (0.45 < 0.7)
→ Route to: llm_judge (if intent_reclassified=False)
```

**What happens:**
1. LLM Judge re-classifies using Gemini LLM
2. Updates confidence (typically to 0.95 if confident)
3. Returns to Confidence Checker for re-evaluation
4. Router evaluates again with new confidence

**After LLM Judge:**
```python
confidence = 0.95  # Updated by LLM Judge
intent_reclassified = True  # Prevents infinite loop
entities = {}

# Router evaluates again:
# ✅ High confidence (0.95 ≥ 0.7)
# ❌ Missing entities
# → Route to: clarification
```

---

#### Scenario 4: Complex Query (Regardless of Confidence) 🧠

**Example:**
```python
query = "Show me all claims from January to March that contributed to my deductible"
confidence = 0.78  # High (≥ 0.7)
entities = {}
is_complex = True  # Contains "all", "from...to", aggregation keywords
intent_reclassified = False
```

**Routing Decision:**
```
✅ High confidence (0.78 ≥ 0.7)
🧠 Complex query (is_complex=True)
→ Route to: llm_judge (for expert review)
```

**What happens:**
1. LLM Judge analyzes complex query
2. May update intent, confidence, or extract additional entities
3. Returns to Confidence Checker
4. Router evaluates again (may route to Build Context if confidence improved)

---

#### Scenario 5: Low Confidence + Missing Entities (After LLM Judge) ❌

**Example:**
```python
query = "Tell me about my claim"
confidence = 0.35  # Low (< 0.7) - even after LLM Judge
entities = {}  # Missing claim_id
intent_reclassified = True  # LLM Judge already ran
```

**Routing Decision:**
```
❌ Low confidence (0.35 < 0.7) - even after LLM Judge
❌ Missing entities
intent_reclassified = True (won't route to LLM Judge again)
→ Route to: clarification
```

**What happens:**
1. Clarification Node prepares question
2. Response Agent generates: "I need your claim ID to look that up. Could you provide it?"
3. User provides claim_id
4. Flow restarts with entities now present

---

### Entity Checking in Conversation History

**Important:** The router checks **both current message AND conversation history** for entities!

**Example:**
```python
# Message 1:
query = "What's my claim status?"
entities = {}  # No claim_id in current message
confidence = 0.88

# But conversation history contains:
history = [
    {"role": "user", "content": "My claim ID is 12345"},
    {"role": "assistant", "content": "Got it!"}
]

# Router checks history:
has_entities_in_history = True  # Found claim_id in history!
has_entities_anywhere = True  # Either current OR history

# Routing Decision:
✅ High confidence (0.88 ≥ 0.7)
✅ Has entities anywhere (found in history)
→ Route to: build_context
```

**Why this matters:**
- Users don't need to repeat information
- Conversational continuity is maintained
- Entities from previous messages are reused

---

### Summary: Confidence + Entity Matrix

| Confidence | Has Entities | is_complex | intent_reclassified | Route |
|------------|--------------|------------|---------------------|-------|
| ≥ 0.7 | ✅ Yes | ❌ No | ❌ No | `build_context` |
| ≥ 0.7 | ✅ Yes (in history) | ❌ No | ❌ No | `build_context` |
| ≥ 0.7 | ❌ No | ❌ No | ❌ No | `clarification` |
| < 0.7 | ✅ Yes | ❌ No | ❌ No | `llm_judge` |
| < 0.7 | ❌ No | ❌ No | ❌ No | `llm_judge` |
| Any | Any | ✅ Yes | ❌ No | `llm_judge` |
| ≥ 0.7 | ✅ Yes | ❌ No | ✅ Yes | `build_context` |
| ≥ 0.7 | ❌ No | ❌ No | ✅ Yes | `clarification` |
| < 0.7 | Any | ❌ No | ✅ Yes | `clarification` |
| Any | Any | Any | Any | `response_safety_pii_precheck` (if greeting/help/out_of_scope) |

**Key Points:**
1. **High confidence + entities** → Proceed to API (`build_context`)
2. **High confidence + no entities** → Ask for missing info (`clarification`)
3. **Low confidence** → Get expert review (`llm_judge`) - unless already ran
4. **Complex query** → Always get expert review (`llm_judge`)
5. **After LLM Judge** → Re-evaluate with new confidence/entities
6. **History matters** → Entities can come from previous messages

**Routing Paths:**

1. **LLM Judge** (`llm_judge`):
   - Triggered when: `is_complex=True` OR `confidence < 0.7`
   - Purpose: Re-classify intent using LLM (Gemini) for better accuracy
   - Uses LLM to analyze the query more deeply
   - Updates confidence and potentially intent/entities
   - Sets `intent_reclassified=True` (prevents infinite loops)
   - Returns to confidence_checker for re-evaluation

2. **Clarification** (`clarification`):
   - Triggered when: `needs_clarification=True` OR missing required entities
   - Purpose: Ask user for missing information (claim_id, member_id, etc.)
   - Prepares clarification context
   - Routes to Response Agent to generate follow-up question

3. **Build Context** (`build_context`):
   - Triggered when: High confidence (≥0.7) + has entities
   - Purpose: Gather conversation history, facts, and slots
   - Prepares context for API call
   - Routes to Call Claims API

4. **Direct Response** (`response_safety_pii_precheck`):
   - Triggered when: Intent is `greeting`, `help`, or `out_of_scope`
   - Purpose: Skip API call, go directly to Response Agent
   - No API needed for these intents

---

### Step 12: Post-Intent Classification Flow

After routing, the flow continues:

**If routed to Build Context:**
```
Build Context Node
    ↓
Call Claims API (with entities)
    ↓
Response Agent (Generate natural response from API data)
    ↓
Safety Postcheck (Unmask PII)
    ↓
Update Memory
    ↓
Cache Response
    ↓
Return to User
```

**If routed to LLM Judge:**
```
LLM Judge Node (Re-classify with LLM)
    ↓
Confidence Checker (Re-evaluate)
    ↓
Router (Routes again based on new confidence)
    ↓
[Continues to Build Context, Clarification, or Direct Response]
```

**If routed to Clarification:**
```
Clarification Node (Prepare question context)
    ↓
Response Agent (Generate follow-up question)
    ↓
Safety Postcheck
    ↓
Update Memory
    ↓
Return to User (asks for missing info)
```

---

## 🎬 Complete Example: End-to-End

Let's trace through a real example:

### Input:
```python
state = {
    "text": "What's the status of claim #12345?",
    "session_id": "session_123"
}
```

### Step-by-Step Execution:

**1. Function called:**
```python
result = await extended_intent_agent_node(state)
```

**2. Extract input:**
```python
text = "What's the status of claim #12345?"
```

**3. Call classifier wrapper:**
```python
intent_result = classify_intent_unified(text)
# Routes to CVSIntentEmbedded (use_embedding_classifier=True)
```

**4. Initialize classifier (first time):**
```python
classifier = CVSIntentEmbedded()
# Loads 600 examples, generates embeddings (or loads from cache)
# intent_embeddings = {
#     "claim_status": [20 embeddings],
#     "rejection_reasons": [20 embeddings],
#     ...
# }
```

**5. Convert query to embedding:**
```python
query_embedding = get_embedding("What's the status of claim #12345?")
# Returns: [0.1, 0.3, -0.2, 0.5, ...] (768 or 1536 numbers)
```

**6. Calculate similarities:**
```python
intent_scores = {
    "claim_status": 0.91,      # Highest! (very similar to examples)
    "rejection_reasons": 0.52,
    "pricing_info": 0.45,
    ...
}
```

**7. Find top intent:**
```python
top_intent = "claim_status"
top_score = 0.91  # Above threshold (0.50) ✅
```

**8. Extract entities:**
```python
entities = {"claim_id": "12345"}
```

**9. Return:**
```python
return {
    "intent": "claim_status",
    "confidence": 0.91,
    "entities": {"claim_id": "12345"},
    "is_complex": False,
    "needs_clarification": False,
    ...
}
```

**10. LangGraph merges:**
```python
# Updated state:
state = {
    "text": "What's the status of claim #12345?",
    "session_id": "session_123",
    "intent": "claim_status",      # ← Added
    "confidence": 0.91,             # ← Added
    "entities": {"claim_id": "12345"},  # ← Added
    "is_complex": False,            # ← Added
    "needs_clarification": False   # ← Added
}
```

**11. Confidence Checker processes:**
```python
# Confidence Checker Node
# Logs to database
# Prepares for routing
```

**12. Router evaluates:**
```python
# confidence = 0.91 (≥ 0.7 threshold) ✅
# has_entities = True (claim_id present) ✅
# is_complex = False
# needs_clarification = False

# Route: build_context (high confidence + has entities)
```

**13. Build Context gathers information:**
```python
# Gets conversation history
# Extracts entities from history
# Prepares context for API call
```

**14. Call Claims API:**
```python
# POST /api/claims/status
# Body: {"claim_id": "12345"}
# Returns: Claim status data
```

**15. Response Agent generates answer:**
```python
# Uses Gemini LLM
# Generates natural language response from API data
# Response: "Your claim #12345 is currently approved and processed."
```

**16. Safety Postcheck:**
```python
# Unmasks PII tokens
# Checks for data leakage
# Returns response
```

**17. Update Memory & Cache:**
```python
# Stores conversation in memory
# Caches response for future queries
# Returns to user
```

---

## 🔀 Routing and Post-Classification Flow

### LLM Judge Node

**Purpose:** Re-classify intent using LLM when initial classifier has low confidence or detects complexity.

**When it runs:**
- `is_complex=True` (query contains aggregations, comparisons, date ranges)
- `confidence < 0.7` (initial classifier uncertain)
- `intent_reclassified=False` (hasn't run yet - prevents infinite loops)

**What it does:**
```python
# nodes/llm_judge.py

async def llm_judge_node(state: AgentState) -> Dict[str, Any]:
    # Takes original intent classification (low confidence)
    # Uses Gemini LLM to re-classify with more context
    # Updates confidence (typically to 0.95 if mock_high_conf=True)
    # Sets intent_reclassified=True (prevents infinite loop)
    # Returns to confidence_checker for re-evaluation
```

**Key Features:**
- Preserves original embedding classifier confidence in metadata
- Stores LLM judge confidence separately
- Prevents infinite loops with `intent_reclassified` flag
- Routes back to confidence_checker for re-evaluation

---

### Clarification Node

**Purpose:** Ask user for missing information when required entities are not present.

**When it runs:**
- `needs_clarification=True` (missing required entities)
- After LLM Judge if still missing entities or low confidence

**What it does:**
```python
# nodes/clarification.py

async def clarification_node(state: AgentState) -> Dict[str, Any]:
    # Determines WHY clarification is needed
    # Prepares clarification_context
    # Sets needs_clarification=True
    # Routes to Response Agent to generate follow-up question
```

**Example:**
```python
# User: "What's my claim status?"
# Missing: claim_id
# Clarification: "I need your claim ID to look that up. Could you provide it?"
```

---

### Build Context Node

**Purpose:** Gather comprehensive context for API calls.

**When it runs:**
- High confidence (≥0.7) + has entities
- After LLM Judge if confidence improved and entities present

**What it does:**
```python
# nodes/context.py

async def build_context_node(state: AgentState) -> Dict[str, Any]:
    # Gets conversation history (last N messages)
    # Extracts entities from history (if not in current message)
    # Gets relevant facts from memory store
    # Builds comprehensive context object
    # Prepares for API call
```

**Context includes:**
- Recent conversation history
- Extracted entities (from current message + history)
- Intent and confidence
- User profile information
- Domain context

---

## 🧠 ML Concepts Explained

### 1. **Embeddings (Vector Representations)**

**What they are:**
- Numerical vectors that capture semantic meaning of text
- Similar meanings → Similar vectors (close in vector space)
- Generated by neural networks trained on massive text corpora

**How they work:**
- Text → Neural Network → Vector of numbers
- Example: "What's my claim status?" → `[0.1, 0.3, -0.2, 0.5, ...]`
- The numbers encode meaning, not just words

**In our code:**
```python
# Azure OpenAI: text-embedding-ada-002 (1536 dimensions)
# Google Cloud: text-embedding-005 (768 dimensions)
query_embedding = get_embedding("What's my claim status?")
```

**Why embeddings?**
- Understands synonyms: "status" and "progress" are similar
- Handles variations: "What's my claim status?" ≈ "Check claim status"
- Language-agnostic: Works across different phrasings

---

### 2. **Cosine Similarity**

**What it is:**
- Measures how similar two vectors are
- Range: -1.0 (opposite) to 1.0 (identical)
- 0.0 = orthogonal (unrelated)

**Formula:**
```
cos(θ) = (A · B) / (||A|| × ||B||)
```

**In our code:**
```python
similarities = self.embeddings_service.batch_similarity(
    query_embedding, example_embeddings
)
```

**Why cosine similarity?**
- Normalized (not affected by vector magnitude)
- Focuses on direction (semantic meaning), not magnitude
- Fast to compute

**Example:**
- Query: "What's my claim status?"
- Example 1: "Show the current status of this claim" → Similarity: 0.91 ✅
- Example 2: "Find a pharmacy near me" → Similarity: 0.15 ❌

---

### 3. **Zero-Shot Learning**

**What it is:**
- Classifying examples without training on labeled data
- Uses pre-computed embeddings of example queries
- No model training required!

**In our code:**
- 600 pre-labeled examples (20 per intent)
- Compare new query to examples → Find most similar → Return that intent
- No training step needed!

**Why zero-shot?**
- Fast to set up (just need examples)
- Easy to add new intents (add 20 examples)
- No retraining required

---

### 4. **Embedding Providers**

**Azure OpenAI:**
- Model: `text-embedding-ada-002`
- Dimensions: 1536
- Authentication: API Key or Azure AD (Service Principal)
- Cost: Pay per token

**Google Cloud Vertex AI:**
- Model: `text-embedding-005`
- Dimensions: 768
- Authentication: Application Default Credentials (same as Gemini LLM)
- Cost: Pay per request

**Configuration:**
```python
# config/config.py
use_google_embeddings: bool = False  # True = Google, False = Azure
```

**Dimension Mismatch Handling:**
- Cache automatically validates dimensions
- If mismatch detected → Deletes cache and regenerates
- Prevents errors when switching providers

---

### 5. **Confidence Scores**

**What it is:**
- A number (0.0 to 1.0) indicating similarity to the best-matching intent
- 1.0 = Perfect match, 0.0 = No match

**In our code:**
```python
confidence = top_score  # Highest cosine similarity
```

**Threshold:**
```python
confidence_threshold = 0.50  # Below this → out_of_scope
```

**Why it matters:**
- High confidence (0.8+): Very clear intent
- Medium confidence (0.5-0.8): Somewhat clear
- Low confidence (<0.5): Unclear → Returns `out_of_scope`

**Example:**
- "What's my claim status?" → confidence: 0.91 (very clear)
- "Help me" → confidence: 0.35 (unclear → out_of_scope)

---

## 🔍 Code Deep Dive: Key Sections

### The 600 Intent Examples

```python
# classifiers/embedded_classifier.py - Lines 439-1124

CVS_INTENT_EXAMPLES = {
    "claim_status": [
        "Generate a full claim summary for this claim.",
        "Show the current status of this claim.",
        "Display all details for this claim.",
        # ... 17 more examples
    ],
    "rejection_reasons": [
        "Generate the specific rejection reasons for this claim.",
        "Show the edits that caused this claim to reject.",
        # ... 18 more examples
    ],
    # ... 28 more intents
}
```

**What this is:**
- 600 LLM-generated examples from real CVS queries
- 30 intents × 20 examples each
- Examples are embedded in the code (no external file)
- Test queries excluded to avoid data leakage

**Why 20 examples per intent?**
- More examples = better coverage of variations
- Balances accuracy vs. computation cost
- Can increase if needed for better accuracy

---

### Embedding Cache

```python
# classifiers/embedded_classifier.py - Lines 161-241

def _embed_all_examples(self) -> Dict[str, np.ndarray]:
    cache_file = "classifiers/intent_embeddings_cache.pkl"
    
    if os.path.exists(cache_file):
        # Load from cache (INSTANT - no API calls!)
        intent_embeddings = pickle.load(f)
        
        # Validate dimensions match current provider
        if self._validate_cache_dimensions(intent_embeddings):
            return intent_embeddings  # ✅ Cache hit!
        else:
            # Dimension mismatch → Delete cache and regenerate
            os.remove(cache_file)
    
    # Generate embeddings (fallback)
    for intent, examples in self.intent_examples.items():
        embeddings = embeddings_service.embed(examples)  # Batch API call
        intent_embeddings[intent] = np.array(embeddings)
    
    # Save to cache
    pickle.dump(intent_embeddings, f)
```

**What this does:**
- **First run**: Generates 600 embeddings via API (takes time, costs money)
- **Subsequent runs**: Loads from cache (instant!)
- **Provider switch**: Detects dimension mismatch, regenerates automatically

**Cache benefits:**
- Instant initialization (no API calls)
- Cost savings (no repeated API calls)
- Automatic regeneration on provider change

---

### Dimension Validation

```python
# classifiers/embedded_classifier.py - Lines 123-159

def _validate_cache_dimensions(self, intent_embeddings: Dict[str, np.ndarray]) -> bool:
    expected_dim = self._get_expected_embedding_dimension()  # 768 or 1536
    actual_dim = first_intent.shape[-1]  # From cache
    
    if actual_dim != expected_dim:
        logger.warning(f"⚠️  Cache dimension mismatch: cached={actual_dim}, expected={expected_dim}")
        return False  # Cache invalid!
    
    return True  # Cache valid ✅
```

**What this does:**
- Checks if cached embeddings match current provider dimensions
- Azure: 1536 dimensions
- Google: 768 dimensions
- If mismatch → Cache is invalid, must regenerate

**Why it's important:**
- Prevents errors when switching providers
- Ensures embeddings are compatible with current provider

---

### Error Handling and Fallback

```python
# classifiers/intent_classifier_wrapper.py - Lines 50-63

try:
    classifier = CVSIntentEmbedded()
    result = classifier.classify(query)
except RuntimeError as e:
    # Embeddings unavailable - return special flag to route to LLM
    logger.error(f"❌ Embedding classifier failed: {e}")
    logger.info("🔄 Routing query directly to Response LLM Agent")
    return {
        'intent': 'embedding_failed',
        'confidence': 0.0,
        'embedding_failed': True,  # Special flag for router
        'fallback_reason': str(e)
    }
```

**What this does:**
- If embedding generation fails → Routes to LLM fallback
- Prevents system crash
- User still gets a response (via LLM)

**Why it's important:**
- Embedding APIs can fail (network, auth, rate limits)
- Better to fallback than crash
- LLM can still classify intent (slower, more expensive)

---

## 📊 Real-World Examples

### Example 1: Clear Intent (High Confidence)

**Input:**
```
"What's the status of my claim #12345?"
```

**Processing:**
1. Query embedding: `[0.1, 0.3, -0.2, ...]`
2. Compare to 600 examples
3. Highest similarity: `claim_status` examples (0.91)
4. Extract entity: `claim_id = "12345"`

**Output:**
```json
{
    "intent": "claim_status",
    "confidence": 0.91,
    "entities": {"claim_id": "12345"},
    "is_complex": false,
    "needs_clarification": false
}
```

---

### Example 2: Unclear Intent (Low Confidence)

**Input:**
```
"Help me"
```

**Processing:**
1. Query embedding: `[-0.2, 0.1, 0.4, ...]`
2. Compare to 600 examples
3. Highest similarity: `greeting` examples (0.35)
4. Below threshold (0.50) → `out_of_scope`

**Output:**
```json
{
    "intent": "out_of_scope",
    "confidence": 0.35,
    "entities": {},
    "is_complex": false,
    "needs_clarification": false
}
```

**What happens next:**
- Confidence is low (< 0.50 threshold)
- System routes to clarification or LLM fallback

---

### Example 3: Complex Query

**Input:**
```
"Show me all claims from January to March that contributed to my deductible"
```

**Processing:**
1. Query embedding: `[0.2, -0.1, 0.5, ...]`
2. Compare to 600 examples
3. Highest similarity: `date_range_claims` examples (0.78)
4. Detects complexity: Contains "all", "from...to", aggregation keywords

**Output:**
```json
{
    "intent": "date_range_claims",
    "confidence": 0.78,
    "entities": {},
    "is_complex": true,  // ← Complexity detected!
    "needs_clarification": false
}
```

**What happens next:**
- `is_complex: true` → Routes to LLM for reasoning
- Embedding classifier identifies intent, LLM handles complex logic

---

## 🎓 Key Takeaways

1. **Intent classification = Semantic similarity search**
   - Takes natural language input
   - Compares to 600 pre-labeled examples
   - Returns most similar intent with confidence score

2. **Uses Embeddings (not LLM)**
   - Fast (~200ms vs 500ms+ for LLM)
   - Cost-effective (embedding API cheaper than LLM)
   - Zero-shot (no training required)

3. **600 Pre-computed Examples**
   - 30 intents × 20 examples each
   - LLM-generated from real CVS queries
   - Embedded in code (no external file)

4. **Embedding Providers**
   - Azure OpenAI: 1536 dimensions
   - Google Cloud: 768 dimensions
   - Automatic cache validation and regeneration

5. **Error Handling**
   - Embedding failure → Routes to LLM fallback
   - Dimension mismatch → Auto-regenerates cache
   - Always returns a result (never crashes)

6. **Production-Ready**
   - 30 CVS-specific intents
   - Handles complexity detection
   - Entity extraction included
   - Observability metadata

---

## 🔧 How to Improve It

### 1. **Add More Examples**
- Increase from 20 to 30+ examples per intent
- Better coverage of variations
- Higher accuracy

### 2. **Fine-tune Embeddings**
- Train custom embeddings on CVS domain data
- Better semantic understanding of pharmacy terminology
- More accurate similarity scores

### 3. **Hybrid Approach**
- Use embeddings for simple queries (fast, cheap)
- Use LLM for complex queries (accurate, expensive)
- Best of both worlds

### 4. **Add New Intents**
- Add 20 examples for new intent
- No retraining required
- Instant availability

---

## 📚 Summary

The intent classifier is like a **semantic search engine**:
- **Input**: Natural language ("What's my claim status?")
- **Process**: Convert to embedding → Compare to 600 examples → Find most similar
- **Output**: Structured data (`{"intent": "claim_status", "confidence": 0.91, ...}`)

It uses:
- **Embeddings** to capture semantic meaning
- **Cosine similarity** to find matches
- **Zero-shot learning** (no training required)
- **600 pre-computed examples** (30 intents × 20 examples)

### Complete Flow Summary

1. **Pre-Processing**: Orchestrator → Safety Precheck → Cache Check
2. **Intent Classification**: Generate embedding → Compare to 600 examples → Find best match
3. **Post-Processing**: Confidence Checker → Router → [LLM Judge / Clarification / Build Context]
4. **Response Generation**: API Call (if needed) → Response Agent → Safety Postcheck → Update Memory → Cache → Return

### Key Components

- **Embedding Classifier**: Fast, cost-effective semantic similarity (200ms)
- **LLM Judge**: Expert reviewer for complex/low-confidence queries
- **Confidence Router**: Smart routing based on confidence, complexity, and entity completeness
- **Clarification Engine**: Handles missing information gracefully
- **Build Context**: Gathers comprehensive context for API calls

The code is well-structured, handles errors gracefully, and works with both Azure OpenAI and Google Cloud Vertex AI embeddings.

**That's how intent classification works in this codebase!** 🎉

> **📊 For a visual representation of this complete flow, see `images/INTENT_CLASSIFIER_FLOW.png`**
