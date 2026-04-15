# Tool Call Agent Explained - Complete Guide

## 🎯 What is the Tool Call Agent?

Think of the **Tool Call Agent** like a **smart assistant that makes phone calls to other systems** for you!

When you ask: "What's the status of my claim #12345?"

The Tool Call Agent:
1. **Understands** what you need (from the Intent Agent)
2. **Finds** the right phone number (API endpoint) to call
3. **Prepares** the information to send (request body)
4. **Calls** the external system (makes HTTP request)
5. **Gets** the data back (API response)
6. **Packages** it nicely for the Response Agent to use

It's like having a personal assistant who knows exactly which department to call and what information to ask for!

---

## 🤖 How It Works: The Big Picture

```
User: "What's my claim status?"
    ↓
Intent Agent: "User wants claim_status, has claim_id: 12345"
    ↓
Tool Call Agent: "I need to call the Claims API"
    ↓
Tool Call Agent: Finds the right API endpoint
    ↓
Tool Call Agent: Prepares request: {"claimId": "12345"}
    ↓
Tool Call Agent: Calls external API (HTTP POST)
    ↓
External API: Returns claim data
    ↓
Tool Call Agent: Packages data in ToolResult format
    ↓
Response Agent: Uses this data to generate answer
```

**In simple terms:**
1. Intent Agent tells Tool Call Agent: "User wants claim status, here's the claim ID"
2. Tool Call Agent finds the right API to call (like looking up a phone number)
3. Tool Call Agent prepares the request (like writing down what to ask)
4. Tool Call Agent calls the API (like making a phone call)
5. API responds with claim data (like getting an answer)
6. Tool Call Agent packages it for the Response Agent (like writing down the answer)

---

## 📝 Step-by-Step: How the Code Works

### Step 0: Where Does This Fit in the Flow?

The Tool Call Agent runs **after** the Intent Agent and **before** the Response Agent:

```
Orchestrator → Safety Precheck → Cache Check → Intent Agent
    ↓
Confidence Checker → Router
    ↓
Build Context (gathers conversation history)
    ↓
Tool Call Agent ← YOU ARE HERE
    ↓
Response Agent (generates answer from API data)
    ↓
Safety Postcheck → Update Memory → Cache → Return to User
```

**Why it's needed:**
- Intent Agent knows **what** the user wants
- Tool Call Agent **gets the data** from external systems
- Response Agent **formats the answer** for the user

---

### Step 1: The Function Gets Called

```python
# tools/claims_api.py - Line 45

async def call_claims_tool_node(state: AgentState) -> Dict[str, Any]:
    """Primary orchestrator node called by agent."""
```

**What happens:**
- LangGraph automatically calls this function
- It passes the full `AgentState` (which contains intent, entities, etc.)
- The function must return `{"tool_results": {...}}` to update the state

**Think of it like:** A function that receives a shopping list (intent + entities) and returns the groceries (API data).

---

### Step 2: Extract Intent and Entities from State

```python
# tools/claims_api.py - Lines 78-100

# Handle both AgentState (dict) and IntentResult (Pydantic model)
if isinstance(state, dict):
    # Called from LangGraph with AgentState
    intent = state.get("intent")
    
    # CRITICAL: Merge extracted_slots (from conversation history) with current entities
    extracted_slots = state.get("extracted_slots", {})
    current_entities = state.get("entities", {})
    # Current entities take precedence over extracted ones
    entities = {**extracted_slots, **current_entities}
```

**What this does:**
- Gets the **intent** (what the user wants): e.g., `"claim_status"`
- Gets the **entities** (specific information): e.g., `{"claim_id": "12345"}`
- **Merges entities from conversation history** with current message entities
  - If user said "claim 12345" earlier, and now says "what's its status"
  - The Tool Call Agent remembers the claim_id from earlier!

**Example:**
```python
# State from Intent Agent:
state = {
    "intent": "claim_status",
    "entities": {"claim_id": "12345"},
    "extracted_slots": {"claim_id": "12345"}  # From previous message
}

# After extraction:
intent = "claim_status"
entities = {"claim_id": "12345"}  # Merged from both sources
```

**Why merge entities?**
- Users don't always repeat information
- "What's my claim status?" (after saying "My claim is 12345" earlier)
- Tool Call Agent remembers the claim_id from the conversation!

---

### Step 3: Validate Intent and Entities

```python
# tools/claims_api.py - Lines 102-142

# Validate intent exists
if not intent:
    logger.warning("⚠️ Validation failed: No intent provided")
    # Return error ToolResult
    return {"tool_results": tool_result.dict()}

# Validate entities exist
if not entities:
    logger.warning("⚠️ Validation failed: No entities provided")
    # Return error ToolResult
    return {"tool_results": tool_result.dict()}
```

**What this does:**
- Checks if intent is present (can't call API without knowing what to ask for!)
- Checks if entities are present (can't call API without the required information!)
- If either is missing → Returns error immediately (doesn't try to call API)

**Think of it like:** Checking you have both the phone number AND the information to ask before making the call.

**Example - Missing Intent:**
```python
# State has no intent:
state = {
    "entities": {"claim_id": "12345"}
    # Missing: "intent"
}

# Result:
{
    "tool_results": {
        "status": "failure",
        "error_message": "No intent provided in state",
        "error_code": "E1001"
    }
}
```

**Example - Missing Entities:**
```python
# State has intent but no entities:
state = {
    "intent": "claim_status"
    # Missing: "entities"
}

# Result:
{
    "tool_results": {
        "status": "failure",
        "error_message": "No entities provided in state",
        "error_code": "E1001"
    }
}
```

---

### Step 4: Normalize Entities (Convert to API Format)

```python
# tools/claims_api.py - Line 144

entities = normalize_entities(entities)
```

**What this does:**
- Converts entity names from **snake_case** (Python style) to **camelCase** (API style)
- Example: `claim_id` → `claimId`
- Merges entities from different sources (top-level and `raw_entities`)

**Why normalize?**
- Python code uses `claim_id` (snake_case)
- External APIs expect `claimId` (camelCase)
- The Tool Call Agent translates between the two!

**Entity Mapping:**
```python
# tools/claims_api.py - Lines 476-487

ENTITY_MAP = {
    "claim_number": "claimNumber",
    "member_id": "memberId",
    "prescription_number": "prescriptionNumber",
    "claim_sequence": "claimSequence",
    "claim_id": "claimId",
    # ... more mappings
}
```

**Example:**
```python
# Input (from Intent Agent):
entities = {
    "claim_id": "12345",
    "claim_sequence": "1"
}

# After normalization:
entities = {
    "claimId": "12345",        # claim_id → claimId
    "claimSequence": "1"       # claim_sequence → claimSequence
}
```

**Why this matters:**
- If you send `claim_id` to the API, it won't understand
- API expects `claimId` (camelCase)
- Normalization ensures the API gets the right field names

---

### Step 5: Special Case - Enriched Claim Details Flow

```python
# tools/claims_api.py - Lines 152-271

if intent == "claim_details":
    logger.info("🔄 Special flow detected: claim_details intent - using enriched 2-step flow")
    
    # Validate required entities
    if "claimNumber" not in entities or "claimSequence" not in entities:
        # Return error
        return {"tool_results": tool_result.dict()}
    
    # Execute enriched claim details flow
    result = combine_claim_details_and_list(
        claimNumber=entities["claimNumber"],
        claimSequence=entities["claimSequence"]
    )
```

**What this does:**
- For `claim_details` intent, uses a **special 2-step process**:
  1. Calls **Claim Details API** (gets detailed information)
  2. Calls **Claim List API** (gets summary information)
  3. **Merges** both responses together (enriched data)

**Why 2 steps?**
- Claim Details API has detailed information (pricing, dates, etc.)
- Claim List API has summary information (status, rejection codes, etc.)
- Combining both gives the **most complete** information!

**The 2-Step Flow:**
```python
# Step 1: Get claim details
claim_details = get_claim_details(claimNumber, claimSequence)
# Returns: Detailed pricing, dates, drug info, etc.

# Step 2: Get claim list
claim_list = get_claim_list(claimNumber, claimSequence)
# Returns: Status, rejection codes, summary info

# Step 3: Filter and merge
# Find matching claim in list
matched_claim = find_matching_claim(claim_list, claimNumber, claimSequence)

# Step 4: Combine
enriched_details = {
    **claim_details,           # All detailed info
    "list_data": matched_claim  # Summary info from list
}
```

**Example:**
```python
# User asks: "Show me details for claim 12345, sequence 1"

# Step 1: Call Details API
claim_details = {
    "claimDetails": {
        "primary": {
            "medD": {
                "approvedTotalAmount": "128.24",
                "approvedIngredientCost": "116.24",
                # ... lots of detailed fields
            }
        }
    }
}

# Step 2: Call List API
claim_list = {
    "claims": [{
        "claimInformation": {
            "claimStatus": "P",
            "claimStatusDescription": "Paid",
            "rejectCodes": []
        }
    }]
}

# Step 3: Merge
enriched = {
    **claim_details,
    "list_data": claim_list["claims"][0]
}

# Result: Complete information from both APIs!
```

---

### Step 6: Standard Flow - Match API from Registry

```python
# tools/claims_api.py - Lines 276-300

# Match API (Option A: treat no match as validation error)
logger.debug(f"🔍 Matching API for intent: {intent}")
api = match_api(intent, entities)

if not api:
    logger.warning(f"❌ No matching API found")
    # Return error
    return {"tool_results": tool_result.dict()}

logger.info(f"✅ [MATCH] Selected API → {api.name}")
```

**What this does:**
- Looks through the **API Registry** (list of available APIs)
- Finds the **best matching API** based on:
  - Intent keywords (does the intent match this API's keywords?)
  - Required entities (does the user have all required information?)
- Returns the matched API or `None` if no match

**Think of it like:** Looking through a phone book to find the right department to call.

**API Registry:**
```python
# tools/api_repository.py

registry = [
    API_REPOSITORY(
        name="get_claim_list",
        endpoint="/myclaims/claims/v1/list",
        required_entities=["claimId"],
        intent_keywords=["claim_search", "find_claim", "status", "check"],
        # ...
    ),
    API_REPOSITORY(
        name="get_claim_details",
        endpoint="/myclaims/claims/v1/details",
        required_entities=["claimNumber", "claimSequence"],
        intent_keywords=["details", "claim details"],
        # ...
    )
]
```

**Matching Algorithm:**
```python
# tools/claims_api.py - Lines 530-555

def match_api(intent: str, entities: Dict[str, Any]):
    """
    Scoring algorithm:
    1. Required entities check (disqualifies if missing)
    2. Intent keyword matching (+1 point per match)
    3. Specificity bonus (+N points where N = number of required entities)
    """
    for api in registry:
        # Step 1: Check required entities
        if not all(req in entities for req in api.required_entities):
            continue  # Skip this API (missing required info)
        
        # Step 2: Count intent keyword matches
        score = sum(1 for kw in api.intent_keywords if kw in intent.lower())
        
        # Step 3: Add specificity bonus
        score += len(api.required_entities)
        
        # Step 4: Track best match
        if score > best_score:
            best_api = api
    
    return best_api
```

**Example Matching:**
```python
# Input:
intent = "find_claim"
entities = {"claimId": "12345"}

# Scoring:
# API 1: get_claim_list
#   - Required entities: ["claimId"] ✅ Present
#   - Intent keywords: ["find_claim"] ✅ Matches "find_claim"
#   - Score: 1 (keyword match) + 1 (specificity) = 2 points ✅ WINNER

# API 2: get_claim_details
#   - Required entities: ["claimNumber", "claimSequence"] ❌ Missing claimSequence
#   - Score: DISQUALIFIED (missing required entity)

# Result: get_claim_list wins!
```

---

### Step 7: Build Request Body

```python
# tools/claims_api.py - Lines 304-329

# Build request body
logger.debug(f"🔨 Building request body for API: {api.name}")
try:
    body = api.body_template(entities)
    body["requester"] = {
        "xCorrelationId": str(uuid.uuid4()),
        "xConsumerAppName": "PSS-MYCLAIMSPOC-CLAIM-MFE"
    }
    logger.debug(f"✅ Request body built successfully")
except Exception as e:
    logger.error(f"❌ Failed to build request body: {e}")
    # Return error
    return {"tool_results": tool_result.dict()}
```

**What this does:**
- Uses the API's **body template** (a function that builds the request)
- Fills in the template with normalized entities
- Adds **requester metadata** (correlation ID, app name)
- Creates the final request body to send to the API

**Think of it like:** Writing down exactly what to ask for on the phone call.

**Body Template Example:**
```python
# API definition:
API_REPOSITORY(
    name="get_claim_list",
    body_template=lambda e: {
        "claimsRequest": {
            "claimId": e["claimId"]
        }
    }
)

# With entities = {"claimId": "12345"}:
body = {
    "claimsRequest": {
        "claimId": "12345"
    },
    "requester": {
        "xCorrelationId": "uuid-abc-123",
        "xConsumerAppName": "PSS-MYCLAIMSPOC-CLAIM-MFE"
    }
}
```

**Why requester metadata?**
- `xCorrelationId`: Unique ID to track this request (for debugging)
- `xConsumerAppName`: Identifies which app is making the call (for API logging)

---

### Step 8: Call External API (With Retry Logic)

```python
# tools/claims_api.py - Lines 346-390

# Call external API (sync call wrapped with retry decorator)
logger.info(f"🌐 Calling external API: {api.name} → {api.full_url}")
start = time.time()
try:
    result = call_external_api(api, body)
    elapsed_ms = (time.time() - start) * 1000.0
    
    logger.info(f"✅ API call succeeded in {elapsed_ms:.2f}ms")
    
    # Create success ToolResult
    tool_result = ToolResult(
        tool_name=api.name,
        status=ToolExecutionStatus.SUCCESS,
        data=result,
        execution_time_ms=elapsed_ms,
        api_endpoint=api.full_url,
        http_status_code=200
    )
    return {"tool_results": tool_result.dict()}
```

**What this does:**
- Makes an **HTTP POST request** to the external API
- Sends the request body (from Step 7)
- Waits for response
- Measures how long it took
- Wraps the response in a `ToolResult` object

**The Actual API Call:**
```python
# tools/claims_api.py - Lines 561-594

@retry(attempts=3, retry_on=(ExternalAPIError, ToolTimeoutError))
def call_external_api(api, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous HTTP call wrapper with automatic retry.
    """
    url = api.full_url  # e.g., "https://.../myclaims/claims/v1/list"
    method = "POST"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-correlation-id": "test",
        "x-consumerAppName": "LOCAL-TEST"
    }
    
    # Make HTTP request
    resp = requests.request(method, url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()  # Raises exception if HTTP error
    
    # Parse JSON response
    return resp.json()
```

**What happens behind the scenes:**
1. **HTTP Request**: Sends POST request to API URL
2. **Headers**: Tells API what format we want (JSON)
3. **Body**: Sends the request data (claimId, etc.)
4. **Timeout**: Waits up to 30 seconds for response
5. **Response**: Gets JSON data back
6. **Parse**: Converts JSON string to Python dictionary

**Example API Call:**
```python
# Request:
POST https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com/myclaims/claims/v1/list
Headers: {
    "Content-Type": "application/json"
}
Body: {
    "claimsRequest": {
        "claimId": "253152732536005"
    },
    "requester": {
        "xCorrelationId": "abc-123",
        "xConsumerAppName": "PSS-MYCLAIMSPOC-CLAIM-MFE"
    }
}

# Response (after ~1 second):
{
    "success": true,
    "totalCount": 1,
    "claims": [{
        "claimInformation": {
            "claimNumber": "253152732536005",
            "claimStatus": "P",
            "claimStatusDescription": "Paid"
        },
        "drug": {
            "productName": "ATORVASTATIN"
        },
        "pricing": {
            "patientPay": "10.00"
        }
    }]
}
```

---

### Step 9: Retry Logic (If API Call Fails)

```python
# tools/claims_api.py - Line 561

@retry(attempts=3, retry_on=(ExternalAPIError, ToolTimeoutError))
def call_external_api(api, body: Dict[str, Any]) -> Dict[str, Any]:
    # ... API call code ...
```

**What this does:**
- If the API call fails with a **retriable error** (timeout, server error), it **automatically retries**
- Tries up to **3 times** total
- Waits a bit longer between each retry (exponential backoff)

**Retry Strategy:**
```
Attempt 1: Call API immediately
  ↓ (fails with timeout)
Wait 0.5 seconds
  ↓
Attempt 2: Call API again
  ↓ (fails with timeout)
Wait 1.0 seconds
  ↓
Attempt 3: Call API again (final attempt)
  ↓ (succeeds or fails permanently)
```

**What errors are retriable?**
- **Timeouts**: API took too long to respond (network issue)
- **Server errors (5xx)**: API server had a problem (temporary)
- **Connection errors**: Can't reach the API (network issue)

**What errors are NOT retriable?**
- **Client errors (4xx)**: Bad request, not found, etc. (won't work if retried)
- **Validation errors**: Missing required fields (won't work if retried)

**Example:**
```python
# Attempt 1: Timeout after 30 seconds
# Wait 0.5 seconds

# Attempt 2: Timeout after 30 seconds
# Wait 1.0 seconds

# Attempt 3: Success! Got response in 2 seconds
# Total time: ~62.5 seconds (but got the data!)
```

---

### Step 10: Handle Errors (If API Call Fails)

```python
# tools/claims_api.py - Lines 392-429

except Exception as exc:
    elapsed_ms = (time.time() - start) * 1000.0
    logger.error(f"❌ API call failed after {elapsed_ms:.2f}ms: {str(exc)}")
    
    # Map exception to AgentError
    ae = to_agent_error(exc, node=api.name)
    
    # Create failure ToolResult
    tool_result = ToolResult(
        tool_name=api.name,
        status=ToolExecutionStatus.FAILURE,
        data={},
        error_message=ae.message,
        error_code=ae.error_code.value,
        agent_error=ae,
        api_endpoint=api.full_url,
        http_status_code=ae.metadata.get("status_code"),
        is_retryable=ae.is_retryable
    )
    return {"tool_results": tool_result.dict()}
```

**What this does:**
- If the API call fails (after all retries), catches the error
- Converts the error to a structured `AgentError` object
- Creates a `ToolResult` with failure status
- Returns the error result (doesn't crash the system!)

**Error Types:**

**1. Timeout Error:**
```python
{
    "tool_results": {
        "status": "failure",
        "error_message": "Timeout: Request exceeded 30s",
        "error_code": "E3001",
        "is_retryable": true  # Could retry later
    }
}
```

**2. HTTP 500 (Server Error):**
```python
{
    "tool_results": {
        "status": "failure",
        "error_message": "External API returned 500",
        "error_code": "E3001",
        "http_status_code": 500,
        "is_retryable": true  # Server might recover
    }
}
```

**3. HTTP 404 (Not Found):**
```python
{
    "tool_results": {
        "status": "failure",
        "error_message": "External API returned 404",
        "error_code": "E3001",
        "http_status_code": 404,
        "is_retryable": false  # Won't work if retried
    }
}
```

**Why handle errors gracefully?**
- System doesn't crash if API is down
- User gets a helpful error message
- Error is logged for debugging
- Response Agent can still generate a response (even if it's an error message)

---

### Step 11: Create ToolResult Object

```python
# tools/claims_api.py - Lines 374-387

tool_result = ToolResult(
    tool_name=getattr(api, "name", "claims_api"),
    status=ToolExecutionStatus.SUCCESS,
    data=result,  # API response data
    error_message=None,
    error_code=None,
    agent_error=None,
    execution_time_ms=elapsed_ms,
    api_endpoint=getattr(api, "full_url", None),
    http_status_code=200,
    is_retryable=False,
    metadata={}
)
```

**What this does:**
- Wraps the API response in a **standardized ToolResult object**
- Includes metadata: execution time, API endpoint, status code
- Makes it easy for Response Agent to use the data

**ToolResult Structure:**
```python
{
    "tool_name": "get_claim_list",           # Which API was called
    "status": "success",                     # success | failure | timeout
    "data": {                                # The actual API response
        "claims": [...],
        "totalCount": 1
    },
    "error_message": None,                   # Error message (if failed)
    "error_code": None,                      # Error code (if failed)
    "agent_error": None,                     # Full error object (if failed)
    "execution_time_ms": 1234.56,            # How long it took
    "api_endpoint": "https://.../list",      # Full URL called
    "http_status_code": 200,                 # HTTP status (200 = success)
    "is_retryable": False,                   # Can this be retried?
    "metadata": {}                           # Additional info
}
```

**Why use ToolResult?**
- **Standardized format**: All tools return the same structure
- **Easy to use**: Response Agent knows exactly what to expect
- **Error handling**: Success and failure look the same (just different status)
- **Observability**: Can track which APIs were called, how long they took, etc.

---

### Step 12: Return ToolResult to State

```python
# tools/claims_api.py - Lines 387-390

result_dict = {"tool_results": tool_result.dict()}
if isinstance(state, dict):
    await log_state_snapshot(state, "call_claims_tool", result_dict)
return result_dict
```

**What this does:**
- Wraps ToolResult in `{"tool_results": {...}}` dictionary
- Logs the state snapshot (for telemetry/debugging)
- Returns the result to LangGraph

**Why wrap in `{"tool_results": ...}`?**
- LangGraph merges returned dictionaries into state
- `return {"tool_results": {...}}` updates `state["tool_results"]`
- If you returned `tool_result.dict()` directly, it would create new top-level keys!

**Example:**
```python
# ✅ CORRECT:
return {"tool_results": tool_result.dict()}
# Updates: state["tool_results"] = {...}

# ❌ WRONG:
return tool_result.dict()
# Creates: state["tool_name"] = "...", state["status"] = "...", etc.
```

---

## 🎬 Complete Example: End-to-End

Let's trace through a real example:

### Input:
```python
state = {
    "text": "What's the status of claim #12345?",
    "intent": "claim_status",
    "entities": {"claim_id": "12345"},
    "session_id": "session_123"
}
```

### Step-by-Step Execution:

**1. Function called:**
```python
result = await call_claims_tool_node(state)
```

**2. Extract intent and entities:**
```python
intent = "claim_status"
entities = {"claim_id": "12345"}
```

**3. Validate:**
```python
# ✅ Intent present: "claim_status"
# ✅ Entities present: {"claim_id": "12345"}
# Continue...
```

**4. Normalize entities:**
```python
entities = normalize_entities({"claim_id": "12345"})
# Result: {"claimId": "12345"}
```

**5. Match API:**
```python
api = match_api("claim_status", {"claimId": "12345"})
# Matches: get_claim_list (has "status" keyword + has claimId)
# Result: APIEntry(name="get_claim_list", endpoint="/myclaims/claims/v1/list", ...)
```

**6. Build request body:**
```python
body = api.body_template({"claimId": "12345"})
# Result:
# {
#     "claimsRequest": {
#         "claimId": "12345"
#     },
#     "requester": {
#         "xCorrelationId": "uuid-abc-123",
#         "xConsumerAppName": "PSS-MYCLAIMSPOC-CLAIM-MFE"
#     }
# }
```

**7. Call external API:**
```python
result = call_external_api(api, body)
# Makes HTTP POST to: https://.../myclaims/claims/v1/list
# Waits for response...
# Result: {
#     "totalCount": 1,
#     "claims": [{
#         "claimInformation": {
#             "claimNumber": "12345",
#             "claimStatus": "P",
#             "claimStatusDescription": "Paid"
#         }
#     }]
# }
```

**8. Create ToolResult:**
```python
tool_result = ToolResult(
    tool_name="get_claim_list",
    status=ToolExecutionStatus.SUCCESS,
    data=result,
    execution_time_ms=1234.56,
    api_endpoint="https://.../list",
    http_status_code=200
)
```

**9. Return to state:**
```python
return {"tool_results": tool_result.dict()}
```

**10. LangGraph merges:**
```python
# Updated state:
state = {
    "text": "What's the status of claim #12345?",
    "intent": "claim_status",
    "entities": {"claim_id": "12345"},
    "tool_results": {  # ← Added!
        "tool_name": "get_claim_list",
        "status": "success",
        "data": {
            "totalCount": 1,
            "claims": [...]
        },
        "execution_time_ms": 1234.56,
        ...
    }
}
```

**11. Response Agent uses tool_results:**
```python
# Response Agent receives state with tool_results
# Generates natural language response from the claim data
# Returns: "Your claim #12345 is currently paid and processed."
```

---

## 🧠 Key Concepts Explained

### 1. **API Registry (The Phone Book)**

**What it is:**
- A list of all available external APIs
- Each API has: name, endpoint, required entities, intent keywords, body template

**Where it's defined:**
```python
# tools/api_repository.py

@lru_cache(maxsize=1)  # Built once, cached forever
def get_api_repository() -> List[API_REPOSITORY]:
    registry = [
        API_REPOSITORY(
            name="get_claim_list",
            endpoint="/myclaims/claims/v1/list",
            required_entities=["claimId"],
            intent_keywords=["claim_search", "find_claim", "status"],
            body_template=lambda e: {
                "claimsRequest": {"claimId": e["claimId"]}
            }
        ),
        # ... more APIs
    ]
    return registry
```

**Why it's useful:**
- **Centralized**: All APIs defined in one place
- **Easy to add**: Just add a new `API_REPOSITORY` entry
- **Dynamic matching**: Tool Call Agent automatically finds the right API

---

### 2. **Entity Normalization (Translation)**

**What it is:**
- Converting entity names from one format to another
- Python uses `snake_case` (claim_id)
- APIs use `camelCase` (claimId)

**The mapping:**
```python
ENTITY_MAP = {
    "claim_id": "claimId",
    "claim_number": "claimNumber",
    "member_id": "memberId",
    "claim_sequence": "claimSequence",
    # ... more mappings
}
```

**Why it's needed:**
- Different systems use different naming conventions
- Tool Call Agent translates automatically
- You don't have to remember which format each API uses!

---

### 3. **Retry Logic (Automatic Retry)**

**What it is:**
- If an API call fails with a retriable error, automatically tries again
- Up to 3 attempts total
- Waits longer between each retry

**When it retries:**
- Timeout errors (API took too long)
- Server errors (5xx - API had a problem)
- Connection errors (can't reach API)

**When it doesn't retry:**
- Client errors (4xx - bad request, won't work if retried)
- Validation errors (missing required fields)

**Example:**
```
Attempt 1: Call API → Timeout (30s)
Wait 0.5s
Attempt 2: Call API → Timeout (30s)
Wait 1.0s
Attempt 3: Call API → Success! (2s)
Total: ~62.5 seconds
```

---

### 4. **ToolResult (Standardized Response)**

**What it is:**
- A standardized format for all tool responses
- Includes: status, data, error info, metadata
- Makes it easy for Response Agent to use

**Structure:**
```python
{
    "tool_name": str,              # Which API was called
    "status": str,                 # "success" | "failure"
    "data": dict,                  # API response (if success)
    "error_message": str | None,  # Error message (if failure)
    "error_code": str | None,      # Error code (if failure)
    "execution_time_ms": float,    # How long it took
    "api_endpoint": str,           # Full URL called
    "http_status_code": int,       # HTTP status (200, 500, etc.)
    "is_retryable": bool           # Can this be retried?
}
```

**Why it's useful:**
- **Consistent**: All tools return the same format
- **Informative**: Includes all relevant metadata
- **Error-friendly**: Success and failure look the same (just different status)

---

### 5. **Enriched Claim Details (2-Step Flow)**

**What it is:**
- Special flow for `claim_details` intent
- Calls **two APIs** and merges the results
- Gets the most complete information possible

**The 2 steps:**
1. **Claim Details API**: Detailed information (pricing, dates, etc.)
2. **Claim List API**: Summary information (status, rejection codes, etc.)

**Why 2 steps?**
- Different APIs have different information
- Combining both gives the **most complete** picture
- User gets all the details they need!

**Example:**
```python
# Step 1: Get details
details = {
    "claimDetails": {
        "primary": {
            "medD": {
                "approvedTotalAmount": "128.24",
                # ... lots of detailed fields
            }
        }
    }
}

# Step 2: Get list
list_data = {
    "claims": [{
        "claimInformation": {
            "claimStatus": "P",
            "rejectCodes": []
        }
    }]
}

# Step 3: Merge
enriched = {
    **details,              # All detailed info
    "list_data": list_data  # Summary info
}
```

---

## 📊 Real-World Examples

### Example 1: Successful API Call

**User Query:**
```
"What's the status of claim #12345?"
```

**Flow:**
1. Intent Agent: `intent="claim_status", entities={"claim_id": "12345"}`
2. Tool Call Agent: Matches `get_claim_list` API
3. Tool Call Agent: Calls API with `{"claimId": "12345"}`
4. API Response: `{"claims": [{"claimStatus": "P", ...}]}`
5. Tool Call Agent: Returns ToolResult with success status
6. Response Agent: Generates "Your claim #12345 is paid and processed."

**ToolResult:**
```json
{
    "tool_name": "get_claim_list",
    "status": "success",
    "data": {
        "totalCount": 1,
        "claims": [{
            "claimInformation": {
                "claimNumber": "12345",
                "claimStatus": "P",
                "claimStatusDescription": "Paid"
            }
        }]
    },
    "execution_time_ms": 1234.56,
    "http_status_code": 200
}
```

---

### Example 2: API Timeout (With Retry)

**User Query:**
```
"Show me claim #12345"
```

**Flow:**
1. Intent Agent: `intent="claim_status", entities={"claim_id": "12345"}`
2. Tool Call Agent: Matches `get_claim_list` API
3. Tool Call Agent: Calls API → **Timeout after 30 seconds**
4. Tool Call Agent: Waits 0.5 seconds, retries → **Timeout again**
5. Tool Call Agent: Waits 1.0 seconds, retries → **Success!**
6. Tool Call Agent: Returns ToolResult with success (but took ~62 seconds)

**ToolResult:**
```json
{
    "tool_name": "get_claim_list",
    "status": "success",
    "data": {...},
    "execution_time_ms": 62500.0,  # Total time including retries
    "retry_count": 2  # Retried 2 times before success
}
```

---

### Example 3: Missing Required Entity

**User Query:**
```
"What's my claim status?"
```

**Flow:**
1. Intent Agent: `intent="claim_status", entities={}` (no claim_id!)
2. Tool Call Agent: Validates entities → **Missing claimId!**
3. Tool Call Agent: Returns error immediately (doesn't call API)

**ToolResult:**
```json
{
    "tool_name": "claims_api",
    "status": "failure",
    "error_message": "No entities provided in state",
    "error_code": "E1001",
    "is_retryable": false
}
```

**What happens next:**
- Router sees missing entities
- Routes to Clarification node
- Response Agent asks: "I need your claim ID to look that up. Could you provide it?"

---

### Example 4: Enriched Claim Details (2-Step)

**User Query:**
```
"Show me details for claim 12345, sequence 1"
```

**Flow:**
1. Intent Agent: `intent="claim_details", entities={"claim_number": "12345", "claim_sequence": "1"}`
2. Tool Call Agent: Detects `claim_details` intent → Uses enriched flow
3. **Step 1**: Calls Claim Details API → Gets detailed pricing, dates, etc.
4. **Step 2**: Calls Claim List API → Gets status, rejection codes, etc.
5. **Step 3**: Filters list to find matching claim
6. **Step 4**: Merges both responses
7. Tool Call Agent: Returns enriched ToolResult

**ToolResult:**
```json
{
    "tool_name": "claim_details_enriched",
    "status": "success",
    "data": {
        "claimDetails": {
            "primary": {
                "medD": {
                    "approvedTotalAmount": "128.24",
                    "approvedIngredientCost": "116.24"
                }
            }
        },
        "list_data": {
            "claimInformation": {
                "claimStatus": "P",
                "claimStatusDescription": "Paid"
            }
        }
    },
    "execution_time_ms": 2500.0  # Longer (2 API calls)
}
```

---

## 🔍 Code Deep Dive: Key Sections

### The API Matching Function

```python
# tools/claims_api.py - Lines 530-555

def match_api(intent: str, entities: Dict[str, Any]):
    """
    Simple scoring-based matcher:
      - required_entities must all be present
      - +1 per matched intent keyword (in intent lowercase)
      - +len(required_entities) bonus so more specific endpoints win ties
    """
    registry = get_api_repository()
    intent_lower = (intent or "").lower()

    matched_api = None
    matched_api_score = -1

    for api in registry:
        # Step 1: Ensure required entities present
        if not all(req in entities for req in getattr(api, "required_entities", [])):
            continue  # Skip this API (missing required info)

        # Step 2: Count intent keyword matches
        score = sum(1 for kw in getattr(api, "intent_keywords", []) if kw in intent_lower)
        
        # Step 3: Add specificity bonus
        score += len(getattr(api, "required_entities", []))

        # Step 4: Track best match
        if score > matched_api_score:
            matched_api_score = score
            matched_api = api

    return matched_api
```

**How it works:**
1. **Loops through all APIs** in the registry
2. **Checks required entities**: If user doesn't have all required info, skip this API
3. **Scores by keyword matches**: +1 point for each intent keyword that matches
4. **Adds specificity bonus**: +N points where N = number of required entities (more specific = higher score)
5. **Returns the highest scoring API**

**Example Scoring:**
```python
# Intent: "find_claim"
# Entities: {"claimId": "12345"}

# API 1: get_claim_list
#   Required: ["claimId"] ✅ Present
#   Keywords: ["find_claim", "status", "check"]
#   Score: 1 (keyword match) + 1 (specificity) = 2 ✅

# API 2: get_claim_details
#   Required: ["claimNumber", "claimSequence"] ❌ Missing claimSequence
#   Score: DISQUALIFIED

# Result: get_claim_list wins (score 2)
```

---

### The Retry Decorator

```python
# tools/claims_api.py - Line 561

@retry(attempts=3, retry_on=(ExternalAPIError, ToolTimeoutError))
def call_external_api(api, body: Dict[str, Any]) -> Dict[str, Any]:
    # ... API call code ...
```

**What the decorator does:**
- **Wraps the function** with retry logic
- **Catches specific exceptions**: `ExternalAPIError`, `ToolTimeoutError`
- **Retries up to 3 times** if one of these exceptions occurs
- **Waits between retries**: Exponential backoff (0.5s, 1.0s, etc.)

**How it works:**
```python
# First attempt
try:
    result = call_api()
    return result
except (ExternalAPIError, ToolTimeoutError) as e:
    if e.is_retryable and attempt < 3:
        wait(0.5 * attempt)  # Wait before retry
        # Retry...
    else:
        raise  # Give up
```

---

### Entity Normalization Function

```python
# tools/claims_api.py - Lines 490-525

def normalize_entities(entities_obj) -> Dict[str, Any]:
    """
    1. Merges top-level entities with raw_entities
    2. Normalizes all keys from snake_case to camelCase
    """
    # Handle both dicts and Pydantic models
    if isinstance(entities_obj, dict):
        all_entities = {k: v for k, v in entities_obj.items() 
                       if v is not None and k != 'raw_entities'}
        # Merge raw_entities if present
        if 'raw_entities' in entities_obj:
            all_entities.update(entities_obj['raw_entities'])
    else:
        # Pydantic model
        all_entities = entities_obj.model_dump(exclude_none=True, exclude={'raw_entities'})
        all_entities.update(entities_obj.raw_entities)
    
    # Normalize keys to camelCase
    normalized_entities = {}
    for k, v in all_entities.items():
        target_key = ENTITY_MAP.get(k, k)  # Use mapping if available
        normalized_entities[target_key] = v
    
    return normalized_entities
```

**What this does:**
1. **Merges entities**: Combines top-level entities with `raw_entities` dict
2. **Normalizes keys**: Converts `claim_id` → `claimId` using `ENTITY_MAP`
3. **Handles lists**: If entity value is a list, takes first element for singular API params
4. **Returns normalized dict**: All keys in camelCase format

**Example:**
```python
# Input:
entities = {
    "claim_id": "12345",
    "raw_entities": {
        "claim_sequence": "1"
    }
}

# After normalization:
normalized = {
    "claimId": "12345",        # claim_id → claimId
    "claimSequence": "1"       # claim_sequence → claimSequence
}
```

---

## 🎓 Key Takeaways

1. **Tool Call Agent = Smart API Caller**
   - Knows which API to call based on intent
   - Prepares the request correctly
   - Handles errors gracefully
   - Returns standardized results

2. **Uses API Registry**
   - Centralized list of all available APIs
   - Dynamic matching based on intent + entities
   - Easy to add new APIs

3. **Entity Normalization**
   - Translates between Python format (snake_case) and API format (camelCase)
   - Merges entities from multiple sources
   - Ensures APIs get the right field names

4. **Retry Logic**
   - Automatically retries on transient failures
   - Up to 3 attempts with exponential backoff
   - Only retries retriable errors

5. **Standardized ToolResult**
   - All tools return the same format
   - Includes success/failure, data, errors, metadata
   - Easy for Response Agent to use

6. **Special Flows**
   - Enriched claim details uses 2-step API calls
   - Combines data from multiple sources
   - Provides the most complete information

---

## 🔧 How to Add a New API

### Step 1: Add API to Registry

```python
# tools/api_repository.py

API_REPOSITORY(
    name="get_prescription_details",
    endpoint="/pharmacy/prescriptions/v1/details",
    method="POST",
    required_entities=["prescriptionNumber"],
    intent_keywords=["prescription", "rx", "medication", "details"],
    description="Fetch prescription details by prescription number",
    body_template=lambda e: {
        "prescriptionDetailsRequest": {
            "prescriptionNumber": e["prescriptionNumber"],
            "memberId": e.get("memberId")  # Optional
        }
    }
)
```

### Step 2: Add Entity Mapping (If Needed)

```python
# tools/claims_api.py

ENTITY_MAP = {
    # ... existing mappings ...
    "prescription_number": "prescriptionNumber",  # Add this
}
```

### Step 3: Test It

```python
# Test with IntentResult
test_case = IntentResult(
    intent="prescription_details",
    entities={"prescription_number": "RX12345"}
)

result = await call_claims_tool_node(test_case)
# Should match and call the new API!
```

---

## 📚 Summary

The Tool Call Agent is like a **smart assistant that makes phone calls**:
- **Input**: Intent (what user wants) + Entities (specific info like claim_id)
- **Process**: Find right API → Prepare request → Call API → Get data
- **Output**: ToolResult with API data (or error if failed)

It uses:
- **API Registry** to find the right API
- **Entity Normalization** to translate formats
- **Retry Logic** to handle transient failures
- **Standardized ToolResult** for consistent responses

The code is well-structured, handles errors gracefully, and works with both AgentState (from LangGraph) and IntentResult (for testing).

**That's how the Tool Call Agent works in this codebase!** 🎉

