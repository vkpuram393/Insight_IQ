# Claims API Orchestrator - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [API Repository](#api-repository)
6. [Entity Normalization](#entity-normalization)
7. [API Matching Logic](#api-matching-logic)
8. [Error Handling](#error-handling)
9. [Retry Mechanism](#retry-mechanism)
10. [State Management](#state-management)
11. [Testing](#testing)
12. [Configuration](#configuration)
13. [Examples](#examples)
14. [Troubleshooting](#troubleshooting)

---

## Overview

The **Claims API Orchestrator** is a dynamic, intent-based routing system that automatically selects and calls the appropriate external API based on user intent and extracted entities. It's designed as a LangGraph node that seamlessly integrates into the agent workflow.

### Key Features
- ✅ **Dynamic API Selection**: Automatically routes to the correct API based on intent + entities
- ✅ **Automatic Retry**: Built-in exponential backoff retry for transient failures
- ✅ **Entity Normalization**: Converts snake_case to camelCase and handles nested entities
- ✅ **Comprehensive Error Handling**: Structured error responses with AgentError model
- ✅ **State Integration**: Properly updates AgentState.tool_results for LangGraph flow
- ✅ **Telemetry & Logging**: Full audit trail in SQLite database
- ✅ **Dual Input Support**: Works with AgentState (LangGraph) or IntentResult (testing)

### Entry Point
```python
async def call_claims_tool_node(state) -> Dict[str, Any]
```

**Location**: `tools/claims_api.py`

---

## Architecture

### High-Level Flow

```
┌─────────────────────┐
│  LangGraph / User   │
│   (AgentState)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│   call_claims_tool_node()                   │
│   ┌─────────────────────────────────────┐  │
│   │ 1. Extract intent & entities        │  │
│   │ 2. Normalize entities (snake→camel) │  │
│   │ 3. Match API from registry          │  │
│   │ 4. Build request body               │  │
│   │ 5. Call external API (with retry)   │  │
│   │ 6. Wrap result in ToolResult        │  │
│   │ 7. Return {tool_results: {...}}     │  │
│   └─────────────────────────────────────┘  │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  AgentState Updated                         │
│  state["tool_results"] = {                  │
│    "tool_name": "get_claim_list",           │
│    "status": "success",                     │
│    "data": { ... },                         │
│    "execution_time_ms": 1234.56,            │
│    ...                                      │
│  }                                          │
└─────────────────────────────────────────────┘
```

### Component Diagram

```
┌────────────────────────────────────────────────────────────┐
│                    Claims API System                        │
│                                                             │
│  ┌──────────────────┐      ┌────────────────────────┐     │
│  │  claims_api.py   │◄─────│ get_api_repository.py  │     │
│  │  (Orchestrator)  │      │  (API Registry)        │     │
│  └────────┬─────────┘      └────────────────────────┘     │
│           │                                                 │
│           ├──────────────┐                                 │
│           │              │                                 │
│           ▼              ▼                                 │
│  ┌──────────────┐  ┌──────────────┐                      │
│  │ retry.py     │  │error_handler │                      │
│  │ (Retry Logic)│  │   .py        │                      │
│  └──────────────┘  └──────────────┘                      │
│           │              │                                 │
│           │              ▼                                 │
│           │      ┌──────────────┐                         │
│           │      │exceptions.py │                         │
│           │      │(Custom Errors)│                        │
│           │      └──────────────┘                         │
│           │                                                │
│           ▼                                                │
│  ┌──────────────────────────────┐                        │
│  │   External APIs              │                        │
│  │  - Claim List API            │                        │
│  │  - Claim Details API         │                        │
│  └──────────────────────────────┘                        │
└────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. `claims_api.py` - Main Orchestrator

**Primary Function**: `call_claims_tool_node(state)`

**Responsibilities**:
- Accept AgentState (dict) or IntentResult (Pydantic model)
- Extract and normalize entities
- Match the appropriate API
- Build request payload
- Execute API call with retry
- Handle errors gracefully
- Return structured ToolResult wrapped in `{"tool_results": {...}}`

**Key Functions**:

#### `normalize_entities(entities_obj) -> Dict[str, Any]`
Merges and normalizes entities from both top-level and `raw_entities` fields.

**Process**:
1. Extracts top-level entities (excludes `raw_entities` key itself)
2. Merges contents of `raw_entities` dict
3. Maps snake_case keys to camelCase using `ENTITY_MAP`

**Example**:
```python
Input:
{
  "claim_number": "12345",
  "raw_entities": {
    "claim_sequence": "1"
  }
}

Output:
{
  "claimNumber": "12345",
  "claimSequence": "1"
}
```

#### `match_api(intent: str, entities: Dict[str, Any]) -> Optional[APIEntry]`
Scores and selects the best matching API from the registry.

**Scoring Algorithm**:
1. **Required entities check**: All `required_entities` must be present in input
2. **Intent keyword matching**: +1 point per matched keyword
3. **Specificity bonus**: +N points where N = number of required entities

**Returns**: APIEntry with highest score, or None if no match

#### `call_external_api(api, body: Dict[str, Any]) -> Dict[str, Any]`
Synchronous HTTP wrapper with automatic retry decorator.

**Features**:
- Sets standard headers (correlation-id, consumer app name)
- 10-second timeout
- Raises `ToolTimeoutError` for timeouts (retriable)
- Raises `ExternalAPIError` for HTTP errors
- Validates JSON response

---

### 2. `get_api_repository.py` - API Registry

**Function**: `get_api_repository() -> List[APIEntry]`

**Caching**: `@lru_cache(maxsize=1)` - Built once, reused forever

**Current APIs**:

#### API 1: Get Claim Details
```python
APIEntry(
    name="get_claim_details",
    endpoint="/myclaims/claims/v1/details",
    method="POST",
    required_entities=["claimNumber", "claimSequence"],
    intent_keywords=["details", "claim details"],
    body_template=lambda e: {
        "claimDetailsRequest": {
            "claimNumber": e["claimNumber"],
            "claimSequence": e["claimSequence"],
            "expMockFlag": e.get("expMockFlag", "N")
        }
    }
)
```

#### API 2: Get Claim List
```python
APIEntry(
    name="get_claim_list",
    endpoint="/myclaims/claims/v1/list",
    method="POST",
    required_entities=["claimId"],
    intent_keywords=["claim_search", "find_claim", "lookup_claim", "status", "check", "track"],
    body_template=lambda e: {
        "claimsRequest": {
            "claimId": e["claimId"]
        }
    }
)
```

**Adding New APIs**:
1. Create new `APIEntry` in the registry list
2. Define `required_entities` (normalized camelCase names)
3. Define `intent_keywords` for matching
4. Create `body_template` lambda to build request
5. Full URL is auto-constructed from `BASE_URL` + `endpoint`

---

### 3. `error_handler.py` - Error Normalization

**Function**: `to_agent_error(exc: Exception, *, node: Optional[str] = None) -> AgentError`

**Purpose**: Convert any exception to structured `AgentError` model

**Exception Mapping**:

| Exception Type | Maps To | Is Retryable |
|----------------|---------|--------------|
| `ToolTimeoutError` | API Error (timeout) | ✅ Yes |
| `ExternalAPIError` | API Error | Depends on status code |
| `APIBaseError` | API Error | Based on exception flag |
| Any other Exception | Internal Error | ❌ No |

**AgentError Structure**:
```python
{
  "error_code": "E3001",           # Standardized error code
  "category": "api",               # api/validation/system
  "message": "API call failed",    # Human-readable message
  "severity": "error",             # info/warning/error/critical
  "is_retryable": True,            # Can retry?
  "metadata": {                    # Additional context
    "status_code": 500,
    "api_name": "get_claim_list"
  }
}
```

---

### 4. `retry.py` - Retry Decorator

**Function**: `@retry(attempts=3, retry_on=(ExternalAPIError, ToolTimeoutError))`

**Behavior**:
- **Max Attempts**: 3 (configurable)
- **Backoff Strategy**: Linear (0.5 * attempt seconds)
- **Retry Conditions**: 
  - Exception type must be in `retry_on` tuple
  - Exception must have `retriable=True` attribute

**Retry Sequence**:
```
Attempt 1: Execute immediately
  ↓ (fails with retriable error)
Wait 0.5 seconds
  ↓
Attempt 2: Execute
  ↓ (fails with retriable error)
Wait 1.0 seconds
  ↓
Attempt 3: Execute (final)
  ↓ (fails)
Raise exception
```

**Non-Retryable Cases**:
- 4xx client errors (bad request, not found, etc.)
- `retriable=False` flag set
- Non-timeout connection errors

---

### 5. `exceptions.py` - Custom Exceptions

#### Base Class: `APIBaseError`
```python
class APIBaseError(Exception):
    def __init__(
        self, 
        message: str, 
        *, 
        details: Optional[Dict[str, Any]] = None, 
        retriable: bool = False
    )
```

**Attributes**:
- `message`: Error description
- `details`: Arbitrary metadata dict
- `retriable`: Flag for retry logic

#### Derived Classes:

**`ExternalAPIError`**
- Raised for HTTP errors (4xx, 5xx)
- Raised for invalid JSON responses
- Can be retriable (5xx) or non-retriable (4xx)

**`ToolTimeoutError`**
- Raised when request exceeds timeout
- Always retriable by default

---

## Data Flow

### Input: AgentState (from LangGraph)

```python
{
  "text": "Show me claim 253152732536005",
  "session_id": "abc-123",
  "user_info": {"user_id": "U001"},
  "uuid": "req-456",
  "domain": "claims",
  "intent": "find_claim",
  "confidence": 0.92,
  "entities": {
    "raw_entities": {
      "claim_id": "253152732536005"
    }
  },
  "tool_results": None,  # ← Will be updated
  # ... other fields
}
```

### Processing Steps

**Step 1: Entity Extraction**
```python
intent = state.get("intent")  # "find_claim"
entities = state.get("entities", {})
# {"raw_entities": {"claim_id": "253152732536005"}}
```

**Step 2: Entity Normalization**
```python
entities = normalize_entities(entities)
# {"claimId": "253152732536005"}
```

**Step 3: API Matching**
```python
api = match_api(intent, entities)
# Returns: APIEntry(name="get_claim_list", ...)
```

**Step 4: Request Body Building**
```python
body = api.body_template(entities)
# {
#   "claimsRequest": {
#     "claimId": "253152732536005"
#   }
# }

# Add requester metadata
body["requester"] = {
  "xCorrelationId": "uuid-...",
  "xConsumerAppName": "PSS-MYCLAIMSPOC-CLAIM-MFE"
}
```

**Step 5: API Call**
```python
result = call_external_api(api, body)
# Returns parsed JSON response
```

**Step 6: ToolResult Creation**
```python
tool_result = ToolResult(
    tool_name="get_claim_list",
    status=ToolExecutionStatus.SUCCESS,
    data=result,
    error_message=None,
    error_code=None,
    agent_error=None,
    execution_time_ms=1234.56,
    api_endpoint="https://...",
    http_status_code=200,
    is_retryable=False
)
```

**Step 7: State Update**
```python
return {"tool_results": tool_result.dict()}
```

### Output: Updated AgentState

```python
{
  # ... all previous fields unchanged ...
  "tool_results": {
    "tool_name": "get_claim_list",
    "status": "success",
    "data": {
      "success": true,
      "message": "Claims retrieved successfully",
      "totalCount": 1,
      "claims": [ /* full claim data */ ]
    },
    "error_message": null,
    "error_code": null,
    "agent_error": null,
    "execution_time_ms": 1234.56,
    "api_endpoint": "https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com/myclaims/claims/v1/list",
    "http_status_code": 200,
    "retry_count": 0,
    "is_retryable": false,
    "from_cache": false,
    "cache_key": null,
    "timestamp": "2025-11-18T12:27:22.943932+00:00"
  }
}
```

---

## API Repository

### APIEntry Model

```python
class APIEntry(BaseModel):
    name: str                                    # Unique identifier
    endpoint: str                                # API path
    method: str = "POST"                         # HTTP method
    required_entities: List[str]                 # Must-have entities
    intent_keywords: List[str]                   # Keywords for matching
    description: Optional[str] = None            # Human-readable description
    body_template: Callable[[Dict], Dict]        # Request builder lambda
    full_url: Optional[str] = None              # Complete URL (auto-set)
```

### How to Add a New API

**Example: Adding "Get Prescription Details"**

```python
# In get_api_repository.py

APIEntry(
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

**Considerations**:
1. **Entity Names**: Use camelCase (matches what `normalize_entities` outputs)
2. **Intent Keywords**: Broad enough to catch variations, specific enough to avoid conflicts
3. **Body Template**: Lambda receives normalized entities dict
4. **Required vs Optional**: Use `.get()` for optional entities

---

## Entity Normalization

### ENTITY_MAP Configuration

```python
ENTITY_MAP = {
    "claim_number": "claimNumber",
    "member_id": "memberId",
    "prescription_number": "prescriptionNumber",
    "medication_name": "medicationName",
    "date_from": "dateFrom",
    "date_to": "dateTo",
    "claim_sequence": "claimSequence",
    "claim_id": "claimId",
}
```

### Normalization Process

**Scenario 1: Top-level entities only**
```python
Input:
{
  "claim_number": "12345",
  "member_id": "M001"
}

Output:
{
  "claimNumber": "12345",
  "memberId": "M001"
}
```

**Scenario 2: raw_entities only**
```python
Input:
{
  "raw_entities": {
    "claim_id": "67890"
  }
}

Output:
{
  "claimId": "67890"
}
```

**Scenario 3: Mixed (merges both)**
```python
Input:
{
  "claim_number": "12345",
  "raw_entities": {
    "claim_sequence": "1",
    "member_id": "M001"
  }
}

Output:
{
  "claimNumber": "12345",
  "claimSequence": "1",
  "memberId": "M001"
}
```

**Scenario 4: Unmapped keys (pass-through)**
```python
Input:
{
  "custom_field": "value",
  "claim_number": "12345"
}

Output:
{
  "custom_field": "value",    # No mapping, kept as-is
  "claimNumber": "12345"
}
```

### Adding New Entity Mappings

```python
# Add to ENTITY_MAP in claims_api.py
ENTITY_MAP = {
    # ... existing mappings ...
    "new_snake_case": "newCamelCase",
}
```

---

## API Matching Logic

### Scoring Algorithm

```python
def match_api(intent: str, entities: Dict[str, Any]):
    intent_l = intent.lower()
    best_api = None
    best_score = -1
    
    for api in registry:
        # Step 1: Required entities check (disqualifies if missing)
        if not all(req in entities for req in api.required_entities):
            continue
        
        # Step 2: Count intent keyword matches
        score = sum(1 for kw in api.intent_keywords if kw in intent_l)
        
        # Step 3: Add specificity bonus
        score += len(api.required_entities)
        
        # Step 4: Track best
        if score > best_score:
            best_score = score
            best_api = api
    
    return best_api
```

### Matching Examples

**Example 1: Single Match**
```python
Intent: "find_claim"
Entities: {"claimId": "12345"}

Scores:
- get_claim_list:    "find_claim" matches → 1 point
                     + 1 required entity → +1 point
                     = 2 points ✅ WINNER

- get_claim_details: Missing "claimSequence" → DISQUALIFIED
```

**Example 2: Tie-Breaking (Specificity Wins)**
```python
Intent: "details"
Entities: {"claimNumber": "12345", "claimSequence": "1"}

Scores:
- get_claim_details: "details" matches → 1 point
                     + 2 required entities → +2 points
                     = 3 points ✅ WINNER

- get_claim_list:    "details" not in keywords → 0 points
                     (even though entities match)
                     = 0 points
```

**Example 3: No Match**
```python
Intent: "random_intent"
Entities: {"foo": "bar"}

Result: None (no API has required entities present)
```

---

## Error Handling

### Error Flow Diagram

```
┌─────────────────────┐
│   API Call Start    │
└──────────┬──────────┘
           │
           ▼
     ┌─────────┐
     │ Success?│
     └────┬────┘
          │
    ┌─────┴─────┐
    │           │
   Yes          No
    │           │
    │           ▼
    │    ┌──────────────┐
    │    │ Timeout?     │
    │    │ HTTP Error?  │
    │    │ JSON Error?  │
    │    │ Other?       │
    │    └──────┬───────┘
    │           │
    │           ▼
    │    ┌──────────────────┐
    │    │ to_agent_error() │
    │    │ (Normalize)      │
    │    └──────┬───────────┘
    │           │
    ▼           ▼
┌─────────────────────────┐
│   ToolResult Creation   │
│  (SUCCESS or FAILURE)   │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Return {"tool_results": │
│    tool_result.dict()}  │
└─────────────────────────┘
```

### Error Categories

#### 1. Validation Errors (Before API Call)

**Scenario A: No Intent**
```python
{
  "tool_results": {
    "tool_name": "claims_api",
    "status": "failure",
    "error_message": "No intent provided in state",
    "error_code": "E1001",
    "agent_error": {
      "error_code": "E1001",
      "category": "validation",
      "severity": "error"
    }
  }
}
```

**Scenario B: No Entities**
```python
{
  "tool_results": {
    "tool_name": "claims_api",
    "status": "failure",
    "error_message": "No entities provided in state",
    "error_code": "E1001",
    "agent_error": { /* ... */ }
  }
}
```

**Scenario C: No Matching API**
```python
{
  "tool_results": {
    "tool_name": "claims_api",
    "status": "failure",
    "error_message": "No matching API found for given intent/entities",
    "error_code": "E1001",
    "agent_error": { /* ... */ }
  }
}
```

#### 2. API Errors (During/After API Call)

**Scenario A: Timeout**
```python
{
  "tool_results": {
    "tool_name": "get_claim_list",
    "status": "failure",
    "error_message": "Timeout: Request exceeded 10s",
    "error_code": "E3001",
    "is_retryable": true,  # ← Can retry
    "agent_error": {
      "category": "api",
      "is_retryable": true
    }
  }
}
```

**Scenario B: HTTP 500 (Server Error)**
```python
{
  "tool_results": {
    "tool_name": "get_claim_list",
    "status": "failure",
    "error_message": "External API returned 500",
    "error_code": "E3001",
    "http_status_code": 500,
    "is_retryable": true,  # ← Can retry
    "agent_error": { /* ... */ }
  }
}
```

**Scenario C: HTTP 404 (Not Found)**
```python
{
  "tool_results": {
    "tool_name": "get_claim_list",
    "status": "failure",
    "error_message": "External API returned 404",
    "error_code": "E3001",
    "http_status_code": 404,
    "is_retryable": false,  # ← Don't retry
    "agent_error": { /* ... */ }
  }
}
```

#### 3. Internal Errors (System Failures)

**Scenario: Unexpected Exception**
```python
{
  "tool_results": {
    "tool_name": "claims_api",
    "status": "failure",
    "error_message": "Unexpected error: division by zero",
    "error_code": "E9001",
    "agent_error": {
      "error_code": "E9001",
      "category": "system",
      "severity": "critical",
      "stacktrace": "Traceback (most recent call last):\n..."
    }
  }
}
```

---

## Retry Mechanism

### Configuration

```python
@retry(
    attempts=3,
    retry_on=(ExternalAPIError, ToolTimeoutError)
)
def call_external_api(api, body): ...
```

### Retry Decision Matrix

| Error Type | Status Code | retriable Flag | Will Retry? |
|------------|-------------|----------------|-------------|
| `ToolTimeoutError` | N/A | True | ✅ Yes |
| `ExternalAPIError` | 500-599 | True | ✅ Yes |
| `ExternalAPIError` | 400-499 | False | ❌ No |
| `ExternalAPIError` | None (connection) | True | ✅ Yes |
| Other exceptions | N/A | N/A | ❌ No |

### Backoff Timing

```
Attempt 1: t=0s
  ↓ (fails)
Wait: 0.5s
  ↓
Attempt 2: t=0.5s
  ↓ (fails)
Wait: 1.0s
  ↓
Attempt 3: t=1.5s
  ↓ (fails)
Raise exception
```

**Total retry time**: ~1.5 seconds across 3 attempts

### Monitoring Retries

The `retry_count` field in ToolResult tracks attempts:
```python
{
  "tool_results": {
    "retry_count": 2,  # ← Failed twice, succeeded on 3rd attempt
    "execution_time_ms": 2500,  # Total time including retries
    ...
  }
}
```

---

## State Management

### Why Wrap in `{"tool_results": ...}`?

LangGraph merges returned dictionaries into the current state. To update the `tool_results` field in AgentState:

❌ **WRONG** (Returns raw ToolResult dict)
```python
return tool_result.dict()
# Result: Creates new top-level keys in state like "tool_name", "status", etc.
```

✅ **CORRECT** (Wraps in tool_results key)
```python
return {"tool_results": tool_result.dict()}
# Result: Updates state["tool_results"] with the complete ToolResult
```

### State Before API Call

```python
{
  "intent": "find_claim",
  "entities": {"claimId": "12345"},
  "tool_results": None,  # ← Empty
  # ... other fields
}
```

### State After API Call

```python
{
  "intent": "find_claim",
  "entities": {"claimId": "12345"},
  "tool_results": {       # ← Populated!
    "tool_name": "get_claim_list",
    "status": "success",
    "data": { /* claim data */ },
    "execution_time_ms": 1234.56,
    ...
  },
  # ... other fields
}
```

### Accessing Results in Downstream Nodes

```python
# In response_agent.py or other nodes
async def response_agent_node(state):
    tool_results = state.get("tool_results")
    
    if tool_results and tool_results.get("status") == "success":
        claim_data = tool_results["data"]
        # Generate response using claim_data
    else:
        error_msg = tool_results.get("error_message")
        # Handle error case
```

---

## Testing

### Test File: `tools/test_claims_api.py`

**Purpose**: Standalone testing without full LangGraph setup

**Usage**:
```bash
python -m tools.test_claims_api
```

**Test Cases**:

#### Test 1: Successful Match
```python
IntentResult(
    intent="find_claim",
    confidence=0.88,
    entities={"raw_entities": {"claim_id": "253152732536005"}}
)

Expected: Matches get_claim_list API, returns success
```

#### Test 2: Claim Details
```python
IntentResult(
    intent="claim_details",
    confidence=0.95,
    entities={
        "claim_number": "253152732536005",
        "raw_entities": {"claim_sequence": "1"}
    }
)

Expected: Matches get_claim_details API
```

#### Test 3: No Match
```python
IntentResult(
    intent="random_intent",
    confidence=0.93,
    entities={"raw_entities": {"foo": "bar"}}
)

Expected: No matching API error
```

### Test Endpoint: `/utils/test-claims-api`

**Purpose**: HTTP endpoint for testing via API calls

**Request**:
```json
POST /utils/test-claims-api
{
  "text": "Show me claim 253152732536005",
  "intent": "find_claim",
  "entities": {
    "raw_entities": {
      "claim_id": "253152732536005"
    }
  }
}
```

**Response**:
```json
{
  "test_info": {
    "session_id": "abc-123",
    "request_uuid": "req-456",
    "intent": "find_claim",
    "entities": { /* normalized */ }
  },
  "tool_execution": {
    "tool_name": "get_claim_list",
    "status": "success",
    "api_endpoint": "https://...",
    "execution_time_ms": 1234.56,
    "http_status_code": 200
  },
  "result": {
    "success": true,
    "has_data": true,
    "data_size": 1
  },
  "full_tool_results": {
    /* Complete ToolResult object */
  },
  "state_update": {
    "tool_results_field_updated": true
  }
}
```

---

## Configuration

### Environment Variables

**Required**:
```bash
SWAGGER_URL="https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com"
```

**Optional**:
```bash
# Telemetry (default: enabled)
ENABLE_TELEMETRY=true

# Database path (default: data/telemetry.db)
TELEMETRY_DB_PATH="data/telemetry.db"

# Persistence store type (default: sqlite)
PERSISTENCE_STORE_TYPE="sqlite"
```

### Timeout Configuration

**Current**: 10 seconds (hardcoded in `call_external_api`)

**To Change**:
```python
# In claims_api.py, line 127
resp = requests.request(method, url, headers=headers, json=body, timeout=30)  # 30s
```

### Retry Configuration

**Current**: 3 attempts with 0.5s linear backoff

**To Change**:
```python
# In claims_api.py, line 110
@retry(attempts=5, retry_on=(ExternalAPIError, ToolTimeoutError))
```

---

## Examples

### Example 1: Find Claim by ID

**Input**:
```python
state = {
    "intent": "find_claim",
    "entities": {
        "raw_entities": {
            "claim_id": "253152732536005"
        }
    }
}
```

**Processing**:
1. Normalize: `{"claimId": "253152732536005"}`
2. Match API: `get_claim_list` (matches "find_claim" keyword + has claimId)
3. Build body:
   ```json
   {
     "claimsRequest": {
       "claimId": "253152732536005"
     },
     "requester": {
       "xCorrelationId": "uuid...",
       "xConsumerAppName": "PSS-MYCLAIMSPOC-CLAIM-MFE"
     }
   }
   ```
4. Call API: POST to `/myclaims/claims/v1/list`
5. Parse response and return

**Output**:
```python
{
  "tool_results": {
    "tool_name": "get_claim_list",
    "status": "success",
    "data": {
      "totalCount": 1,
      "claims": [
        {
          "claimInformation": {
            "claimNumber": "253152732536005",
            "claimStatus": "P",
            "claimStatusDescription": "Paid",
            ...
          }
        }
      ]
    }
  }
}
```

### Example 2: Get Claim Details

**Input**:
```python
state = {
    "intent": "claim_details",
    "entities": {
        "claim_number": "253152732536005",
        "raw_entities": {
            "claim_sequence": "1"
        }
    }
}
```

**Processing**:
1. Normalize: `{"claimNumber": "253152732536005", "claimSequence": "1"}`
2. Match API: `get_claim_details` (matches "details" + has both required entities)
3. Build body:
   ```json
   {
     "claimDetailsRequest": {
       "claimNumber": "253152732536005",
       "claimSequence": "1",
       "expMockFlag": "N"
     },
     "requester": { ... }
   }
   ```
4. Call API: POST to `/myclaims/claims/v1/details`

### Example 3: Error Case - Missing Entity

**Input**:
```python
state = {
    "intent": "claim_details",
    "entities": {
        "claim_number": "12345"
        # Missing claim_sequence!
    }
}
```

**Output**:
```python
{
  "tool_results": {
    "tool_name": "claims_api",
    "status": "failure",
    "error_message": "No matching API found for given intent/entities",
    "error_code": "E1001",
    "is_retryable": false
  }
}
```

---

## Troubleshooting

### Issue 1: "No matching API found"

**Symptoms**:
```
error_message: "No matching API found for given intent/entities"
```

**Causes**:
1. ✗ Required entities missing from input
2. ✗ Entity names don't match (wrong case or spelling)
3. ✗ Intent keywords don't match any API

**Solutions**:
- Check `entities` dict contains all `required_entities` for target API
- Verify entity names are in camelCase (after normalization)
- Add more `intent_keywords` to API definition
- Check ENTITY_MAP for proper snake_case → camelCase mapping

**Debug**:
```python
# Add logging in claims_api.py
logger.info(f"Normalized entities: {entities}")
logger.info(f"Available APIs: {[api.name for api in get_api_repository()]}")
```

### Issue 2: "success: false" in Test Endpoint

**Symptoms**:
```json
"result": {
  "success": false
}
```

**Cause**: Status comparison was using uppercase `"SUCCESS"` but enum value is lowercase `"success"`

**Verification**:
Check `full_tool_results.status` field:
```json
"full_tool_results": {
  "status": "success"  // ← lowercase = correct
}
```

### Issue 3: Timeout Errors

**Symptoms**:
```
error_message: "Timeout: Request exceeded 10s"
is_retryable: true
```

**Solutions**:
1. Increase timeout in `call_external_api` (line 127)
2. Check network connectivity to external API
3. Verify BASE_URL is correct
4. Check if external API is responsive

**Debug**:
```python
# Test external API directly
curl -X POST https://BASE_URL/myclaims/claims/v1/list \
  -H "Content-Type: application/json" \
  -d '{"claimsRequest": {"claimId": "12345"}}'
```

### Issue 4: Entity Not Normalized

**Symptoms**: API call fails because body has wrong field names

**Example**:
```python
# Sent: {"claim_id": "12345"}  ← snake_case
# Expected: {"claimId": "12345"}  ← camelCase
```

**Solution**: Add mapping to ENTITY_MAP
```python
ENTITY_MAP = {
    # ...
    "claim_id": "claimId",  # Add this mapping
}
```

### Issue 5: Tool Results Not in State

**Symptoms**: Downstream nodes can't access `state["tool_results"]`

**Cause**: Not wrapping return value properly

**Solution**: Ensure all returns are wrapped:
```python
# ✗ WRONG
return tool_result.dict()

# ✅ CORRECT
return {"tool_results": tool_result.dict()}
```

### Issue 6: Duplicate API Data in Response

**Symptoms**: Same data appears in both `result.data` and `full_tool_results.data`

**Cause**: Test endpoint design intentionally includes both for convenience

**Not a Bug**: This is expected behavior
- `result`: Quick access fields
- `full_tool_results`: Complete ToolResult (source of truth)

---

## API Reference

### Function: `call_claims_tool_node(state)`

**Parameters**:
- `state`: Union[AgentState, IntentResult]
  - From LangGraph: dict with AgentState fields
  - From tests: IntentResult Pydantic model

**Returns**: `Dict[str, Any]`
- Always returns `{"tool_results": ToolResult.dict()}`

**Example**:
```python
result = await call_claims_tool_node(state)
# result = {
#   "tool_results": {
#     "tool_name": "...",
#     "status": "success" | "failure",
#     ...
#   }
# }
```

### Function: `normalize_entities(entities_obj)`

**Parameters**:
- `entities_obj`: Union[Dict, PydanticModel]

**Returns**: `Dict[str, Any]`
- Normalized, merged entities in camelCase

**Example**:
```python
normalized = normalize_entities({
    "claim_number": "12345",
    "raw_entities": {"claim_sequence": "1"}
})
# Result: {"claimNumber": "12345", "claimSequence": "1"}
```

### Function: `match_api(intent, entities)`

**Parameters**:
- `intent`: str - User intent
- `entities`: Dict[str, Any] - Normalized entities

**Returns**: `Optional[APIEntry]`
- Matched API or None

**Example**:
```python
api = match_api("find_claim", {"claimId": "12345"})
# Returns: APIEntry(name="get_claim_list", ...)
```

### Function: `call_external_api(api, body)`

**Parameters**:
- `api`: APIEntry - Matched API entry
- `body`: Dict[str, Any] - Request payload

**Returns**: `Dict[str, Any]`
- Parsed JSON response

**Raises**:
- `ToolTimeoutError`: Request timeout
- `ExternalAPIError`: HTTP error or invalid JSON

---

## Best Practices

### 1. Always Normalize Entities Early
```python
# ✅ GOOD
entities = normalize_entities(state.get("entities", {}))
api = match_api(intent, entities)

# ✗ BAD
api = match_api(intent, state.get("entities"))  # Unnormalized
```

### 2. Use Specific Intent Keywords
```python
# ✅ GOOD - Specific, unambiguous
intent_keywords=["prescription_details", "rx_details", "medication_info"]

# ✗ BAD - Too generic, conflicts likely
intent_keywords=["details", "info", "get"]
```

### 3. Handle Both Success and Failure
```python
# ✅ GOOD
tool_results = state.get("tool_results")
if tool_results.get("status") == "success":
    data = tool_results["data"]
else:
    error = tool_results.get("error_message")
    # Handle error
```

### 4. Log Telemetry for Production
```python
# Already implemented in claims_api.py
if settings.enable_telemetry:
    await persistence_store.log_audit(...)
```

### 5. Test with IntentResult First
```python
# ✅ GOOD - Standalone test
test_case = IntentResult(
    intent="find_claim",
    entities={"raw_entities": {"claim_id": "12345"}}
)
result = await call_claims_tool_node(test_case)

# Then integrate with LangGraph
```

---

## Performance Considerations

### Caching
- API repository is cached via `@lru_cache(maxsize=1)`
- Built once per application lifecycle
- No performance penalty for repeated calls

### Retry Impact
- Max 3 attempts = ~1.5s additional latency on failure
- Only retries retriable errors (timeouts, 5xx)
- Non-retriable errors fail fast

### Telemetry Overhead
- SQLite writes are async (non-blocking)
- Minimal impact on response time
- Can disable via `ENABLE_TELEMETRY=false`

### Request Size
- Typical claim request: < 1KB
- Typical claim response: 2-10KB
- No pagination required for single claims

---

## Security Considerations

### API Authentication
- Uses header-based auth (`x-correlation-id`, `x-consumerAppName`)
- No sensitive data in URL parameters
- HTTPS enforced via BASE_URL configuration

### Data Privacy
- Claims data contains PII (names, DOB, etc.)
- Logged to SQLite for audit purposes
- Ensure database files are secured
- Consider encryption at rest

### Error Handling
- Stack traces logged to database (not exposed to user)
- Error messages sanitized for user display
- Detailed errors only in logs/telemetry

---

## Maintenance & Updates

### Adding New APIs
1. Update `get_api_repository.py` with new APIEntry
2. Add entity mappings to ENTITY_MAP if needed
3. Test with `test_claims_api.py`
4. Update this documentation

### Modifying Existing APIs
1. Update APIEntry definition
2. Test backward compatibility
3. Update tests
4. Deploy with version tracking

### Monitoring
- Check telemetry database for error patterns
- Monitor `execution_time_ms` for performance degradation
- Track `retry_count` for network stability issues

---

## Appendix

### Complete ToolResult Fields

```python
{
  "tool_name": str,              # API identifier
  "status": str,                 # "success" | "failure" | "timeout" | "partial"
  "data": dict,                  # API response data
  "error_message": str | None,   # Human-readable error
  "error_code": str | None,      # Machine-readable code
  "agent_error": dict | None,    # Full AgentError object
  "execution_time_ms": float,    # Total execution time
  "api_endpoint": str,           # Full URL called
  "http_status_code": int,       # HTTP status (200, 500, etc.)
  "retry_count": int,            # Number of retries
  "is_retryable": bool,          # Can this be retried?
  "from_cache": bool,            # Was cached? (future)
  "cache_key": str | None,       # Cache key (future)
  "timestamp": str               # ISO 8601 timestamp
}
```

### Error Code Reference

| Code | Category | Description |
|------|----------|-------------|
| E1001 | Validation | Missing or invalid input |
| E3001 | API | External API error |
| E9001 | System | Internal system error |

### HTTP Status Code Handling

| Range | Retriable | Handling |
|-------|-----------|----------|
| 200-299 | N/A | Success ✅ |
| 400-499 | ❌ No | Client error (don't retry) |
| 500-599 | ✅ Yes | Server error (retry) |
| Timeout | ✅ Yes | Network issue (retry) |

---

## Document Version

- **Version**: 1.0.0
- **Date**: 2025-11-18
- **Author**: AI Agent Documentation Generator
- **Last Updated**: Initial creation

---

## Quick Links

- Source: `tools/claims_api.py`
- Tests: `tools/test_claims_api.py`
- Test Endpoint: `utils/test_endpoints.py` (line 637)
- API Registry: `tools/get_api_repository.py`
- State Schema: `state/schema.py`
- Error Models: `core/error_models.py`
- Node Models: `core/node_models.py`

---

**End of Documentation**

