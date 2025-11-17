# Orchestrator Node - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Input/Output Specification](#inputoutput-specification)
4. [Normalization Process](#normalization-process)
5. [Error Handling](#error-handling)
6. [Telemetry Integration](#telemetry-integration)
7. [Integration Points](#integration-points)
8. [Code Examples](#code-examples)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)
12. [User Stories Completion](#user-stories-completion)

---

## Overview

The **Orchestrator Node** is the **initial entry point** for all user inputs in the multi-agent workflow. It performs text normalization and comprehensive error handling before delegating to downstream nodes.

### Key Features

- ✅ **Entry Point**: First node in the workflow
- ✅ **Pure Processing**: No LLM calls (pure text processing)
- ✅ **6-Step Normalization**: Comprehensive text cleaning pipeline
- ✅ **Structured Error Handling**: Pydantic-based error models
- ✅ **Telemetry Integration**: Automatic error logging
- ✅ **Backward Compatible**: No state schema changes required

### Node Classification

| Attribute | Value |
|-----------|-------|
| **Type** | Pure Processing Node |
| **LLM Calls** | None |
| **Position** | Entry Point (First Node) |
| **Async** | Yes |
| **Error Handling** | Graceful Fallback |

---

## Architecture

### Workflow Position

```
User Request → API Routes → Orchestrator → Safety Precheck → Cache → Context → Intent Agent → ...
```

### System Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     User Input (API)                        │
│                   text: "What's my claim?"                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  ORCHESTRATOR NODE                          │
│  ┌───────────────────────────────────────────────────┐     │
│  │  1. Input Validation                              │     │
│  │     - Empty check                                 │     │
│  │     - Type check                                  │     │
│  └───────────────────────────────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│  ┌───────────────────────────────────────────────────┐     │
│  │  2. Normalization Pipeline                        │     │
│  │     - Lowercase                                   │     │
│  │     - Collapse spaces                             │     │
│  │     - Unicode normalization (NFD)                 │     │
│  │     - Remove zero-width characters                │     │
│  │     - Remove punctuation (configurable)           │     │
│  │     - Strip whitespace (FINAL)                    │     │
│  └───────────────────────────────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│  ┌───────────────────────────────────────────────────┐     │
│  │  3. Create OrchestratorResult                     │     │
│  │     - Structured Pydantic model                   │     │
│  │     - Normalized & original text                  │     │
│  │     - Metadata & statistics                       │     │
│  │     - Error information (if any)                  │     │
│  └───────────────────────────────────────────────────┘     │
│                          │                                  │
│                          ▼                                  │
│  ┌───────────────────────────────────────────────────┐     │
│  │  4. Error Handling (if exception)                 │     │
│  │     - Create AgentError                           │     │
│  │     - Log to telemetry                            │     │
│  │     - Graceful fallback                           │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   State Update                              │
│  text: "whats my claim"                                     │
│  metadata:                                                  │
│    orchestrator_metadata: <OrchestratorResult>             │
│    original_text: "What's my claim?"                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
                   Safety Precheck Node
```

---

## Input/Output Specification

### Input (from AgentState)

```python
{
    "text": str,              # User's raw input - REQUIRED
    "session_id": str,        # Session identifier - REQUIRED
    "user_info": Dict,        # Optional user metadata
    "metadata": Dict          # Optional existing metadata
}
```

**Field Details:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `str` | Yes | User's raw input message |
| `session_id` | `str` | Yes | Unique session identifier |
| `user_info` | `Dict[str, Any]` | No | User metadata (contains `user_id` if available) |
| `metadata` | `Dict[str, Any]` | No | Existing metadata from previous processing |

### Output (to AgentState)

```python
{
    "text": str,              # Normalized text - REQUIRED for safety_precheck
    "error": Optional[str],   # Error code (e.g., "E1001") - backward compatible
    "metadata": {
        "orchestrator_metadata": {
            # Full OrchestratorResult serialized as dict
            "normalized_text": str,
            "original_text": str,
            "normalization_applied": bool,
            "original_length": int,
            "normalized_length": int,
            "chars_removed": int,
            "normalization_steps": List[str],
            "error": Optional[Dict],  # Serialized AgentError if error occurred
            "processing_time_ms": Optional[float],
            "timestamp": str
        },
        "original_text": str  # Preserved for logging/display
    }
}
```

**Output Guarantees:**

1. ✅ `text` field is **always a string** (required by safety_precheck_node)
2. ✅ State schema **unchanged** (no breaking changes)
3. ✅ Full structured result in `metadata["orchestrator_metadata"]`
4. ✅ Backward compatible `error` field for legacy code
5. ✅ Graceful fallback on errors (never breaks workflow)

---

## Normalization Process

### Pipeline Steps

The orchestrator applies the following transformations in order:

#### 1. Lowercase Conversion
**Purpose**: Normalize case for consistent processing  
**Example**: `"Hello World"` → `"hello world"`

#### 2. Collapse Spaces
**Purpose**: Replace multiple whitespace characters with single space  
**Example**: `"hello    world"` → `"hello world"`  
**Handles**: Tabs, newlines, and other whitespace

#### 3. Unicode Normalization (NFD)
**Purpose**: Canonical decomposition for consistent Unicode representation  
**Standard**: W3C/Unicode Consortium NFD  
**Example**: `"café"` (é as single char) → `"café"` (e + combining accent)

#### 4. Remove Zero-Width Characters
**Purpose**: Security measure to prevent spoofing attacks  
**Removes**:
- Zero-width space (`\u200b`)
- Zero-width non-joiner (`\u200c`)
- Zero-width joiner (`\u200d`)
- Zero-width no-break space (`\ufeff`)

#### 5. Remove Punctuation (Configurable)
**Purpose**: Clean text for better cache hit rates  
**Configurable**: Via `settings.remove_punctuation_in_normalization`  
**Example**: `"What's my status?"` → `"whats my status"`

#### 6. Strip Whitespace (FINAL STEP)
**Purpose**: Remove leading and trailing whitespace after all transformations  
**Example**: `"  hello  "` → `"hello"`  
**Important**: This is the final step to ensure no trailing spaces remain after punctuation removal

### Complete Example

```python
Input:  "  What's   my claim STATUS?  "
Step 1: "  what's   my claim status?  "  # Lowercase
Step 2: "  what's my claim status?  "    # Collapse spaces
Step 3: "  what's my claim status?  "    # Unicode NFD
Step 4: "  what's my claim status?  "    # Remove zero-width
Step 5: "  whats my claim status  "      # Remove punctuation
Step 6: "whats my claim status"          # Strip whitespace (FINAL)
```

**Statistics Tracked:**
- Original length: 30 characters
- Normalized length: 21 characters
- Characters removed: 9

---

## Error Handling

### Error Types and Mapping

The orchestrator uses structured error models from `core/error_models.py`:

| Error Scenario | Error Code | Category | Severity | Retryable |
|----------------|------------|----------|----------|-----------|
| Empty Input | `E1001` (INVALID_INPUT) | VALIDATION | LOW | No |
| Invalid Type | `E1003` (INVALID_FORMAT) | VALIDATION | LOW | No |
| Normalization Failure | `E9001` (INTERNAL_ERROR) | SYSTEM | MEDIUM | Yes |

### Error Models

#### 1. Empty Input Error

```python
from core.error_models import create_orchestrator_empty_input_error

error = create_orchestrator_empty_input_error(
    session_id="550e8400-e29b-41d4-a716-446655440000",
    user_id="user123"
)

# Returns AgentError with:
# - error_code: ErrorCode.INVALID_INPUT (E1001)
# - category: ErrorCategory.VALIDATION
# - severity: ErrorSeverity.LOW
# - message: "Orchestrator received empty input text"
# - user_message: "Please provide a message to continue."
# - is_retryable: False
```

#### 2. Invalid Type Error

```python
from core.error_models import create_orchestrator_invalid_type_error

error = create_orchestrator_invalid_type_error(
    input_type=int,
    session_id="550e8400-e29b-41d4-a716-446655440000",
    user_id="user123"
)

# Returns AgentError with:
# - error_code: ErrorCode.INVALID_FORMAT (E1003)
# - user_message: "I received an invalid input format. Please try again."
```

#### 3. Normalization Failure Error

```python
from core.error_models import create_orchestrator_normalization_error

error = create_orchestrator_normalization_error(
    exception=ValueError("Unicode error"),
    session_id="550e8400-e29b-41d4-a716-446655440000",
    user_id="user123",
    stacktrace=traceback.format_exc()
)

# Returns AgentError with:
# - error_code: ErrorCode.INTERNAL_ERROR (E9001)
# - severity: ErrorSeverity.MEDIUM
# - is_retryable: True
# - stacktrace: Full Python traceback for debugging
```

### Error Flow

```
Error Occurs → Create AgentError → Log to Telemetry → Create OrchestratorResult (with error) → Return State
```

### Graceful Fallback Strategy

The orchestrator **never breaks the workflow**. On any error:

1. ✅ Create structured `AgentError` object
2. ✅ Log error to telemetry (`EventType.ERROR_OCCURRED`)
3. ✅ Create `OrchestratorResult` with error information
4. ✅ Return fallback text (original input)
5. ✅ Workflow continues to next node

**Example Fallback Behavior:**

```python
# Input: 123 (invalid type)
# Orchestrator action:
# 1. Creates InvalidTypeError
# 2. Logs to telemetry
# 3. Converts to string: "123"
# 4. Continues normalization with "123"
```

---

## Telemetry Integration

### Events Logged

The orchestrator logs errors to the telemetry system using `EventType.ERROR_OCCURRED`:

```python
from core.telemetry import log_event
from persistence import EventType

await log_event(
    EventType.ERROR_OCCURRED,
    session_id,
    {"error": error.model_dump(mode="json")},
    user_id
)
```

### Event Data Structure

```json
{
  "event_type": "error_occurred",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "data": {
    "error": {
      "error_code": "E1001",
      "category": "validation",
      "severity": "low",
      "message": "Orchestrator received empty input text",
      "user_message": "Please provide a message to continue.",
      "timestamp": "2025-01-15T10:30:45.123456Z",
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "node_name": "orchestrator",
      "is_retryable": false,
      "metadata": {
        "error_type": "empty_input"
      }
    }
  },
  "timestamp": "2025-01-15T10:30:45.123456Z"
}
```

### Analytics Queries

Telemetry data can be queried for analytics:

```python
from core.telemetry import get_analytics

# Get orchestrator error rate
analytics = await get_analytics()
orchestrator_errors = [
    event for event in analytics['events']
    if event['type'] == 'error_occurred'
    and event['data'].get('error', {}).get('node_name') == 'orchestrator'
]
```

---

## Integration Points

### Downstream Nodes

#### 1. Safety Precheck Node

**Location**: `nodes/safety.py`  
**Expects**: `state["text"]` as string  
**Compatibility**: ✅ **Fully Compatible**

```python
# In safety_precheck_node:
text = state["text"].lower()  # Expects string
# Orchestrator always provides string in state["text"]
```

#### 2. Cache Node

**Location**: `nodes/cache.py`  
**Uses**: Normalized text for cache key generation  
**Compatibility**: ✅ **Fully Compatible**

```python
# In check_cache_node:
key = f"cache:{_hash(state['text'])}"
# Uses normalized text for better cache hit rates
```

#### 3. Context & Memory Node

**Location**: `nodes/context.py`  
**Uses**: Original text preserved in metadata  
**Compatibility**: ✅ **Fully Compatible**

```python
# Can access original text if needed:
original_text = state.get("metadata", {}).get("original_text")
```

### Upstream Components

#### 1. API Routes

**Location**: `api/routes.py`  
**Integration**: Creates initial state and calls `run_graph()`  
**Compatibility**: ✅ **No Changes Required**

```python
# In chat endpoint:
final_state = await run_graph(
    text=request.text,
    session_id=session_id,
    user_info=request.user_info or {}
)
# Orchestrator is automatically called as entry point
```

#### 2. LangGraph Workflow

**Location**: `langgraph_agent.py`  
**Integration**: Orchestrator set as entry point  
**Compatibility**: ✅ **Fully Integrated**

```python
# In _build_workflow():
workflow.add_node("orchestrator", orchestrator_node)
workflow.set_entry_point("orchestrator")
workflow.add_edge("orchestrator", "safety_precheck")
```

#### 3. State Schema

**Location**: `state/schema.py`  
**Changes**: ✅ **NONE** (all data stored in existing `metadata` field)  
**Compatibility**: ✅ **100% Backward Compatible**

```python
class AgentState(TypedDict):
    text: str                    # ✅ Orchestrator provides this
    metadata: Dict[str, Any]     # ✅ Orchestrator uses this
    error: Optional[str]         # ✅ Orchestrator provides error codes
    # No new fields added!
```

---

## Code Examples

### 1. Creating Orchestrator Result (Success Case)

```python
from core.node_models import create_orchestrator_result
from datetime import datetime, timezone

# Create successful normalization result
result = create_orchestrator_result(
    normalized_text="whats my claim status",
    original_text="What's my claim status?",
    normalization_applied=True,
    original_length=25,
    normalized_length=22,
    chars_removed=3,
    normalization_steps=[
        "strip_whitespace",
        "lowercase",
        "collapse_spaces",
        "unicode_nfd",
        "remove_zero_width",
        "remove_punctuation"
    ],
    timestamp=datetime.now(timezone.utc).isoformat()
)

# Serialize for state
result_dict = result.model_dump(mode="json")

# Return in state update
return {
    "text": result.normalized_text,
    "metadata": {
        "orchestrator_metadata": result_dict,
        "original_text": result.original_text
    }
}
```

### 2. Handling Empty Input Error

```python
from core.error_models import create_orchestrator_empty_input_error
from core.telemetry import log_event
from persistence import EventType

# Detect empty input
if not raw_text:
    # Create structured error
    user_id = state.get("user_info", {}).get("user_id")
    error = create_orchestrator_empty_input_error(
        session_id=session_id,
        user_id=user_id
    )
    
    # Log to telemetry
    await log_event(
        EventType.ERROR_OCCURRED,
        session_id,
        {"error": error.model_dump(mode="json")},
        user_id
    )
    
    # Create result with error
    result = create_orchestrator_result(
        normalized_text="",
        original_text="",
        normalization_applied=False,
        original_length=0,
        normalized_length=0,
        chars_removed=0,
        error=error.model_dump(mode="json")
    )
    
    return {
        "text": "",
        "error": error.error_code.value,  # "E1001"
        "metadata": {
            "orchestrator_metadata": result.model_dump(mode="json"),
            "original_text": ""
        }
    }
```

### 3. Accessing Orchestrator Results from Downstream Nodes

```python
# In any downstream node:
async def my_node(state: AgentState) -> Dict[str, Any]:
    # Access orchestrator metadata
    orchestrator_metadata = state.get("metadata", {}).get("orchestrator_metadata", {})
    
    # Get specific fields
    normalized_text = orchestrator_metadata.get("normalized_text")
    original_text = orchestrator_metadata.get("original_text")
    normalization_applied = orchestrator_metadata.get("normalization_applied")
    error = orchestrator_metadata.get("error")  # None if no error
    
    # Check if orchestrator had an error
    if error:
        logger.warning(f"Orchestrator reported error: {error['error_code']}")
    
    # Reconstruct full OrchestratorResult model if needed
    if orchestrator_metadata:
        from core.node_models import OrchestratorResult
        result = OrchestratorResult(**orchestrator_metadata)
        logger.info(f"Chars removed: {result.chars_removed}")
```

### 4. Exception Handling Example

```python
try:
    # Normalization logic
    normalized = raw_text.strip().lower()
    # ... more steps ...
except Exception as e:
    # Create structured error
    error = create_orchestrator_normalization_error(
        exception=e,
        session_id=session_id,
        user_id=user_id,
        stacktrace=traceback.format_exc()
    )
    
    # Log to telemetry
    await log_event(
        EventType.ERROR_OCCURRED,
        session_id,
        {"error": error.model_dump(mode="json")},
        user_id
    )
    
    # Graceful fallback
    fallback_text = str(raw_text) if raw_text else ""
    
    result = create_orchestrator_result(
        normalized_text=fallback_text,
        original_text=fallback_text,
        normalization_applied=False,
        original_length=len(fallback_text),
        normalized_length=len(fallback_text),
        chars_removed=0,
        error=error.model_dump(mode="json")
    )
    
    return {
        "text": fallback_text,
        "error": error.error_code.value,
        "metadata": {
            "orchestrator_metadata": result.model_dump(mode="json"),
            "original_text": fallback_text
        }
    }
```

---

## Configuration

### Environment Variables

The orchestrator behavior can be configured via `core/config.py`:

```python
# settings.py
class Settings:
    # Orchestrator-specific settings
    remove_punctuation_in_normalization: bool = True
    enable_telemetry: bool = True
    
    # Related settings
    enable_safety_precheck: bool = True
    enable_semantic_cache: bool = True
```

### Configuration Examples

#### Disable Punctuation Removal

```python
# .env file
REMOVE_PUNCTUATION_IN_NORMALIZATION=false
```

**Effect**: Input like `"What's my status?"` becomes `"what's my status?"` instead of `"whats my status"`

#### Disable Telemetry

```python
# .env file
ENABLE_TELEMETRY=false
```

**Effect**: Errors are logged locally but not sent to persistence store

---

## Testing

### Unit Test Cases

#### Test 1: Empty Input

```python
import pytest
from nodes.orchestrator import orchestrator_node
from state.schema import create_initial_state

@pytest.mark.asyncio
async def test_orchestrator_empty_input():
    state = create_initial_state(
        text="",
        session_id="test-session-123",
        user_info={}
    )
    
    result = await orchestrator_node(state)
    
    # Assertions
    assert result["text"] == ""
    assert result["error"] == "E1001"
    assert result["metadata"]["orchestrator_metadata"]["normalization_applied"] == False
    assert result["metadata"]["orchestrator_metadata"]["error"] is not None
    assert result["metadata"]["orchestrator_metadata"]["error"]["error_code"] == "E1001"
```

#### Test 2: Invalid Type Input

```python
@pytest.mark.asyncio
async def test_orchestrator_invalid_type():
    state = create_initial_state(
        text=123,  # Invalid: should be string
        session_id="test-session-123",
        user_info={}
    )
    
    result = await orchestrator_node(state)
    
    # Should convert and continue (graceful fallback)
    assert isinstance(result["text"], str)
    assert result["text"] == "123"
```

#### Test 3: Normal Success

```python
@pytest.mark.asyncio
async def test_orchestrator_normal_success():
    state = create_initial_state(
        text="  What's my claim STATUS?  ",
        session_id="test-session-123",
        user_info={}
    )
    
    result = await orchestrator_node(state)
    
    # Assertions
    assert result["text"] == "whats my claim status"
    assert "error" not in result or result["error"] is None
    assert result["metadata"]["orchestrator_metadata"]["normalization_applied"] == True
    assert result["metadata"]["orchestrator_metadata"]["original_text"] == "  What's my claim STATUS?  "
    assert result["metadata"]["orchestrator_metadata"]["chars_removed"] > 0
```

#### Test 4: Normalization Failure

```python
@pytest.mark.asyncio
async def test_orchestrator_normalization_failure(monkeypatch):
    # Mock unicodedata.normalize to raise exception
    def mock_normalize(*args):
        raise ValueError("Unicode error")
    
    monkeypatch.setattr("unicodedata.normalize", mock_normalize)
    
    state = create_initial_state(
        text="test input",
        session_id="test-session-123",
        user_info={}
    )
    
    result = await orchestrator_node(state)
    
    # Should fall back gracefully
    assert result["text"] == "test input"
    assert result["error"] == "E9001"
    assert result["metadata"]["orchestrator_metadata"]["error"] is not None
```

### Integration Test

```python
@pytest.mark.asyncio
async def test_orchestrator_to_safety_integration():
    """Test that orchestrator output is compatible with safety_precheck"""
    from nodes.orchestrator import orchestrator_node
    from nodes.safety import safety_precheck_node
    
    # Create initial state
    state = create_initial_state(
        text="Hello world",
        session_id="test-session",
        user_info={}
    )
    
    # Run orchestrator
    state_after_orchestrator = await orchestrator_node(state)
    
    # Update state (simulate LangGraph behavior)
    state.update(state_after_orchestrator)
    
    # Run safety check
    state_after_safety = await safety_precheck_node(state)
    
    # Should pass without errors
    assert state_after_safety["safety_precheck_passed"] == True
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Safety Node Fails

**Symptom**: Safety precheck node throws `AttributeError: 'NoneType' object has no attribute 'lower'`

**Root Cause**: Orchestrator did not return `text` field as string

**Solution**: Verify orchestrator always returns `text` as string:

```python
# Check orchestrator output
result = await orchestrator_node(state)
assert "text" in result
assert isinstance(result["text"], str)
```

#### Issue 2: Metadata Not Found

**Symptom**: `KeyError: 'orchestrator_metadata'` in downstream nodes

**Root Cause**: Trying to access metadata before orchestrator has run

**Solution**: Always check for metadata existence:

```python
orchestrator_metadata = state.get("metadata", {}).get("orchestrator_metadata", {})
if not orchestrator_metadata:
    logger.warning("Orchestrator metadata not found")
```

#### Issue 3: Errors Not Logged

**Symptom**: Errors occur but don't appear in telemetry

**Root Cause**: Telemetry disabled or persistence store issue

**Solution**: Check configuration:

```python
# Verify telemetry is enabled
from core.config import settings
assert settings.enable_telemetry == True

# Check persistence store connection
from core.telemetry import get_persistence_store
store = get_persistence_store()
# Should not raise exception
```

#### Issue 4: Pydantic ValidationError

**Symptom**: `ValidationError` when creating `OrchestratorResult`

**Root Cause**: Missing required fields or invalid types

**Solution**: Use helper function and verify all required fields:

```python
from core.node_models import create_orchestrator_result

# Always provide required fields
result = create_orchestrator_result(
    normalized_text="text",      # Required
    original_text="text",        # Required
    normalization_applied=True,  # Required (has default)
    original_length=4,           # Required
    normalized_length=4          # Required
)
```

### Debug Checklist

- [ ] Check `state["text"]` is always a string
- [ ] Verify `metadata["orchestrator_metadata"]` exists
- [ ] Confirm telemetry is enabled in settings
- [ ] Check error codes match `ErrorCode` enum values
- [ ] Verify Pydantic models serialize correctly with `model_dump(mode="json")`
- [ ] Ensure all imports are correct
- [ ] Check session_id is provided in state

---

## User Stories Completion

This section verifies that all orchestrator-related user stories are complete.

### User Story Status

| ID | User Story | Status | Implementation |
|----|------------|--------|----------------|
| **T-1.1** | Orchestrator Agent to be the initial entry point after receiving user input so that it can kick off the multi-agent workflow | ✅ **COMPLETE** | `langgraph_agent.py:80` - `workflow.set_entry_point("orchestrator")` |
| **T-1.2.1** | Orchestrator Agent - Retrieve Cache: Check semantic cache retrieval before deeper processing | ✅ **COMPLETE** | `langgraph_agent.py:87` - Cache check happens after orchestrator via `check_cache_node` |
| **T-1.2.2** | Orchestrator agent - Routing Strategy: Route initial requests to safety pre-check | ✅ **COMPLETE** | `langgraph_agent.py:82` - `workflow.add_edge("orchestrator", "safety_precheck")` |
| **T-1.3.1** | Routing: Orchestrator Agent to be able to delegate the initial user query to the "Context and Memory Agent" so that context is retrieved from the memory | ✅ **COMPLETE** | `langgraph_agent.py:89` - Context building happens after cache miss via `build_context_node` |
| **T-1.3.2** | Integrate error mapping for orchestrator failures | ✅ **COMPLETE** | `nodes/orchestrator.py` - Structured errors with telemetry integration |

### Implementation Evidence

#### T-1.1: Entry Point ✅

**File**: `langgraph_agent.py`

```python
def _build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("orchestrator", orchestrator_node)
    
    # Set orchestrator as entry point
    workflow.set_entry_point("orchestrator")  # ✅ IMPLEMENTED
    
    # Connect orchestrator to safety_precheck
    workflow.add_edge("orchestrator", "safety_precheck")
```

#### T-1.2.1: Cache Retrieval ✅

**File**: `langgraph_agent.py`

```python
# Workflow definition
workflow.add_node("safety_precheck", safety_precheck_node)
workflow.add_node("check_cache", check_cache_node)  # ✅ Cache node exists

# Routing after safety precheck
workflow.add_conditional_edges(
    "safety_precheck", 
    should_continue_after_precheck, 
    {"check_cache": "check_cache", END: END}
)  # ✅ Routes to cache check
```

#### T-1.2.2: Safety Pre-check Routing ✅

**File**: `langgraph_agent.py`

```python
# Direct edge from orchestrator to safety_precheck
workflow.add_edge("orchestrator", "safety_precheck")  # ✅ IMPLEMENTED
```

**Compatibility**: Orchestrator always returns `text` as string, which safety_precheck expects.

#### T-1.3.1: Context & Memory Routing ✅

**File**: `langgraph_agent.py`

```python
# After cache check, route to context building
workflow.add_conditional_edges(
    "check_cache", 
    should_continue_after_cache, 
    {"build_context": "build_context", END: END}
)  # ✅ Routes to context node

workflow.add_edge("build_context", "intent_agent")  # ✅ Context feeds into intent
```

#### T-1.3.2: Error Mapping Integration ✅

**Implementation Details**:

1. **Structured Error Models** (`core/error_models.py`):
   ```python
   # Three orchestrator-specific error helpers
   - create_orchestrator_empty_input_error()      # E1001
   - create_orchestrator_invalid_type_error()     # E1003
   - create_orchestrator_normalization_error()    # E9001
   ```

2. **Structured Output Model** (`core/node_models.py`):
   ```python
   class OrchestratorResult(BaseModel):
       normalized_text: str
       original_text: str
       normalization_applied: bool
       error: Optional[Dict[str, Any]]  # Serialized AgentError
       # ... more fields
   ```

3. **Telemetry Integration** (`nodes/orchestrator.py`):
   ```python
   # On every error:
   await log_event(
       EventType.ERROR_OCCURRED,
       session_id,
       {"error": error.model_dump(mode="json")},
       user_id
   )
   ```

4. **Backward Compatibility**:
   - No changes to `AgentState` schema
   - All structured data serialized to `metadata`
   - Error code stored in backward-compatible `error` field

### Verification Matrix

| Component | Status | Evidence |
|-----------|--------|----------|
| Entry Point | ✅ | `workflow.set_entry_point("orchestrator")` |
| Cache Integration | ✅ | Routes to `check_cache_node` after safety |
| Safety Routing | ✅ | Direct edge to `safety_precheck_node` |
| Context Routing | ✅ | Routes to `build_context_node` after cache miss |
| Error Models | ✅ | `AgentError`, `create_orchestrator_*_error()` |
| Output Models | ✅ | `OrchestratorResult`, `create_orchestrator_result()` |
| Telemetry | ✅ | `log_event(EventType.ERROR_OCCURRED, ...)` |
| Compatibility | ✅ | No state schema changes, full backward compatibility |

---

## Related Files

### Core Files

- **`nodes/orchestrator.py`** - Main orchestrator implementation
- **`core/node_models.py`** - OrchestratorResult Pydantic model
- **`core/error_models.py`** - AgentError and helper functions
- **`core/telemetry.py`** - Telemetry logging functions
- **`persistence/__init__.py`** - EventType enum definition

### Integration Files

- **`langgraph_agent.py`** - Workflow definition (entry point)
- **`state/schema.py`** - State schema (unchanged)
- **`api/routes.py`** - API endpoint that calls workflow
- **`nodes/safety.py`** - Next node in workflow
- **`nodes/cache.py`** - Cache check after safety
- **`nodes/context.py`** - Context building after cache

### Documentation Files

- **`docs/NODE_MODELS_README.md`** - Node models documentation
- **`docs/ERROR_MODELS_README.md`** - Error models documentation
- **`ORCHESTRATOR_README.md`** - Original orchestrator docs
- **`README.md`** - Main project documentation

---

## Appendix

### A. Serialization Pattern

Pydantic models are serialized using `model_dump(mode="json")`:

```python
# Create model
result = OrchestratorResult(...)

# Serialize to dict (JSON-safe)
result_dict = result.model_dump(mode="json")

# Store in state
state["metadata"]["orchestrator_metadata"] = result_dict

# Reconstruct later (if needed)
result_reconstructed = OrchestratorResult(**result_dict)
```

**Why `mode="json"`?**
- Handles enums → string values
- Handles datetime → ISO format strings
- Ensures all values are JSON-serializable
- Compatible with LangGraph state persistence

### B. Backward Compatibility Strategy

The implementation maintains 100% backward compatibility:

1. **No State Schema Changes**: All data stored in existing `metadata` field
2. **Error Field Preserved**: `state["error"]` still contains error code string
3. **Text Field Guaranteed**: `state["text"]` always provided as string
4. **Graceful Fallback**: Errors never break workflow
5. **Optional Telemetry**: Can be disabled without affecting functionality

### C. Performance Considerations

| Operation | Typical Time | Notes |
|-----------|-------------|-------|
| Input validation | < 1ms | String checks only |
| Normalization | 1-5ms | Depends on text length |
| Model creation | < 1ms | Pydantic instantiation |
| Serialization | < 1ms | `model_dump()` call |
| Telemetry logging | 5-20ms | Async, non-blocking |
| **Total** | **~10-30ms** | Per request |

**Optimization Tips**:
- Normalization is pure Python (no LLM overhead)
- Telemetry is async (doesn't block workflow)
- Caching benefits from normalized text

### D. Migration Guide

If upgrading from previous orchestrator version:

**No migration required!** The new implementation is 100% backward compatible.

**Optional: Access structured results**

```python
# Old way (still works):
error = state.get("error")

# New way (more information):
orchestrator_metadata = state["metadata"]["orchestrator_metadata"]
error_details = orchestrator_metadata.get("error")
if error_details:
    print(f"Error code: {error_details['error_code']}")
    print(f"User message: {error_details['user_message']}")
```

---

## Summary

The orchestrator node provides:

- ✅ **Entry point** for all user inputs
- ✅ **6-step normalization** pipeline
- ✅ **Structured error handling** with Pydantic models
- ✅ **Telemetry integration** for all errors
- ✅ **Backward compatibility** (no state schema changes)
- ✅ **Graceful fallback** (never breaks workflow)
- ✅ **Complete user story** implementation

**All user stories are complete and verified.**

---

