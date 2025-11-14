# Error Models Documentation

## Overview

This document describes the comprehensive Pydantic error models for the conversational agent system. These models provide structured error handling across all components including intent classification, API calls, safety checks, and general processing.

## Core Models

### `AgentError`

The main error model containing all error information.

**Key Fields:**

```python
error_code: ErrorCode          # Specific error code (e.g., "E1001")
category: ErrorCategory        # High-level category (e.g., "validation")
severity: ErrorSeverity        # Severity level (low/medium/high/critical)
message: str                   # Technical message for developers
user_message: str              # User-friendly message safe to display
timestamp: str                 # ISO 8601 timestamp
session_id: Optional[str]      # Session identifier
is_retryable: bool             # Can this operation be retried?
```

### `ErrorResponse`

Standard API response format for errors.

**Structure:**

```python
{
    "success": false,
    "error": { ... AgentError ... },
    "session_id": "...",
    "timestamp": "2025-11-10T10:30:45.123456Z"
}
```

## Error Categories

The system includes 16 specialized error categories:

| Category | Description | Examples |
|----------|-------------|----------|
| `VALIDATION` | Input validation failures | Invalid format, missing fields |
| `AUTHENTICATION` | Auth/authorization errors | Invalid credentials, expired token |
| `INTENT_CLASSIFICATION` | Intent detection failures | Cannot determine user intent |
| `CONFIDENCE` | Low confidence issues | Confidence below threshold |
| `CLARIFICATION` | User clarification needed | Ambiguous request |
| `SAFETY` | Safety check failures | Harmful content detected |
| `API_CALL` | External API errors | Connection failures, timeouts |
| `TOOL_EXECUTION` | Tool/function call errors | Tool execution failed |
| `MEMORY` | Memory store errors | Failed to retrieve history |
| `CACHE` | Cache operation errors | Cache write failure |
| `PERSISTENCE` | Database/storage errors | Failed to save telemetry |
| `LLM` | LLM connection/response errors | LLM timeout, rate limit |
| `CONTEXT` | Context retrieval errors | Failed to load relevant facts |
| `SESSION` | Session management errors | Session not found |
| `TIMEOUT` | Request timeout errors | Processing took too long |
| `RATE_LIMIT` | Rate limiting errors | Too many requests |
| `SYSTEM` | General system errors | Configuration error |
| `UNKNOWN` | Unclassified errors | Unexpected error |

## Error Codes

Error codes follow a systematic numbering scheme:

- **1xxx**: Validation errors (E1001-E1099)
- **2xxx**: Intent & confidence errors (E2001-E2099)
- **3xxx**: Safety errors (E3001-E3099)
- **4xxx**: API & tool errors (E4001-E4099)
- **5xxx**: LLM errors (E5001-E5099)
- **6xxx**: Memory & state errors (E6001-E6099)
- **9xxx**: System errors (E9001-E9999)

### Common Error Codes

| Code | Name | Description |
|------|------|-------------|
| E1001 | `INVALID_INPUT` | Input validation failed |
| E2001 | `INTENT_DETECTION_FAILED` | Cannot detect intent |
| E2002 | `LOW_CONFIDENCE` | Confidence below threshold |
| E3001 | `SAFETY_PRECHECK_FAILED` | Input safety check failed |
| E3002 | `SAFETY_POSTCHECK_FAILED` | Output safety check failed |
| E4001 | `API_CONNECTION_ERROR` | Failed to connect to API |
| E5001 | `LLM_CONNECTION_ERROR` | Failed to connect to LLM |
| E6001 | `MEMORY_STORE_ERROR` | Memory operation failed |
| E9001 | `INTERNAL_ERROR` | Internal system error |

## Severity Levels

| Severity | Description | Action Required |
|----------|-------------|-----------------|
| `LOW` | Informational, minor issues | Monitor |
| `MEDIUM` | Handled errors, degraded functionality | Investigate |
| `HIGH` | Significant errors, action required | Fix soon |
| `CRITICAL` | System failure, immediate attention needed | Fix immediately |

## Usage Examples

### 1. Creating Errors with Helper Functions

```python
from core.errors import (
    create_validation_error,
    create_low_confidence_error,
    create_safety_error,
    create_api_error,
    create_internal_error
)

# Validation error
error = create_validation_error(
    message="Email format is invalid",
    field="email",
    value="not-an-email",
    session_id="session-123"
)

# Low confidence error
error = create_low_confidence_error(
    confidence=0.45,
    threshold=0.70,
    intent="unknown",
    session_id="session-123"
)

# API error
error = create_api_error(
    api_name="Claims API",
    error_message="Connection timeout",
    status_code=504,
    session_id="session-123",
    is_retryable=True
)
```

### 2. Creating Custom Errors

```python
from core.errors import AgentError, ErrorCode, ErrorCategory, ErrorSeverity

error = AgentError(
    error_code=ErrorCode.ENTITY_EXTRACTION_FAILED,
    category=ErrorCategory.INTENT_CLASSIFICATION,
    severity=ErrorSeverity.MEDIUM,
    message="Failed to extract claim ID from user input",
    user_message="I couldn't find a claim number in your message. Please provide your claim number.",
    session_id="session-123",
    related_intent="claim_status",
    metadata={"text": "What's my status?"},
    is_retryable=False
)
```

### 3. Using in FastAPI Endpoints

```python
from fastapi import APIRouter, Request
from core.errors import ErrorResponse, create_validation_error

@router.post("/chat")
async def chat(request: dict):
    text = request.get("text", "")
    session_id = request.get("session_id", "unknown")
    
    # Validate input
    if not text.strip():
        error = create_validation_error(
            message="Text field is required",
            field="text",
            session_id=session_id
        )
        return ErrorResponse(error=error, session_id=session_id).model_dump()
    
    # Process request...
```

### 4. Handling in Agent Nodes

```python
from core.errors import create_internal_error
import traceback

async def intent_agent(state):
    try:
        # Process intent...
        result = classify_intent(state["text"])
        return {"intent": result["intent"]}
    except Exception as e:
        error = create_internal_error(
            error_message=f"Intent classification failed: {str(e)}",
            stacktrace=traceback.format_exc(),
            session_id=state["session_id"],
            node_name="intent_agent"
        )
        # Store error in state for downstream handling
        return {"error": error.model_dump_json()}
```

### 5. Logging Errors

```python
from core.errors import AgentError, ErrorSeverity
from core.logger import get_logger
from core.telemetry import log_event
from persistence import EventType

logger = get_logger(__name__)

async def handle_error(error: AgentError):
    # Log based on severity
    if error.severity == ErrorSeverity.CRITICAL:
        logger.critical(f"Critical error: {error.model_dump_json()}")
    elif error.severity == ErrorSeverity.HIGH:
        logger.error(f"Error: {error.model_dump_json()}")
    elif error.severity == ErrorSeverity.MEDIUM:
        logger.warning(f"Warning: {error.model_dump_json()}")
    else:
        logger.info(f"Info: {error.model_dump_json()}")
    
    # Log to telemetry
    await log_event(
        EventType.ERROR_OCCURRED,
        error.session_id or "unknown",
        error.model_dump(),
        None
    )
```

## Integration with Existing Code

### Step 1: Add Exception Handlers to `main.py`

```python
from fastapi.exceptions import RequestValidationError
from core.errors_example import (
    validation_exception_handler,
    general_exception_handler
)

app = FastAPI(...)

# Add exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)
```

### Step 2: Update API Routes in `api/routes.py`

```python
from core.errors import ErrorResponse, create_validation_error, create_internal_error

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Existing logic...
        final_state = await run_graph(...)
        
        # Check for errors in state
        if final_state.get("error"):
            # Error already structured by a node
            return ErrorResponse(
                error=AgentError.model_validate_json(final_state["error"]),
                session_id=session_id
            )
        
        # Return success response...
        
    except Exception as e:
        error = create_internal_error(
            error_message=str(e),
            stacktrace=traceback.format_exc(),
            session_id=session_id
        )
        await log_event(EventType.ERROR_OCCURRED, session_id, error.model_dump(), None)
        return ErrorResponse(error=error, session_id=session_id)
```

### Step 3: Update State Schema in `state/schema.py`

```python
from typing import Optional, Dict, Any

class AgentState(TypedDict):
    # Existing fields...
    
    # Update error field to store structured error
    error: Optional[Dict[str, Any]]  # AgentError as dict
```

### Step 4: Use in Nodes

```python
# nodes/confidence.py
from core.errors import create_low_confidence_error
from core.config import settings

async def confidence_check_node(state):
    confidence = state.get("confidence", 0.0)
    
    if confidence < settings.confidence_threshold:
        error = create_low_confidence_error(
            confidence=confidence,
            threshold=settings.confidence_threshold,
            intent=state.get("intent"),
            session_id=state["session_id"]
        )
        return {
            "needs_clarification": True,
            "error": error.model_dump()
        }
    
    return state
```

## Best Practices

### 1. Always Use Appropriate Severity Levels

```python
# LOW: User input issues, informational
create_validation_error(...)

# MEDIUM: Degraded functionality, needs attention
create_low_confidence_error(...)

# HIGH: Significant errors, service impact
create_api_error(...)

# CRITICAL: System failures, immediate attention
create_internal_error(...) or create_llm_error(...)
```

### 2. Provide Helpful User Messages

```python
# ❌ Bad - technical jargon
user_message="NullPointerException in IntentClassifier.classify()"

# ✅ Good - clear, actionable
user_message="I'm not sure what you're asking. Could you rephrase that?"
```

### 3. Include Context in Metadata

```python
error = AgentError(
    ...,
    metadata={
        "confidence": 0.45,
        "threshold": 0.70,
        "input_length": 150,
        "detected_entities": ["policy_123"],
        "attempted_intent": "policy_lookup"
    }
)
```

### 4. Set `is_retryable` Correctly

```python
# Retryable: transient issues
- API timeouts
- Rate limits
- Temporary connection issues

# Not retryable: permanent issues
- Validation errors
- Safety violations
- Configuration errors
```

### 5. Log Errors Appropriately

```python
# Always log critical errors
logger.critical(f"System failure: {error.model_dump_json()}")

# Include stacktraces for debugging
error = create_internal_error(
    error_message=str(e),
    stacktrace=traceback.format_exc(),  # Include for debugging
    session_id=session_id
)

# Don't expose stacktraces to users
return ErrorResponse(error=error)  # user_message shown, stacktrace hidden
```

## Error Recovery Patterns

### Pattern 1: Retry with Backoff

```python
if error.is_retryable and error.retry_after_seconds:
    await asyncio.sleep(error.retry_after_seconds)
    # Retry operation
```

### Pattern 2: Fallback to Clarification

```python
if error.category == ErrorCategory.CONFIDENCE:
    return {
        "needs_clarification": True,
        "clarifying_question": error.user_message
    }
```

### Pattern 3: Circuit Breaker

```python
if error.category == ErrorCategory.API_CALL:
    # Track failures and open circuit after threshold
    if failure_count > threshold:
        # Stop attempting API calls temporarily
        pass
```

## Testing

Example test cases with logging (following pattern from `test_endpoints.py`):

```python
import pytest
from core.errors import AgentError, ErrorCode, ErrorSeverity, create_validation_error
from core.logger import get_logger

logger = get_logger(__name__)

def test_create_validation_error():
    """Test validation error creation with logging"""
    logger.info("Testing validation error creation")
    try:
        error = create_validation_error(
            message="Invalid email",
            field="email",
            value="bad@",
            session_id="test-123"
        )
        
        assert error.error_code == ErrorCode.INVALID_INPUT
        assert error.severity == ErrorSeverity.LOW
        assert error.session_id == "test-123"
        assert not error.is_retryable
        logger.info("✅ Validation error test passed")
    except Exception as e:
        logger.error(f"Validation error test failed: {e}")
        raise

def test_error_serialization():
    """Test error serialization with logging"""
    logger.info("Testing error serialization")
    try:
        error = AgentError(
            error_code=ErrorCode.LOW_CONFIDENCE,
            category="confidence",
            severity="medium",
            message="Test error",
            user_message="User test error"
        )
        
        # Serialize to JSON
        error_json = error.model_dump_json()
        logger.debug(f"Serialized error: {error_json[:100]}...")
        
        # Deserialize from JSON
        error_restored = AgentError.model_validate_json(error_json)
        
        assert error_restored.error_code == error.error_code
        assert error_restored.message == error.message
        logger.info("✅ Error serialization test passed")
    except Exception as e:
        logger.error(f"Error serialization test failed: {e}")
        raise
```

### Running Tests

```bash
# Run all error model tests with verbose logging
pytest tests/test_error_models.py -v

# Run specific test
pytest tests/test_error_models.py::TestAgentErrorModel::test_create_basic_error -v

# Run with debug logging
pytest tests/test_error_models.py -v --log-cli-level=DEBUG
```

## Migration Guide

### Updating Existing Error Handling

**Before:**

```python
try:
    result = await some_operation()
except Exception as e:
    logger.error(f"Error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

**After:**

```python
try:
    result = await some_operation()
except Exception as e:
    error = create_internal_error(
        error_message=str(e),
        stacktrace=traceback.format_exc(),
        session_id=session_id
    )
    logger.error(f"Error: {error.model_dump_json()}")
    await log_event(EventType.ERROR_OCCURRED, session_id, error.model_dump(), None)
    return ErrorResponse(error=error, session_id=session_id)
```

## Serialization Helpers

The project provides **generic TypeVar-based serialization helpers** in `utils/serialization.py` that work with ANY Pydantic model (errors, node results, etc.):

```python
from utils.serialization import (
    to_dict,           # Works with ANY Pydantic model
    from_dict,         # Generic model creation
    to_json,           # Generic JSON conversion
    from_json,         # Generic JSON parsing
    copy_model,        # Generic model copying
    to_dict_list,      # Generic list conversion
    from_dict_list,    # Generic list parsing
)

from core.error_models import AgentError, ErrorResponse

# Convert any model to dictionary
error_dict = to_dict(error)
response_dict = to_dict(error_response)

# Create model from dictionary (requires model class)
error = from_dict(AgentError, error_dict)
response = from_dict(ErrorResponse, response_dict)

# Convert to/from JSON
json_str = to_json(error)
error = from_json(AgentError, json_str)

# Copy any model with updates
new_error = copy_model(original_error, session_id="new-session")

# Work with lists of any models
error_dicts = to_dict_list([error1, error2, error3])
errors = from_dict_list(AgentError, error_dicts)
```

**Why generic helpers?**
- **One API for all models**: Same functions work with errors, intents, tools, responses, etc.
- **Type-safe**: TypeVar provides proper type inference and IDE autocomplete
- **Less code**: 7 generic functions instead of 40+ model-specific ones
- **Extensible**: New models automatically work with existing helpers


## Additional Resources

- **Models**: `docs/examples/error_models.py` - Complete model definitions
- **Integration**: See existing nodes for error handling patterns
- **FastAPI**: Reference FastAPI documentation for exception handling

## Support

For questions or issues with error models:
1. Check this documentation
2. Review example code in `docs/examples/errors_example.py`
3. Look at existing node implementations
4. Contact the development team

