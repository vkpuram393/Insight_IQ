

# Node Result Models Documentation

## Overview

This document describes the comprehensive Pydantic models for all node outputs in the LangGraph conversational agent system. These models provide structured, type-safe data contracts between nodes in your agent graph.

## Architecture

Each node in the system returns structured data using Pydantic models instead of plain dictionaries. This provides:

- **Type Safety**: Catch errors at development time
- **Validation**: Automatic validation of all fields
- **Documentation**: Self-documenting code
- **Serialization**: Easy JSON conversion for APIs
- **IDE Support**: Better autocomplete and type hints

## Model Categories

### 1. Intent Classification Models
### 2. Safety/Guardrail Models
### 3. Tool Execution Models
### 4. Context Building Models
### 5. Clarification Models
### 6. Response Generation Models
### 7. Cache Models
### 8. Confidence Check Models

---

## 1. Intent Classification Models

### `IntentResult`

The result from intent classification containing detected intent, confidence, and extracted entities.

**Fields:**

```python
intent: str                    # Detected intent name
confidence: float              # Score 0.0 to 1.0
needs_clarification: bool      # Whether more info needed
all_scores: Dict[str, float]   # All candidate intent scores
top_candidates: List[Tuple]    # Top N candidates
is_simple: bool                # Can use templates
is_complex: bool               # Needs LLM processing
entities: EntityExtractionResult  # Extracted entities
reasoning: str                 # Classification explanation
classification_method: str     # Method used
processing_time_ms: float      # Processing time
```

**Example:**

```python
from core.node_models import IntentResult, EntityExtractionResult

# Create result
result = IntentResult(
    intent="claim_status",
    confidence=0.85,
    needs_clarification=False,
    all_scores={"claim_status": 0.85, "claim_details": 0.42},
    is_simple=True,
    entities=EntityExtractionResult(claim_number="CLM-12345"),
    classification_method="keyword_matching",
    processing_time_ms=15.3
)

# Access fields with type safety
print(result.intent)  # "claim_status"
print(result.confidence)  # 0.85
print(result.entities.claim_number)  # "CLM-12345"

# Serialize to dict
result_dict = result.model_dump()

# Serialize to JSON
result_json = result.model_dump_json()
```

### `EntityExtractionResult`

Entities extracted from user input.

**Fields:**

```python
claim_number: Optional[str]
member_id: Optional[str]
prescription_number: Optional[str]
medication_name: Optional[str]
date_from: Optional[str]
date_to: Optional[str]
raw_entities: Dict[str, Any]  # Additional entities
```

**Example:**

```python
entities = EntityExtractionResult(
    claim_number="CLM-12345",
    member_id="MEM-67890",
    medication_name="Lipitor"
)
```

---

## 2. Safety/Guardrail Models

### `SafetyResult` (alias: `GuardrailResult`)

Result from safety checks (both precheck and postcheck).

**Fields:**

```python
check_type: SafetyCheckType     # PRECHECK or POSTCHECK
passed: bool                    # Whether check passed
violation_type: SafetyViolationType  # Type of violation
block_reason: str               # Why blocked
detected_keywords: List[str]    # Harmful keywords found
confidence_score: float         # Confidence in assessment
suggested_action: str           # block/warn/allow
user_message: str               # Safe message for user
processing_time_ms: float
```

**Example:**

```python
from core.node_models import (
    SafetyResult, 
    SafetyCheckType, 
    SafetyViolationType
)

# Failed safety check
result = SafetyResult(
    check_type=SafetyCheckType.PRECHECK,
    passed=False,
    violation_type=SafetyViolationType.SELF_HARM,
    block_reason="Detected self-harm content",
    detected_keywords=["suicide"],
    confidence_score=0.95,
    suggested_action="block",
    user_message="I cannot process that request."
)

# Passed safety check
result = SafetyResult(
    check_type=SafetyCheckType.PRECHECK,
    passed=True,
    violation_type=SafetyViolationType.NONE
)
```

**Enums:**

```python
class SafetyCheckType(str, Enum):
    PRECHECK = "precheck"
    POSTCHECK = "postcheck"

class SafetyViolationType(str, Enum):
    SELF_HARM = "self_harm"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    PII_EXPOSURE = "pii_exposure"
    MALICIOUS_INPUT = "malicious_input"
    EXCESSIVE_LENGTH = "excessive_length"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    NONE = "none"
```

---

## 3. Tool Execution Models

### `ToolResult`

Result from external API/tool calls.

**Fields:**

```python
tool_name: str                  # Name of tool/API
status: ToolExecutionStatus     # SUCCESS/FAILURE/TIMEOUT/PARTIAL
data: Dict[str, Any]            # Data returned
error_message: str              # Error if failed
error_code: str                 # Error code
execution_time_ms: float        # Execution time
api_endpoint: str               # API endpoint called
http_status_code: int           # HTTP status
retry_count: int                # Number of retries
is_retryable: bool              # Can retry?
from_cache: bool                # From cache?
cache_key: str                  # Cache key
```

**Example:**

```python
from core.node_models import ToolResult, ToolExecutionStatus

# Successful tool call
result = ToolResult(
    tool_name="claims_api",
    status=ToolExecutionStatus.SUCCESS,
    data={
        "claim_id": "CLM-12345",
        "status": "processing",
        "submitted_date": "2025-01-10"
    },
    execution_time_ms=245.8,
    api_endpoint="/api/v1/claims/CLM-12345",
    http_status_code=200,
    is_retryable=True,
    from_cache=False
)

# Failed tool call
result = ToolResult(
    tool_name="claims_api",
    status=ToolExecutionStatus.FAILURE,
    data={},
    error_message="Connection timeout",
    error_code="TIMEOUT",
    is_retryable=True,
    retry_count=1
)
```

---

## 4. Context Building Models

### `ContextResult`

Result from context retrieval containing conversation history and facts.

**Fields:**

```python
conversation_history: List[ConversationMessage]
relevant_facts: List[SessionFact]
history_length: int
facts_count: int
context_window_size: int
memory_source: str              # inmemory/redis/memorystore
retrieval_time_ms: float
```

**Supporting Models:**

```python
class ConversationMessage(BaseModel):
    role: str                   # user or assistant
    content: str
    timestamp: str
    metadata: Dict[str, Any]

class SessionFact(BaseModel):
    fact_type: str              # Type of fact
    data: Dict[str, Any]
    extracted_at: str
    relevance_score: float
```

**Example:**

```python
from core.node_models import (
    ContextResult, 
    ConversationMessage, 
    SessionFact
)

result = ContextResult(
    conversation_history=[
        ConversationMessage(
            role="user",
            content="What's my claim status?",
            timestamp="2025-11-10T10:29:00Z"
        ),
        ConversationMessage(
            role="assistant",
            content="Your claim is processing.",
            timestamp="2025-11-10T10:29:05Z"
        )
    ],
    relevant_facts=[
        SessionFact(
            fact_type="claim_mention",
            data={"claim_number": "CLM-12345"},
            extracted_at="2025-11-10T10:29:00Z"
        )
    ],
    history_length=2,
    facts_count=1,
    memory_source="redis",
    retrieval_time_ms=12.4
)
```

---

## 5. Clarification Models

### `ClarificationResult`

Result when system needs clarification from user.

**Fields:**

```python
needs_clarification: bool
clarifying_question: str
clarification_type: ClarificationType
original_intent: str
missing_entities: List[str]
suggested_intents: List[str]
question_template: str
confidence_score: float
expected_entity_types: List[str]
max_clarification_attempts: int
current_attempt: int
```

**Example:**

```python
from core.node_models import ClarificationResult, ClarificationType

result = ClarificationResult(
    needs_clarification=True,
    clarifying_question="Could you provide your claim number?",
    clarification_type=ClarificationType.MISSING_ENTITY,
    original_intent="claim_status",
    missing_entities=["claim_number"],
    expected_entity_types=["claim_number"],
    current_attempt=1,
    max_clarification_attempts=3
)
```

**Clarification Types:**

```python
class ClarificationType(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    MISSING_ENTITY = "missing_entity"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    INCOMPLETE_INFO = "incomplete_info"
    OUT_OF_SCOPE = "out_of_scope"
```

---

## 6. Response Generation Models

### `ResponsePayload`

Result from response generation with metadata.

**Fields:**

```python
response: str                   # Generated response text
response_type: ResponseType
response_source: ResponseSource
llm_model: str
prompt_template: str
temperature: float
input_tokens: int               # Token usage
output_tokens: int
total_tokens: int
estimated_cost_usd: float
confidence_score: float
completeness_score: float
context_used: bool
tools_used: List[str]
generation_time_ms: float
total_processing_time_ms: float
safety_checked: bool
safety_passed: bool
```

**Example:**

```python
from core.node_models import (
    ResponsePayload, 
    ResponseType, 
    ResponseSource
)

result = ResponsePayload(
    response="Your claim #12345 is being processed.",
    response_type=ResponseType.DIRECT_ANSWER,
    response_source=ResponseSource.LLM_GENERATED,
    llm_model="gemini-2.5-flash",
    temperature=0.7,
    input_tokens=234,
    output_tokens=89,
    total_tokens=323,
    estimated_cost_usd=0.000161,
    confidence_score=0.92,
    completeness_score=0.95,
    context_used=True,
    tools_used=["claims_api"],
    generation_time_ms=456.2,
    safety_checked=True,
    safety_passed=True
)
```

---

## 7. Cache Models

### `CacheResult`

Result from cache operations.

**Fields:**

```python
cache_hit: bool
status: CacheStatus
cached_response: str
cached_intent: str
cached_confidence: float
cache_key: str
cache_age_seconds: float
ttl_seconds: int
retrieval_time_ms: float
cache_backend: str
```

**Example:**

```python
from core.node_models import CacheResult, CacheStatus

# Cache hit
result = CacheResult(
    cache_hit=True,
    status=CacheStatus.HIT,
    cached_response="Your claim is processing.",
    cached_intent="claim_status",
    cached_confidence=0.89,
    cache_key="cache:a3f2b1c9d8e7f6",
    cache_age_seconds=245.3,
    ttl_seconds=3600,
    retrieval_time_ms=3.2,
    cache_backend="redis"
)

# Cache miss
result = CacheResult(
    cache_hit=False,
    status=CacheStatus.MISS
)
```

---

## 8. Confidence Check Models

### `ConfidenceCheckResult`

Result from confidence checking/routing logic.

**Fields:**

```python
decision: ConfidenceCheckDecision  # PROCEED/CLARIFY/FALLBACK
confidence: float
threshold: float
passed_threshold: bool
required_entities: List[str]
missing_entities: List[str]
entities_complete: bool
decision_reason: str
next_node: str
intent_checked: str
```

**Example:**

```python
from core.node_models import (
    ConfidenceCheckResult, 
    ConfidenceCheckDecision
)

# Need clarification
result = ConfidenceCheckResult(
    decision=ConfidenceCheckDecision.CLARIFY,
    confidence=0.62,
    threshold=0.70,
    passed_threshold=False,
    required_entities=["claim_number"],
    missing_entities=["claim_number"],
    entities_complete=False,
    decision_reason="Confidence below threshold",
    next_node="clarification",
    intent_checked="claim_status"
)
```

---

## Integration Guide

### Step 1: Update Node Functions

**Before:**

```python
async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    result = classifier.classify(state["text"])
    return {
        "intent": result["intent"],
        "confidence": result["confidence"]
    }
```

**After:**

```python
from core.node_models import IntentResult

async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    result = classifier.classify(state["text"])
    
    intent_result = IntentResult(
        intent=result["intent"],
        confidence=result["confidence"],
        classification_method="keyword_matching"
    )
    
    return {
        "intent": intent_result.intent,
        "confidence": intent_result.confidence,
        "metadata": {
            **state.get("metadata", {}),
            "intent_result": intent_result.model_dump()
        }
    }
```

### Step 2: Use in Downstream Nodes

```python
async def response_agent_node(state: AgentState) -> Dict[str, Any]:
    # Access structured intent result from metadata
    intent_result_dict = state["metadata"].get("intent_result", {})
    
    # Or use state fields directly
    intent = state["intent"]
    confidence = state["confidence"]
    
    # Generate response...
```

### Step 3: Store in State Metadata

All structured results should be stored in `state["metadata"]` for auditing, debugging, and analytics:

```python
return {
    "field1": value1,
    "field2": value2,
    "metadata": {
        **state.get("metadata", {}),
        "node_name_result": structured_result.model_dump()
    }
}
```

## Best Practices

### 1. Always Use Structured Models

```python
# ❌ Bad - unstructured dict
return {"intent": "claim_status", "conf": 0.85}

# ✅ Good - structured model
result = IntentResult(intent="claim_status", confidence=0.85)
return {"intent": result.intent, "confidence": result.confidence}
```

### 2. Store Full Results in Metadata

```python
# Store complete result for debugging
return {
    "response": result.response,
    "metadata": {
        **state.get("metadata", {}),
        "response_payload": result.model_dump()  # Full structured data
    }
}
```

### 3. Use Helper Functions

```python
from core.node_models import create_intent_result

# Quick creation with defaults
result = create_intent_result(
    intent="greeting",
    confidence=1.0
)
```

### 4. Validate at Boundaries

```python
# When receiving data from external sources
try:
    tool_result = ToolResult(
        tool_name="external_api",
        status=ToolExecutionStatus.SUCCESS,
        data=external_data
    )
except ValidationError as e:
    # Handle validation error
    logger.error(f"Invalid tool result: {e}")
```

### 5. Serialize for APIs

```python
@router.get("/node/intent")
async def get_intent_result():
    result = IntentResult(...)
    
    # Returns properly formatted JSON
    return result.model_dump()
```

## Testing

Example tests with logging (following pattern from `test_endpoints.py`):

```python
import pytest
from core.node_models import IntentResult, EntityExtractionResult
from core.logger import get_logger

logger = get_logger(__name__)

def test_intent_result_creation():
    """Test intent result creation with logging"""
    logger.info("Testing intent result creation")
    try:
        result = IntentResult(
            intent="claim_status",
            confidence=0.85
        )
        
        assert result.intent == "claim_status"
        assert 0.0 <= result.confidence <= 1.0
        
        # Test serialization
        result_dict = result.model_dump()
        logger.debug(f"Serialized result: {list(result_dict.keys())}")
        assert result_dict["intent"] == "claim_status"
        
        # Test JSON round-trip
        result_json = result.model_dump_json()
        restored = IntentResult.model_validate_json(result_json)
        assert restored.intent == result.intent
        logger.info("✅ Intent result test passed")
    except Exception as e:
        logger.error(f"Intent result test failed: {e}")
        raise

def test_complete_intent_flow():
    """Test complete intent flow with entities"""
    logger.info("Testing complete intent classification flow")
    try:
        # Create entities
        entities = EntityExtractionResult(claim_number="CLM-12345")
        logger.debug(f"Created entities: {entities.claim_number}")
        
        # Create intent result
        intent_result = IntentResult(
            intent="claim_status",
            confidence=0.89,
            entities=entities,
            is_simple=True,
            classification_method="keyword_matching"
        )
        logger.debug(f"Created intent result: {intent_result.intent}")
        
        assert intent_result.intent == "claim_status"
        assert intent_result.entities.claim_number == "CLM-12345"
        logger.info("✅ Complete intent flow test passed")
    except Exception as e:
        logger.error(f"Complete intent flow test failed: {e}")
        raise
```

### Running Tests

```bash
# Run all node model tests with verbose logging
pytest tests/test_node_models.py -v

# Run specific test class
pytest tests/test_node_models.py::TestIntentResult -v

# Run with debug logging
pytest tests/test_node_models.py -v --log-cli-level=DEBUG
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

from core.node_models import (
    IntentResult,
    ToolResult,
    ResponsePayload,
    ConversationMessage,
)

# Convert any model to dictionary
intent_dict = to_dict(intent_result)
tool_dict = to_dict(tool_result)
response_dict = to_dict(response_payload)

# Create model from dictionary (requires model class)
intent = from_dict(IntentResult, intent_dict)
tool = from_dict(ToolResult, tool_dict)

# Convert to/from JSON
intent_json = to_json(intent_result)
intent = from_json(IntentResult, intent_json)

# Copy any model with updates
new_intent = copy_model(original_intent, confidence=0.95)
new_tool = copy_model(original_tool, retry_count=1)

# Work with lists of any models
intent_dicts = to_dict_list([intent1, intent2, intent3])
intents = from_dict_list(IntentResult, intent_dicts)

# Conversation messages
message_dicts = to_dict_list(conversation_history)
messages = from_dict_list(ConversationMessage, message_dicts)
```

**Why generic helpers?**
- **One API for all models**: Same functions work with errors, intents, tools, responses, etc.
- **Type-safe**: TypeVar provides proper type inference and IDE autocomplete
- **Less code**: 7 generic functions instead of 40+ model-specific ones
- **Extensible**: New models automatically work with existing helpers

### Use Cases

1. **Storing in State**: Convert models to dicts for state metadata
2. **Caching**: Serialize models for cache storage
3. **Database**: Store model data in databases
4. **API Responses**: Convert models for JSON responses
5. **Testing**: Create test data from dictionaries
6. **Copying**: Clone models with field updates

See `docs/examples/` for complete usage examples.

## Additional Resources

- **Models**: `core/node_models.py` - Complete model definitions
- **Examples**: `docs/examples/serialization_examples.py` - Serialization usage examples
- **Examples**: `docs/examples/node_models_examples.py` - Node Models usage examples
- **Tests**: `tests/test_node_models.py` - Model tests
- **Error Models**: `docs/examples/error_models.py` - Separate error handling

## Model Summary Table

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `IntentResult` | Intent classification | intent, confidence, entities |
| `SafetyResult` | Safety checks | passed, violation_type, block_reason |
| `ToolResult` | API/tool calls | tool_name, status, data |
| `ContextResult` | Context retrieval | conversation_history, relevant_facts |
| `ClarificationResult` | Need more info | clarifying_question, missing_entities |
| `ResponsePayload` | Response generation | response, tokens, cost |
| `CacheResult` | Cache operations | cache_hit, cached_response |
| `ConfidenceCheckResult` | Routing decision | decision, next_node |

## Support

For questions:
1. Review this documentation
2. Check examples
3. Run tests: `pytest core/test_node_models.py -v`
4. Review Pydantic documentation: https://docs.pydantic.dev/

