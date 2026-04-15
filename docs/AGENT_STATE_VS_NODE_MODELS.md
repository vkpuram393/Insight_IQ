# AgentState vs Node Models: Understanding the Difference

## Quick Answer

**AgentState** is **YOUR code** - a TypedDict you define in `state/schema.py` that flows through the LangGraph.

**Node Models** are **Pydantic models** in `core/node_models.py` - they're structured output models but **currently not used** in the actual nodes (nodes return `Dict[str, Any]` instead).

---

## 1. AgentState (TypedDict) - The Graph State

### What It Is
- **Defined by YOU** in `state/schema.py`
- A `TypedDict` (Python's type-safe dictionary)
- The **single source of truth** that flows through your LangGraph
- LangGraph uses this to manage state between nodes

### How It Works
```python
# state/schema.py - YOU define this
class AgentState(TypedDict):
    text: str
    session_id: str
    intent: Optional[str]
    confidence: Optional[float]
    # ... all fields
```

**Flow:**
1. You create initial state: `create_initial_state(text, session_id)`
2. LangGraph passes full `AgentState` to each node
3. Each node returns `Dict[str, Any]` with only fields it updates
4. LangGraph **merges** the partial update into the full state
5. Updated state flows to next node

**Example:**
```python
# Node receives FULL state
async def confidence_checker_node(state: AgentState) -> Dict[str, Any]:
    # Can read ANY field from state
    intent = state.get("intent")
    confidence = state.get("confidence")
    
    # Returns ONLY what it updates
    return {
        "needs_clarification": True,
        "clarifying_question": "..."
    }
    # LangGraph merges this into the full AgentState
```

### Key Points
- ✅ **Single object** flows through entire graph
- ✅ **Accumulates data** as it passes through nodes
- ✅ **TypedDict** = type hints but runtime is just a dict
- ✅ **LangGraph manages** the merging automatically

---

## 2. Node Models (Pydantic) - Structured Output Models

### What They Are
- **Defined by YOU** in `core/node_models.py`
- **Pydantic BaseModel** classes
- Provide **structured, validated** output contracts
- **Currently NOT used** in actual node implementations

### Current Status
```python
# core/node_models.py - These exist but aren't used
class IntentResult(BaseModel):
    intent: str
    confidence: float
    entities: EntityExtractionResult
    # ...

class ConfidenceCheckResult(BaseModel):
    decision: str
    next_node: str
    # ...
```

**Reality:**
- Nodes currently return `Dict[str, Any]` (plain dictionaries)
- Node models exist for **documentation/type hints** but aren't enforced
- Could be used for **API responses** or **future validation**

### Potential Use Cases
1. **API Response Models**: Use when returning data to external systems
2. **Type Validation**: Enforce structure at node boundaries (future enhancement)
3. **Documentation**: Self-documenting code with examples
4. **Testing**: Validate node outputs match expected structure

---

## 3. Key Differences

| Aspect | AgentState (TypedDict) | Node Models (Pydantic) |
|-------|----------------------|------------------------|
| **Type** | TypedDict | Pydantic BaseModel |
| **Purpose** | Graph state management | Structured output contracts |
| **Used By** | LangGraph framework | Your code (optional) |
| **Validation** | Type hints only | Runtime validation |
| **Current Usage** | ✅ Active (flows through graph) | ⚠️ Defined but not enforced |
| **Location** | `state/schema.py` | `core/node_models.py` |
| **Merging** | LangGraph merges partial updates | N/A (separate objects) |

---

## 4. Why This Architecture?

### AgentState (TypedDict)
- **LangGraph Requirement**: LangGraph expects a TypedDict for state
- **Efficiency**: Only changed fields need to be returned
- **Flexibility**: Easy to add new fields without breaking existing nodes
- **Simplicity**: Plain dictionaries are easy to work with

### Node Models (Pydantic)
- **Type Safety**: Catch errors at development time
- **Validation**: Ensure data structure is correct
- **Documentation**: Self-documenting with examples
- **Future-Proof**: Can be integrated later for stricter validation

---

## 5. Best Practices

### For AgentState
1. ✅ Keep it as TypedDict (LangGraph requirement)
2. ✅ Add fields as needed (backward compatible)
3. ✅ Use Optional for fields that may not exist initially
4. ✅ Document fields clearly in docstrings

### For Node Models
1. ✅ Keep them for documentation/type hints
2. ✅ Use them for API responses if needed
3. ⚠️ Consider enforcing them in nodes (future enhancement)
4. ✅ Use them in tests to validate structure

---

## 6. Example: How They Work Together

```python
# 1. AgentState flows through graph
state: AgentState = {
    "text": "What's my claim status?",
    "session_id": "session-123",
    "intent": None,  # Not set yet
    "confidence": None
}

# 2. Node processes and returns partial update
def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    # Process...
    return {
        "intent": "claim_status",
        "confidence": 0.95
    }

# 3. LangGraph merges into state
# state now has: intent="claim_status", confidence=0.95

# 4. Node Models (optional) - could validate output
def intent_agent_node_validated(state: AgentState) -> IntentResult:
    result = IntentResult(
        intent="claim_status",
        confidence=0.95,
        entities=EntityExtractionResult(...)
    )
    # Return as dict for LangGraph
    return result.model_dump()
```

---

## Summary

- **AgentState**: Your TypedDict that LangGraph uses to manage state - **actively used**
- **Node Models**: Pydantic models for structured outputs - **defined but not enforced**
- **Current Pattern**: Nodes return `Dict[str, Any]` which LangGraph merges into AgentState
- **Future**: Could enforce Node Models for stricter validation, but current approach works well

