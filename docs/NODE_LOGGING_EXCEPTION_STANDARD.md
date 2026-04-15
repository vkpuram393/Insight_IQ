# Node Logging and Exception Handling Standard

## ✅ Standard Pattern (Most Nodes Follow This)

### 1. Imports
```python
from core.logging_context import extract_logging_context, log_state_snapshot
from core.errors.models import create_internal_error, create_llm_error
from persistence import PersistenceStoreFactory
from config.config import settings
import traceback
```

### 2. Node Structure
```python
async def my_node(state: AgentState) -> Dict[str, Any]:
    node_name = "my_node"
    log_ctx = extract_logging_context(state)
    
    try:
        # Node processing logic
        result = {
            # ... node results ...
        }
        
        # Log state snapshot after successful execution
        await log_state_snapshot(state, node_name, result)
        return result
        
    except Exception as e:
        # Standard exception handling
        tb = traceback.format_exc()
        error = create_internal_error(
            error_message=f"Node failed: {str(e)}",
            stacktrace=tb,
            session_id=log_ctx["session_id"],
            node_name=node_name
        )
        
        persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        await persistence_store.log_exception(
            error_code=error.error_code.value,
            category=error.category.value,
            severity=error.severity.value,
            message=error.message,
            user_message=error.user_message,
            session_id=log_ctx["session_id"],
            request_id=log_ctx["request_id"],
            node_name=node_name,
            stacktrace=error.stacktrace,
            metadata=error.metadata,
            user_id=log_ctx["user_id"]
        )
        
        logger.error(f"🚨 Exception in {node_name}: {e}\n{tb}")
        
        result = {
            "error": error.user_message,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result
```

## 📊 Current Status by Node

### ✅ Nodes Following Standard Pattern

1. **cache.py** (check_cache_node, cache_response_node)
   - ✅ Uses extract_logging_context
   - ✅ Uses log_state_snapshot
   - ✅ Uses create_internal_error
   - ✅ Uses persistence_store.log_exception
   - ✅ Logs state snapshot in exception handler

2. **clarification.py** (clarification_node)
   - ✅ Uses extract_logging_context
   - ✅ Uses log_state_snapshot
   - ✅ Uses create_internal_error
   - ✅ Uses persistence_store.log_exception
   - ✅ Logs state snapshot in exception handler

3. **confidence.py** (confidence_checker_node)
   - ✅ Uses extract_logging_context
   - ✅ Uses log_state_snapshot
   - ✅ Uses create_internal_error
   - ✅ Uses persistence_store.log_exception
   - ✅ Logs state snapshot in exception handler

4. **context.py** (build_context_node, update_memory_node)
   - ✅ Uses extract_logging_context
   - ✅ Uses log_state_snapshot
   - ✅ Uses create_internal_error
   - ✅ Uses persistence_store.log_exception
   - ✅ Logs state snapshot in exception handler

5. **llm_judge.py** (llm_judge_node)
   - ✅ Uses extract_logging_context
   - ✅ Uses log_state_snapshot
   - ✅ Uses create_internal_error
   - ✅ Uses persistence_store.log_exception
   - ✅ Logs state snapshot in exception handler

6. **safety.py** (safety_precheck_node, response_safety_pii_precheck_node, response_safety_pii_postcheck_node)
   - ✅ Uses extract_logging_context
   - ✅ Uses log_state_snapshot
   - ✅ Uses create_internal_error
   - ✅ Uses persistence_store.log_exception
   - ✅ Logs state snapshot in exception handler

7. **orchestrator.py** (orchestrator_node)
   - ✅ Uses extract_logging_context
   - ✅ Uses log_state_snapshot
   - ✅ Uses create_orchestrator_*_error (specialized errors)
   - ✅ Uses persistence_store.log_exception
   - ✅ Logs state snapshot in exception handler

8. **agents/intent_agent.py** (intent_agent_node)
   - ✅ Uses extract_logging_context (via session_id extraction)
   - ✅ Uses log_state_snapshot
   - ✅ Uses create_internal_error or create_llm_error
   - ✅ Uses persistence_store.log_exception
   - ✅ Logs state snapshot in exception handler

9. **agents/response_agent.py** (response_agent_node)
   - ✅ Uses extract_logging_context (via session_id extraction)
   - ✅ Uses log_state_snapshot
   - ✅ Uses create_internal_error or create_llm_error
   - ✅ Uses persistence_store.log_exception
   - ✅ Logs state snapshot in exception handler

### ⚠️ Nodes with Minor Variations

1. **tools/claims_api.py** (call_claims_tool_node)
   - ⚠️ Uses different pattern - doesn't use extract_logging_context at start
   - ⚠️ Has nested exception handling (AgentError vs Exception)
   - ✅ Uses persistence_store.log_exception
   - ⚠️ Uses log_state_snapshot but with conditional check
   - **Recommendation**: Standardize to use extract_logging_context pattern

## 📋 Standard Checklist

For each node, ensure:

- [ ] **Imports**: `extract_logging_context`, `log_state_snapshot`, error creators
- [ ] **Context Extraction**: `log_ctx = extract_logging_context(state)` at start
- [ ] **Try/Except**: Wraps main logic
- [ ] **Exception Handling**:
  - [ ] `traceback.format_exc()` for stacktrace
  - [ ] `create_internal_error()` or `create_llm_error()` for error object
  - [ ] `persistence_store.log_exception()` with all required fields
  - [ ] `logger.error()` with emoji and details
  - [ ] Return error state with metadata
  - [ ] `log_state_snapshot()` in exception handler
- [ ] **Success Logging**: `log_state_snapshot()` after successful execution

## 🔍 Key Components

### extract_logging_context()
Extracts standard logging context from state:
- `session_id`
- `request_id`
- `user_id`

### log_state_snapshot()
Logs full AgentState after node execution:
- Merges state with node result
- Creates audit log entry
- Handles telemetry disabled gracefully

### create_internal_error() / create_llm_error()
Creates structured error objects with:
- `error_code`
- `category`
- `severity`
- `message` (technical)
- `user_message` (user-friendly)
- `stacktrace`
- `metadata`

### persistence_store.log_exception()
Logs exception to persistence store with:
- All error fields
- Logging context (session_id, request_id, user_id)
- Node name
- Stacktrace
- Metadata

## 🎯 Recommendations

1. **Standardize claims_api.py** to use `extract_logging_context()` pattern
2. **All nodes should use** `log_state_snapshot()` in both success and exception paths
3. **Consistent error creation**: Use `create_internal_error()` for general errors, `create_llm_error()` for LLM-specific errors
4. **Consistent logging**: All nodes should log exceptions to persistence store

## ✅ Summary

**Overall Status**: ✅ **Mostly Standardized**

- **8/9 node files** follow the standard pattern
- **1 file** (tools/claims_api.py) has minor variations but still logs exceptions
- All nodes use `persistence_store.log_exception()`
- All nodes use `log_state_snapshot()` for state logging
- Exception handling is consistent across nodes

The codebase is well-standardized for logging and exception handling!

