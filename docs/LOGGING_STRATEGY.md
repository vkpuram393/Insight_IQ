# Logging Strategy: AgentState vs Node Input/Output

## 🎯 Your Questions

1. **Should you log every node input/output (current) or just AgentState?**
2. **Is current logging overkill?**
3. **Is there a way to know which node updated the state?**

---

## 📊 Current Logging Approach

### What You're Currently Logging

Looking at your code, you're logging:
- **Node-specific events** (e.g., `confidence_check_decision`, `context_builder_input`)
- **Partial data** (only what each node processes)
- **Multiple log entries per node** (input, output, decisions)

**Example from `confidence_checker_node`:**
```python
# Logs confidence check decision
await persistence_store.log_audit(
    node_name="confidence_checker",
    event_type="confidence_check_decision",
    data={"decision": "proceed", "confidence": 0.95}
)

# Logs context builder input
await persistence_store.log_audit(
    node_name="confidence_checker",
    event_type="context_builder_input",
    data={...full input object...}
)
```

**Example from `build_context_node`:**
```python
# Logs context builder output summary
await persistence_store.log_audit(
    node_name="build_context",
    event_type="context_builder_output",
    data={"history_length": 5, "facts_count": 2}
)

# Logs full planner context
await persistence_store.log_audit(
    node_name="build_context",
    event_type="planner_context",
    data={...full planner_context object...}
)
```

---

## ✅ Recommended Approach: Log AgentState at Each Node

### Why Log AgentState Instead?

**Advantages:**
1. ✅ **Complete picture** - See full state at each step
2. ✅ **Single source of truth** - AgentState contains everything
3. ✅ **Easier debugging** - See exactly what each node received and produced
4. ✅ **Simpler queries** - One log entry per node = easier to trace
5. ✅ **Less duplication** - No need to log input/output separately
6. ✅ **Automatic tracking** - Know which node updated which fields

**Current approach issues:**
- ❌ Multiple log entries per node (input, output, decisions)
- ❌ Partial data (only what node processes)
- ❌ Harder to reconstruct full state
- ❌ More complex queries needed

---

## 🎯 Recommended Logging Pattern

### Option 1: Log Full AgentState After Each Node (Recommended)

```python
# In each node, at the END, log the full AgentState
async def confidence_checker_node(state: AgentState) -> Dict[str, Any]:
    log_ctx = extract_logging_context(state)
    
    try:
        # ... node processing ...
        
        result = {
            "needs_clarification": True,
            "clarifying_question": "..."
        }
        
        # Log FULL AgentState after this node's update
        # (LangGraph will merge result into state, but we log before that)
        updated_state = {**state, **result}  # Simulate merge
        
        await persistence_store.log_audit(
            session_id=log_ctx["session_id"],
            request_id=log_ctx["request_id"],
            user_id=log_ctx["user_id"],
            node_name="confidence_checker",
            event_type="state_snapshot",  # Standard event type
            data=updated_state  # Full AgentState
        )
        
        return result
    except Exception as e:
        # ... exception handling ...
```

### Option 2: Use LangGraph Checkpointing (Best for Full State)

LangGraph's checkpointing already saves full state! You can:
1. Use checkpointing for full state snapshots
2. Log only key events/decisions for querying

```python
# LangGraph automatically checkpoints state at each node
# You can query checkpoints to see full state at any point
```

### Option 3: Hybrid Approach (Recommended for Your Use Case)

**Log:**
1. **Full AgentState** after each node (for complete picture)
2. **Key decisions** (for easy querying - confidence checks, routing decisions)
3. **Exceptions** (already doing this)

```python
async def confidence_checker_node(state: AgentState) -> Dict[str, Any]:
    log_ctx = extract_logging_context(state)
    
    try:
        # ... processing ...
        
        decision = "clarification" if low_confidence else "proceed"
        
        # Log key decision (for easy querying)
        await persistence_store.log_audit(
            node_name="confidence_checker",
            event_type="confidence_check_decision",
            data={"decision": decision, "confidence": confidence}
        )
        
        result = {...}
        updated_state = {**state, **result}
        
        # Log full state (for complete picture)
        await persistence_store.log_audit(
            node_name="confidence_checker",
            event_type="state_snapshot",
            data=updated_state  # Full AgentState
        )
        
        return result
    except Exception as e:
        # ... exception handling ...
```

---

## 🔍 Tracking Which Node Updated State

### Current: No Direct Way

**AgentState doesn't have a `last_updated_by` field currently.**

### Solution: Add to Metadata

You can track which node updated state in the `metadata` field:

```python
# In each node
async def confidence_checker_node(state: AgentState) -> Dict[str, Any]:
    result = {
        "needs_clarification": True,
        "metadata": {
            **state.get("metadata", {}),
            "last_updated_by": "confidence_checker",  # ← Track which node
            "last_updated_at": datetime.now().isoformat()
        }
    }
    return result
```

### Better Solution: Log Node Name in Logs Table

**You're already doing this!** The `logs` table has `node_name`:

```sql
SELECT 
    node_name,           -- ← Which node
    event_type,
    timestamp,
    data
FROM logs
WHERE request_id = 'your-uuid'
ORDER BY timestamp;
```

This tells you:
- Which node logged (node_name)
- When it logged (timestamp)
- What the state was (data)

---

## 📋 Recommended Logging Schema

### Add to AgentState (Optional)

```python
# state/schema.py
class AgentState(TypedDict):
    # ... existing fields ...
    
    # === TRACKING (optional) ===
    last_updated_by: Optional[str]  # Node that last updated state
    last_updated_at: Optional[str]  # Timestamp of last update
    update_history: Optional[List[Dict[str, Any]]]  # History of updates
```

### Or Use Logs Table (Current - Recommended)

Your current `logs` table already tracks:
- `node_name` - Which node
- `timestamp` - When
- `data` - Full state (if you log AgentState)
- `request_id` - Links all logs for a request

**Query to see state progression:**
```sql
SELECT 
    node_name,
    timestamp,
    json_extract(data, '$.intent') as intent,
    json_extract(data, '$.confidence') as confidence,
    json_extract(data, '$.response') as response
FROM logs
WHERE request_id = 'your-uuid'
  AND event_type = 'state_snapshot'
ORDER BY timestamp;
```

---

## 🎯 Final Recommendations

### 1. **Log Full AgentState After Each Node**

```python
# Standard pattern for all nodes
async def my_node(state: AgentState) -> Dict[str, Any]:
    log_ctx = extract_logging_context(state)
    
    try:
        # ... node processing ...
        result = {...}
        
        # Log full state after this node
        updated_state = {**state, **result}
        await persistence_store.log_audit(
            session_id=log_ctx["session_id"],
            request_id=log_ctx["request_id"],
            user_id=log_ctx["user_id"],
            node_name="my_node",
            event_type="state_snapshot",
            data=updated_state  # Full AgentState
        )
        
        return result
    except Exception as e:
        # ... exception handling ...
```

### 2. **Keep Key Decision Logs (Optional)**

For important decisions, keep separate logs for easy querying:
- `confidence_check_decision`
- `routing_decision`
- `cache_hit`
- etc.

### 3. **Use Standard Event Types**

```python
# Standard event types
"state_snapshot"      # Full AgentState after node
"node_decision"       # Key decision made by node
"node_input"          # Input to node (if needed)
"node_output"         # Output from node (if needed)
"exception"           # Exception occurred
```

### 4. **Query Pattern**

```sql
-- See full state progression for a request
SELECT 
    node_name,
    timestamp,
    data
FROM logs
WHERE request_id = 'your-uuid'
  AND event_type = 'state_snapshot'
ORDER BY timestamp;

-- See all decisions for a request
SELECT 
    node_name,
    event_type,
    timestamp,
    data
FROM logs
WHERE request_id = 'your-uuid'
  AND event_type IN ('confidence_check_decision', 'routing_decision')
ORDER BY timestamp;
```

---

## 📊 Comparison: Current vs Recommended

| Aspect | Current Approach | Recommended Approach |
|--------|-----------------|---------------------|
| **Logs per node** | 2-3 logs (input, output, decision) | 1-2 logs (state_snapshot, decision) |
| **Data completeness** | Partial (only what node processes) | Complete (full AgentState) |
| **Query complexity** | Multiple joins needed | Single query per request |
| **Debugging** | Hard to reconstruct state | Easy - see full state at each step |
| **Storage** | More entries, partial data | Fewer entries, complete data |
| **Tracking updates** | Via node_name in logs | Via node_name + full state |

---

## ✅ Summary

1. **Log Full AgentState** after each node (not just input/output)
2. **Current logging is somewhat overkill** - multiple partial logs vs one complete log
3. **Track which node updated state** via `node_name` in logs table (you already have this!)
4. **Use standard event type** `"state_snapshot"` for full state logs
5. **Keep key decision logs** for easy querying (optional)

**Benefits:**
- ✅ Complete picture at each step
- ✅ Easier debugging
- ✅ Simpler queries
- ✅ Less duplication
- ✅ Automatic tracking via node_name

**Your current approach works, but logging full AgentState would be more efficient and complete!**

