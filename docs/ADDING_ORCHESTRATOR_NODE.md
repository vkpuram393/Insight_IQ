# Adding Orchestrator as a Node - Guide

## 🎯 Answer to Your Questions

### 1. Does LangGraph pass initial state to the first node?

**YES!** LangGraph automatically passes the initial state to the **entry point node**.

Looking at your code:
```python
# langgraph_agent.py - Line 76
workflow.set_entry_point("safety_precheck")  # ← This is the first node
```

When you call:
```python
final_state = await app.ainvoke(initial_state)
```

LangGraph internally does:
```python
# LangGraph automatically calls the entry point with initial_state
result = await safety_precheck_node(initial_state)  # ← Passes initial_state to first node
```

### 2. Can you add orchestrator as a node?

**YES!** You can absolutely add orchestrator logic as a node. Here's how:

---

## 📝 Step-by-Step: Adding Orchestrator Node

### Step 1: Create the Orchestrator Node Function

Create a new file or add to existing file:

```python
# nodes/orchestrator.py

from typing import Dict, Any
from state.schema import AgentState
from core.logger import get_logger
import uuid
from datetime import datetime

logger = get_logger(__name__)

async def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """
    Orchestrator Node - First node in the graph
    
    Responsibilities:
    - Normalize user input
    - Extract/generate UUID
    - Determine domain
    - Pre-process data
    - Set up initial metadata
    """
    logger.info("🎯 Orchestrator Node: Normalizing input")
    
    # Extract raw input
    text = state.get("text", "")
    session_id = state.get("session_id", "unknown")
    user_info = state.get("user_info", {})
    
    # ========================================================================
    # NORMALIZATION
    # ========================================================================
    normalized_text = text.strip().lower()  # Normalize text
    
    # ========================================================================
    # GENERATE/EXTRACT UUID
    # ========================================================================
    # If UUID not provided, generate one
    request_uuid = state.get("uuid")
    if not request_uuid:
        request_uuid = str(uuid.uuid4())
        logger.info(f"Generated UUID: {request_uuid}")
    
    # ========================================================================
    # DETERMINE DOMAIN
    # ========================================================================
    # Extract domain from state or determine from context
    domain = state.get("domain")
    if not domain:
        # You can add logic here to determine domain
        # For example, check user_info, text content, etc.
        domain = "claims"  # Default or determine from context
        logger.info(f"Determined domain: {domain}")
    
    # ========================================================================
    # PRE-PROCESS DATA
    # ========================================================================
    normalized_data = {
        "cleaned_text": normalized_text,
        "original_text": text,
        "language": "en",  # Could detect language here
        "timestamp": datetime.now().isoformat(),
        "normalized_at": datetime.now().isoformat()
    }
    
    # ========================================================================
    # UPDATE STATE
    # ========================================================================
    return {
        "text": normalized_text,  # Update with normalized text
        "uuid": request_uuid,     # Set UUID
        "domain": domain,          # Set domain
        "metadata": {
            **state.get("metadata", {}),
            "normalized_data": normalized_data,
            "orchestrator_processed": True,
            "request_uuid": request_uuid
        }
    }
```

### Step 2: Export the Node

```python
# nodes/__init__.py

from nodes.orchestrator import orchestrator_node

__all__ = [
    # ... existing nodes ...
    "orchestrator_node",
]
```

### Step 3: Add to Graph and Set as Entry Point

```python
# langgraph_agent.py

from nodes import (
    orchestrator_node,  # ← Add this import
    safety_precheck_node,
    # ... rest of nodes
)

def _build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    # ========================================================================
    # ADD ORCHESTRATOR NODE (FIRST)
    # ========================================================================
    workflow.add_node("orchestrator", orchestrator_node)  # ← Add orchestrator node
    
    # Add all other nodes
    workflow.add_node("safety_precheck", safety_precheck_node)
    workflow.add_node("check_cache", check_cache_node)
    # ... rest of nodes ...
    
    # ========================================================================
    # SET ORCHESTRATOR AS ENTRY POINT
    # ========================================================================
    workflow.set_entry_point("orchestrator")  # ← Change entry point to orchestrator
    
    # ========================================================================
    # CONNECT ORCHESTRATOR TO NEXT NODE
    # ========================================================================
    workflow.add_edge("orchestrator", "safety_precheck")  # ← Orchestrator → Safety Precheck
    
    # Rest of your edges...
    workflow.add_conditional_edges(
        "safety_precheck", should_continue_after_precheck, {"check_cache": "check_cache", END: END}
    )
    # ... rest of edges ...
    
    return workflow
```

---

## 🎬 Complete Flow After Adding Orchestrator

```
START
  ↓
orchestrator_node (NEW - Entry Point)
  ↓ (normalizes input, sets UUID, domain)
safety_precheck
  ↓
check_cache
  ↓
build_context
  ↓
intent_agent
  ↓
... rest of graph
```

---

## 📊 What Happens When Graph Runs

### Before (Current):

```python
# Initial state from API
initial_state = {
    "text": "What's my claim status?",
    "session_id": "session_123",
    "uuid": None,        # ← Not set
    "domain": None,      # ← Not set
    ...
}

# LangGraph calls entry point
result = await safety_precheck_node(initial_state)  # ← First node
```

### After (With Orchestrator):

```python
# Initial state from API
initial_state = {
    "text": "What's my claim status?",
    "session_id": "session_123",
    "uuid": None,        # ← Not set yet
    "domain": None,      # ← Not set yet
    ...
}

# LangGraph calls entry point (orchestrator)
result1 = await orchestrator_node(initial_state)  # ← First node (NEW!)
# Returns: {
#     "text": "what's my claim status?",  # Normalized
#     "uuid": "abc-123-def",              # Generated
#     "domain": "claims",                  # Determined
#     "metadata": {"normalized_data": {...}}
# }

# LangGraph merges and calls next node
updated_state = merge(initial_state, result1)
result2 = await safety_precheck_node(updated_state)  # ← Second node
```

---

## 🎯 Key Points

### 1. **Entry Point is Set Explicitly**

```python
workflow.set_entry_point("orchestrator")  # ← This tells LangGraph: "Start here!"
```

### 2. **LangGraph Automatically Passes State**

When you call `app.ainvoke(initial_state)`:
- LangGraph looks at the entry point
- Calls that node with `initial_state`
- Merges the result
- Calls the next node (based on edges you defined)
- Continues until END

### 3. **Orchestrator Node Can Do Anything**

The orchestrator node can:
- Normalize input
- Generate UUID
- Determine domain
- Pre-process data
- Validate input
- Extract metadata
- Set up logging context
- etc.

### 4. **You Control the Flow**

```python
# You define the flow
workflow.add_edge("orchestrator", "safety_precheck")  # Orchestrator → Safety

# LangGraph executes it automatically
```

---

## ✅ Summary

1. **Yes, LangGraph passes initial state to the first node** (entry point)
2. **Yes, you can add orchestrator as a node** - just:
   - Create `orchestrator_node()` function
   - Add it to graph: `workflow.add_node("orchestrator", orchestrator_node)`
   - Set as entry point: `workflow.set_entry_point("orchestrator")`
   - Connect to next node: `workflow.add_edge("orchestrator", "safety_precheck")`

That's it! LangGraph will automatically:
- Call orchestrator first with initial_state
- Merge orchestrator's result
- Call safety_precheck next
- Continue through the graph

