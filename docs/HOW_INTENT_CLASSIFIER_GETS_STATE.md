# How Intent Classifier Gets State and Extracts Values

## 🎯 How Intent Classifier Receives State

### 1. **Function Signature**

The intent classifier is defined as a function that takes `AgentState` as a parameter:

```python
# agents/intent_agent.py
async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Intent Classifier Node
    
    LangGraph automatically calls this function and passes the FULL AgentState.
    You don't call this function yourself - LangGraph does it!
    """
```

### 2. **LangGraph Automatically Passes State**

When LangGraph executes the graph:

```python
# langgraph_agent.py
workflow.add_node("intent_agent", intent_agent_node)  # Register the function
workflow.add_edge("build_context", "intent_agent")      # Connect nodes

# When graph runs:
# LangGraph internally does:
current_state = {...}  # Full AgentState with all fields
result = await intent_agent_node(current_state)  # ← LangGraph calls it automatically!
```

**You never call `intent_agent_node()` directly** - LangGraph does it automatically when it reaches that node in the graph.

---

## 📥 How to Extract Values from State

### Method 1: Direct Dictionary Access (Use When Field is Required)

```python
async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    # Direct access - will raise KeyError if missing
    text = state["text"]                    # ✅ User's input message (required field)
    session_id = state["session_id"]        # ✅ Session ID (required field)
```

**Use this when:**
- The field is **required** (defined as non-Optional in AgentState)
- You want an error if the field is missing

### Method 2: `.get()` with Default (Use When Field is Optional)

```python
async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    # Safe access with defaults - won't raise error if missing
    uuid = state.get("uuid")                           # Returns None if missing
    domain = state.get("domain")                       # Returns None if missing
    intent = state.get("intent")                       # Returns None if missing
    confidence = state.get("confidence")               # Returns None if missing
    
    # With custom default value
    conversation_history = state.get("conversation_history", [])  # Returns [] if missing
    entities = state.get("entities", {})              # Returns {} if missing
```

**Use this when:**
- The field is **optional** (defined as `Optional[...]` in AgentState)
- You want a default value if missing

### Method 3: Nested Dictionary Access

```python
async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    # Access nested dictionaries
    user_info = state.get("user_info", {})            # Get user_info dict
    user_id = user_info.get("user_id")                # Get nested field
    
    # Or chain it in one line
    user_id = state.get("user_info", {}).get("user_id")
    
    # Access metadata
    metadata = state.get("metadata", {})
    normalized_data = metadata.get("normalized_data")
```

---

## 📋 Real Example from Your Code

Here's how the intent classifier **currently** extracts values:

```python
# agents/intent_agent.py - Line 74-95
async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    node_name = "intent_agent"
    
    # Extract logging context (optional fields)
    session_id = state.get("session_id", "unknown")    # Optional, default to "unknown"
    request_id = state.get("uuid")                     # Optional, returns None if missing
    user_id = state.get("user_info", {}).get("user_id")  # Nested access
    
    # Extract required fields (direct access)
    text = state["text"]                              # Required field - direct access
    
    # Extract optional fields (with defaults)
    history = state.get("conversation_history", [])   # Optional, default to []
    
    # Use the extracted values
    logger.info(f"Processing: {text}")
    # ... do intent classification with text and history ...
```

---

## 🎯 Best Practices for Extracting Values

### ✅ DO: Use `.get()` for Optional Fields

```python
# ✅ GOOD - Safe for optional fields
uuid = state.get("uuid")
domain = state.get("domain")
normalized_data = state.get("metadata", {}).get("normalized_data")
```

### ✅ DO: Use Direct Access for Required Fields

```python
# ✅ GOOD - Required fields are guaranteed to exist
text = state["text"]
session_id = state["session_id"]
```

### ❌ DON'T: Use Direct Access on Optional Fields

```python
# ❌ BAD - Will raise KeyError if uuid is None
uuid = state["uuid"]  # Don't do this if uuid is Optional[str]
```

### ✅ DO: Provide Defaults for Collections

```python
# ✅ GOOD - Prevents None errors
history = state.get("conversation_history", [])
entities = state.get("entities", {})
metadata = state.get("metadata", {})
```

---

## 📊 Complete Example: Extracting All Orchestrator Data

Here's how the intent classifier should extract values from orchestrator data:

```python
async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Intent Classifier Node
    
    Extracts values from AgentState that was populated by orchestrator.
    """
    
    # ========================================================================
    # REQUIRED FIELDS (Direct Access)
    # ========================================================================
    text = state["text"]                              # User's input message
    session_id = state["session_id"]                  # Session identifier
    
    # ========================================================================
    # ORCHESTRATOR DATA (Optional - Use .get())
    # ========================================================================
    request_uuid = state.get("uuid")                  # UUID from orchestrator
    domain = state.get("domain")                      # Domain from orchestrator
    
    # ========================================================================
    # USER INFO (Nested Access)
    # ========================================================================
    user_info = state.get("user_info", {})
    user_id = user_info.get("user_id")
    plan_type = user_info.get("plan_type")
    
    # ========================================================================
    # NORMALIZED DATA (Nested in metadata)
    # ========================================================================
    metadata = state.get("metadata", {})
    normalized_data = metadata.get("normalized_data")
    
    if normalized_data:
        cleaned_text = normalized_data.get("cleaned_text")
        language = normalized_data.get("language")
        # Use normalized data if available
        text_to_classify = cleaned_text or text
    else:
        text_to_classify = text
    
    # ========================================================================
    # CONTEXT DATA (Optional - Use .get() with defaults)
    # ========================================================================
    conversation_history = state.get("conversation_history", [])
    relevant_facts = state.get("relevant_facts", [])
    
    # ========================================================================
    # PREVIOUS NODE OUTPUTS (Optional - Use .get())
    # ========================================================================
    intent = state.get("intent")                      # From previous run (if any)
    confidence = state.get("confidence")              # From previous run (if any)
    entities = state.get("entities", {})              # From previous run (if any)
    
    # ========================================================================
    # USE THE EXTRACTED VALUES
    # ========================================================================
    logger.info(f"Classifying intent for: {text_to_classify}")
    logger.info(f"Domain: {domain}, UUID: {request_uuid}, User: {user_id}")
    
    # Use domain to customize classification
    if domain == "claims":
        # Use claims-specific prompts
        pass
    elif domain == "prescriptions":
        # Use prescriptions-specific prompts
        pass
    
    # Use normalized_data if available
    # Use conversation_history for context
    # ... do intent classification ...
    
    # Return partial update
    return {
        "intent": "claim_status",
        "confidence": 0.95,
        "entities": {"claim_number": "12345"}
    }
```

---

## 🔍 Step-by-Step: What Happens When Graph Runs

### Step 1: Graph Execution Starts

```python
# langgraph_agent.py
initial_state = create_initial_state(
    text="What's my claim status?",
    session_id="session_123",
    user_info={"user_id": "member_123"},
    uuid="abc-123-def",        # From orchestrator
    domain="claims"            # From orchestrator
)

# initial_state = {
#     "text": "What's my claim status?",
#     "session_id": "session_123",
#     "uuid": "abc-123-def",
#     "domain": "claims",
#     "user_info": {"user_id": "member_123"},
#     "intent": None,
#     "confidence": None,
#     ...
# }
```

### Step 2: LangGraph Passes State to Intent Classifier

```python
# LangGraph internally does:
current_state = {
    "text": "What's my claim status?",
    "session_id": "session_123",
    "uuid": "abc-123-def",
    "domain": "claims",
    "user_info": {"user_id": "member_123"},
    "intent": None,
    "confidence": None,
    "conversation_history": [...],
    ...
}

# LangGraph calls your function:
result = await intent_agent_node(current_state)  # ← Passes FULL state
```

### Step 3: Intent Classifier Extracts Values

```python
async def intent_agent_node(state: AgentState):
    # Extract values
    text = state["text"]                           # "What's my claim status?"
    uuid = state.get("uuid")                       # "abc-123-def"
    domain = state.get("domain")                   # "claims"
    user_id = state.get("user_info", {}).get("user_id")  # "member_123"
    
    # Use the values
    logger.info(f"Processing: {text} (domain: {domain}, uuid: {uuid})")
    
    # ... do classification ...
    
    return {"intent": "claim_status", "confidence": 0.95}
```

### Step 4: LangGraph Merges Result

```python
# LangGraph merges the result into state:
updated_state = {
    "text": "What's my claim status?",      # ← Kept from before
    "session_id": "session_123",            # ← Kept from before
    "uuid": "abc-123-def",                  # ← Kept from before
    "domain": "claims",                     # ← Kept from before
    "intent": "claim_status",               # ← UPDATED (from node output)
    "confidence": 0.95,                     # ← UPDATED (from node output)
    ...
}
```

---

## 🎯 Key Takeaways

1. **LangGraph automatically passes state** - You don't call the function, LangGraph does
2. **Function receives FULL AgentState** - All fields are available
3. **Use `state["key"]` for required fields** - Direct access
4. **Use `state.get("key", default)` for optional fields** - Safe access with defaults
5. **Use nested `.get()` for nested dictionaries** - `state.get("user_info", {}).get("user_id")`
6. **Return only what you update** - Return partial dictionary, LangGraph merges it

---

## 📝 Summary

**How intent classifier gets state:**
- LangGraph automatically calls `intent_agent_node(state)` when graph execution reaches that node
- The function receives the **full AgentState** as a parameter
- You don't need to do anything special - just define the function with `state: AgentState` parameter

**How to extract values:**
- **Required fields**: `text = state["text"]`
- **Optional fields**: `uuid = state.get("uuid")`
- **Nested fields**: `user_id = state.get("user_info", {}).get("user_id")`
- **With defaults**: `history = state.get("conversation_history", [])`

**That's it!** LangGraph handles the passing, you just extract what you need from the state parameter.

