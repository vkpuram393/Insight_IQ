# Who Invokes `app.ainvoke()`? - Call Chain Explained

## 🎯 The Call Chain

Here's exactly who calls what in your code:

```
ORCHESTRATOR (External System)
    ↓
    HTTP POST /chat
    ↓
API ENDPOINT (api/routes.py - line 56)
    ↓
    Calls: await run_graph(...)
    ↓
run_graph() (langgraph_agent.py - line 131)
    ↓
    Calls: await app.ainvoke(initial_state)
    ↓
LANGGRAPH (Executes the graph)
```

---

## 📍 Step-by-Step with Actual Code

### Step 1: Orchestrator Calls API

**Orchestrator (External System)** makes an HTTP request:

```python
# Orchestrator code (external, not in this repo)
import httpx

response = await httpx.post(
    "http://localhost:8000/chat",
    json={
        "text": "What's my claim status?",
        "session_id": "session_123",
        "user_info": {"user_id": "member_123"}
    }
)
```

### Step 2: API Endpoint Receives Request

**File: `api/routes.py` - Line 56**

```python
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Time the request processing
        async with RequestTimer(session_id, EventType.RESPONSE_GENERATED) as timer:
            # ← THIS IS WHERE run_graph() IS CALLED
            final_state = await run_graph(
                text=request.text,
                session_id=session_id,
                user_info=request.user_info or {}
            )
            # ↑ Line 56: API endpoint calls run_graph()
        
        # Extract response and return to orchestrator
        return ChatResponse(
            response=final_state.get("response", ""),
            session_id=session_id,
            intent=final_state.get("intent"),
            ...
        )
```

**Who calls this?** The FastAPI framework when it receives an HTTP POST to `/chat`.

### Step 3: `run_graph()` Calls `app.ainvoke()`

**File: `langgraph_agent.py` - Line 131**

```python
async def run_graph(text: str, session_id: str, user_info: dict = None):
    from state.schema import create_initial_state
    await init_graph()
    initial_state = create_initial_state(text, session_id, user_info)
    config = {"configurable": {"thread_id": session_id}}
    
    # ← THIS IS WHERE app.ainvoke() IS CALLED
    final_state = await _graph_compiled.ainvoke(initial_state, config)  # Line 131
    # ↑ run_graph() calls app.ainvoke()
    
    return final_state
```

**Who calls this?** The API endpoint (`api/routes.py` line 56) calls `run_graph()`.

### Step 4: LangGraph Executes

**LangGraph framework** (inside `app.ainvoke()`) executes the graph:
- Calls each node in sequence
- Manages state transitions
- Returns final state

---

## 🔍 Visual Flow with Code References

```
┌─────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR (External System)                               │
│                                                              │
│ POST http://localhost:8000/chat                             │
│ {                                                             │
│   "text": "What's my claim status?",                         │
│   "session_id": "session_123"                                │
│ }                                                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓ HTTP Request
┌─────────────────────────────────────────────────────────────┐
│ FastAPI Framework                                            │
│ Receives POST /chat                                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓ Routes to handler
┌─────────────────────────────────────────────────────────────┐
│ api/routes.py - Line 35-60                                   │
│                                                              │
│ @router.post("/chat")                                        │
│ async def chat(request: ChatRequest):                        │
│     ...                                                      │
│     final_state = await run_graph(  ← LINE 56                │
│         text=request.text,                                   │
│         session_id=session_id,                               │
│         user_info=request.user_info or {}                   │
│     )                                                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓ Function call
┌─────────────────────────────────────────────────────────────┐
│ langgraph_agent.py - Line 126-132                           │
│                                                              │
│ async def run_graph(...):                                   │
│     initial_state = create_initial_state(...)               │
│     final_state = await _graph_compiled.ainvoke(  ← LINE 131 │
│         initial_state,                                       │
│         config                                               │
│     )                                                        │
│     return final_state                                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓ LangGraph execution
┌─────────────────────────────────────────────────────────────┐
│ LangGraph Framework (Internal)                               │
│                                                              │
│ 1. Call safety_precheck_node(state)                         │
│ 2. Merge result into state                                  │
│ 3. Call check_cache_node(state)                             │
│ 4. Merge result into state                                  │
│ 5. Call build_context_node(state)                           │
│ 6. ... (continues through all nodes)                        │
│ 7. Return final_state                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Answer: Who Invokes `app.ainvoke()`?

**Answer: `run_graph()` function invokes `app.ainvoke()`**

**Location:** `langgraph_agent.py` - Line 131

**Who calls `run_graph()`?** The API endpoint (`api/routes.py` - Line 56)

**Who calls the API endpoint?** The orchestrator (external system) via HTTP POST

---

## 📝 Summary

1. **Orchestrator** (external) → HTTP POST `/chat`
2. **FastAPI** → Routes to `chat()` function in `api/routes.py`
3. **API Endpoint** (`api/routes.py` line 56) → Calls `run_graph()`
4. **`run_graph()`** (`langgraph_agent.py` line 131) → Calls `app.ainvoke()`
5. **LangGraph** → Executes the graph automatically

**So the direct answer is:** `run_graph()` function invokes `app.ainvoke()`, and `run_graph()` itself is called by the API endpoint when it receives an HTTP request.

