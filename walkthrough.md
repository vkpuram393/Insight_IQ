# PBM LangGraph Framework - Complete Code Walkthrough

## Overview
This document provides a comprehensive, line-by-line explanation of the PBM (Pharmacy Benefit Management) LangGraph Conversational AI project. It's designed for team members, especially those new to AI and Machine Learning, to understand the entire architecture and code flow.

This system is a **multi-agent conversational AI** built with Python, FastAPI, and LangGraph. When a user sends a chat message (e.g., from Postman), it flows through a series of agents and nodes to understand the user's intent, gather necessary information, and generate an intelligent, context-aware response.

---

## 🎯 What Happens When You Hit the Endpoint

The process starts when a client sends a request to our API.

### Your Postman Request:
```http
POST http://localhost:8000/api/v1/chat
Content-Type: application/json

{
  "text": "why was my claim 12345 rejected?",
  "session_id": "user-session-abc"
}
```

### The Journey of a Request:
1.  **FastAPI Receives Request**: The `main.py` file, running via a Uvicorn server, accepts the incoming HTTP request.
2.  **Routes to Chat Endpoint**: The request is directed to the `/api/v1/chat` endpoint, which is handled by code in `api/routes.py`.
3.  **Enters the LangGraph System**: The router calls the main graph execution function in `langgraph_agent.py`.
4.  **Flows Through Agents & Nodes**: The user's message travels through a predefined graph of agents and helper functions (`nodes/`) to be processed.
    *   First, the **Intent Agent** (`agents/intent_agent.py`) classifies what the user wants.
    *   Then, various nodes might run to fetch data from a database, check for missing information, or perform other business logic.
    *   Finally, the **Response Agent** (`agents/response_agent.py`) formulates the final answer.
5.  **Returns a Response**: The final generated response is sent back to Postman.

---

## 📁 File-by-File Walkthrough

Here is a detailed breakdown of each critical file in the project.

### 1️⃣ `main.py` - The Application Entry Point

This file is the starting point of our application. It sets up the FastAPI web server and orchestrates the application's lifecycle (startup and shutdown).

```python
// filepath: main.py

import sys
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
// ... other imports

# --- Boot Diagnostics (Lines 8-13) ---
# Purpose: When the server starts, these lines print crucial diagnostic information
# to the console. This is extremely helpful for debugging environment issues.
print(f"[BOOT] Python Version: {sys.version}")
print(f"[BOOT] Environment: {os.environ.get('ENVIRONMENT')}")
# It tells you which Python executable is being used, what the current settings are
# (e.g., are we in "development" or "production"?), and whether mock services are enabled.

# --- FastAPI App Creation (Lines 28-32) ---
# Purpose: This creates the main FastAPI application object.
app = FastAPI(
    title="PBM LangGraph Framework",
    description="A multi-agent framework for PBM",
    version="1.0.0"
)
# The `title`, `description`, and `version` are used to automatically generate
# interactive API documentation, which you can view at http://localhost:8000/docs.

# --- CORS Middleware (Lines 35-41) ---
# Purpose: Enables Cross-Origin Resource Sharing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows web pages from any domain to call our API.
    # ... other settings
)
# This is essential if you have a separate frontend (e.g., a React app) that
# needs to communicate with this backend.

# --- Startup Event Handler (Lines 44-48) ---
# Purpose: This function runs ONCE when the server starts up.
@app.on_event("startup")
async def startup_event():
    # It's the perfect place to initialize our LangGraph agent system.
    await init_graph()
    print("[STARTUP] LangGraph system initialized.")
# By initializing the graph here, we ensure the entire AI system is ready
# before the server starts accepting any chat requests. This prevents errors
# on the first request.

# --- Shutdown Event Handler (Lines 51-55) ---
# Purpose: This function runs ONCE when the server is shutting down (e.g., you press Ctrl+C).
@app.on_event("shutdown")
async def shutdown_event():
    # It's used for graceful cleanup.
    await close_graph()
    print("[SHUTDOWN] Resources cleaned up.")
# This is important for closing database connections or other resources properly.

# --- Include API Router (Line 58) ---
# Purpose: This line attaches all the API endpoints defined in `api/routes.py`.
app.include_router(api_router, prefix="/api/v1")
# The `prefix` means all routes from that file will start with `/api/v1`.
# So, a route defined as `/chat` in `routes.py` becomes `/api/v1/chat`.

# --- Main Execution Block (Lines 70-75) ---
# Purpose: This block runs only when you execute `python main.py` directly.
if __name__ == "__main__":
    # It starts the Uvicorn web server.
    uvicorn.run(
        "main:app",
        host="0.0.0.0", # Makes the server accessible on your local network.
        port=8000,
        reload=True # The server will auto-restart when you save a file.
    )
```

### 2️⃣ `api/routes.py` - The Chat Endpoint Handler

This file defines the actual `/api/v1/chat` endpoint that receives the user's message and returns the AI's response.

```python
// filepath: api/routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
// ... other imports

# --- Pydantic Models (Lines 11-22) ---
# Purpose: These classes define the expected JSON structure for requests and responses.
# FastAPI uses them to automatically validate incoming and outgoing data.
class ChatRequest(BaseModel):
    text: str # The user's message. This field is required.
    session_id: Optional[str] = None # An optional ID to track the conversation.

class ChatResponse(BaseModel):
    response: str # The AI's final answer.
    session_id: str
    # ... other fields like intent, confidence, etc.

# --- API Router (Line 24) ---
# Purpose: Creates a router object to group our chat-related endpoints.
router = APIRouter()

# --- Chat Endpoint (Lines 27-55) ---
# Purpose: This is the main function that handles POST requests to `/api/v1/chat`.
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # `request: ChatRequest` tells FastAPI to expect a JSON body that matches
    # the `ChatRequest` model. If it doesn't match, FastAPI sends back an error automatically.

    # A unique session ID is crucial for maintaining conversation history.
    # If the client doesn't provide one, we create a new one.
    session_id = request.session_id or str(uuid.uuid4())

    try:
        # This is the core of the operation. We call the `run_graph` function,
        # passing the user's text and the session ID.
        # The `await` keyword means we wait here until the entire LangGraph
        # system has finished processing and produced a final result.
        final_state = await run_graph(
            text=request.text,
            session_id=session_id,
            user_info=request.user_info or {}
        )

        # After the graph runs, we construct a `ChatResponse` object
        # from the results stored in the `final_state` dictionary.
        return ChatResponse(
            response=final_state.get("response", "Sorry, I encountered an error."),
            session_id=session_id,
            intent=final_state.get("intent"),
            # ... and so on for other fields.
        )
    except Exception as e:
        # If anything goes wrong during the graph execution, we log the error
        # and return a 500 Internal Server Error to the client.
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 3️⃣ `agents/intent_agent.py` - The First Agent

This is the first "brain" in our system. Its job is to analyze the user's raw text and figure out what they want. This is known as **Intent Classification**.

```python
// filepath: agents/intent_agent.py

import re
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
// ... other imports

# --- Mock LLM (Lines 15-51) ---
# Purpose: A fake Large Language Model (LLM) for development.
class MockLLM:
    # This class simulates the behavior of a real AI model without making actual API calls.
    # This is great for testing because it's free, fast, and predictable.
    async def ainvoke(self, messages: List[Any]):
        # It uses simple keyword matching to guess the intent.
        if "reject" in text:
            intent = "claim_rejection_reason"
        # It even uses a regular expression (regex) to find a claim number.
        claim_match = re.search(r"\b\d{4,10}\b", text)
        if claim_match:
            entities["claim_number"] = claim_match.group(0)
        # It returns a JSON string, just like the real OpenAI API would.
        return Response(content=json.dumps({...}))

# --- Intent Agent Node (Lines 56-149) ---
# Purpose: This is the main function for the intent agent. It gets called by LangGraph.
async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    logger.info("🤖 AGENT 1: Intent Classification")

    # It extracts the user's message and conversation history from the current state.
    text = state["text"]
    history = state.get("conversation_history", [])

    # --- LLM Selection (Lines 90-98) ---
    # It checks the `use_mock_llm` setting from our config file.
    if settings.use_mock_llm:
        llm = MockLLM() # Use the fake LLM.
    else:
        llm = ChatOpenAI(...) # Use the real OpenAI LLM.

    # --- Prompt Engineering (Lines 101-125) ---
    # This is one of the most important parts of modern AI. We are carefully
    # crafting the instructions (the "prompt") for the LLM.
    raw_system_prompt = (
        "You are an intent classification agent...\n"
        "Available intents:\n"
        "- claim_status: User wants to check claim status\n"
        "- claim_rejection_reason: User wants to know why claim was rejected\n"
        "Respond ONLY with JSON like:\n"
        '{"intent": "claim_status", "confidence": 0.95, "entities": {"claim_number": "12345"}}\n'
    )
    # We tell the LLM exactly what its job is, what the possible intents are,
    # and what format it MUST respond in (JSON). This makes the output reliable.

    # We create a prompt template that will include the system instructions,
    # the user's message, and the conversation history.
    prompt = ChatPromptTemplate.from_messages([...])

    # --- LLM Invocation (Line 135) ---
    # We send the fully formatted prompt to the LLM (either mock or real).
    response = await llm.ainvoke(messages)

    # --- Parsing the Response (Lines 138-142) ---
    # The LLM returns a JSON string. We need to parse it into a Python dictionary.
    try:
        result = json.loads(response.content)
    except Exception:
        # If the LLM messes up and doesn't return valid JSON, we handle it gracefully.
        result = {"intent": "unknown", "confidence": 0.1, "entities": {}}

    # We extract the intent, confidence, and any entities from the result.
    intent = result.get("intent", "unknown")
    confidence = float(result.get("confidence", 0.1))
    entities = result.get("entities") or {}

    logger.info(f"🎯 Intent: {intent} ({confidence:.2f}) | Entities: {entities}")

    # --- Returning the New State (Lines 144-149) ---
    # The function returns a dictionary containing the newly found information.
    # LangGraph will automatically merge this into the main `AgentState`.
    return {
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
    }
```

### 4️⃣ `langgraph_agent.py` - The Graph Orchestrator

This file is the heart of the LangGraph system. It defines the state, the nodes, and the edges (the flow) of our conversational AI.

```python
// filepath: langgraph_agent.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from state.schema import AgentState
// ... other imports

# --- Global Variables ---
# These hold the compiled graph and the checkpointer, so we don't have to
# rebuild them for every single request.
_graph_compiled = None
_checkpointer = None

# --- Graph Initialization (`init_graph`) ---
# This function is called once at startup from `main.py`.
async def init_graph():
    global _graph_compiled, _checkpointer

    # --- Checkpointer Setup ---
    # The checkpointer is responsible for saving the state of every conversation.
    # Here, we're using a simple SQLite database file.
    _checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

    # --- Graph Definition ---
    # We create a new `StateGraph` and tell it to use our `AgentState` schema.
    workflow = StateGraph(AgentState)

    # --- Adding Nodes ---
    # Each agent and helper function is added as a "node" in the graph.
    workflow.add_node("safety_precheck", safety_precheck_node)
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("intent_agent", intent_agent_node)
    workflow.add_node("clarification_or_response", clarification_or_response_router)
    // ... and so on for all other nodes.

    # --- Defining the Flow (Edges) ---
    # This is where we define the conversational logic.
    workflow.set_entry_point("safety_precheck") # The first node to run.
    workflow.add_edge("safety_precheck", "check_cache") # After precheck, check the cache.
    workflow.add_edge("check_cache", "build_context") # After cache, build context.

    # --- Conditional Edges ---
    # LangGraph can also handle conditional logic.
    workflow.add_conditional_edges(
        "clarification_or_response", # Starting from this node...
        should_ask_for_clarification, # ...call this function to decide where to go next...
        {
            "ask_clarification": "clarification_node", # If it returns "ask_clarification", go here.
            "generate_response": "response_agent"  # If it returns "generate_response", go there.
        }
    )

    # --- Compiling the Graph ---
    # Finally, we compile the workflow. This turns our definition into an
    # executable object. We also attach the checkpointer to it.
    _graph_compiled = workflow.compile(checkpointer=_checkpointer)

# --- Graph Execution (`run_graph`) ---
# This is the function called by `api/routes.py` for each chat request.
async def run_graph(text: str, session_id: str, user_info: dict) -> dict:
    # It defines the initial state for the conversation.
    initial_state = {
        "text": text,
        "user_info": user_info,
        "messages": [("user", text)]
    }

    # The `config` dictionary tells LangGraph which conversation thread we're working on.
    # This is how it loads the correct history from the `checkpoints.db`.
    config = {"configurable": {"thread_id": session_id}}

    # We invoke the compiled graph with the initial state and config.
    # LangGraph handles the rest, running through the nodes and edges we defined.
    final_state = await _graph_compiled.ainvoke(initial_state, config)
    return final_state
```

---

## 🧠 Key ML & AI Concepts Explained

*   **Agent**: An autonomous component that can make decisions and take actions. In our case, the `Intent Agent` decides what the user wants, and the `Response Agent` decides what to say.
*   **LLM (Large Language Model)**: A massive neural network (like OpenAI's GPT-4) trained on vast amounts of text. It's incredibly good at understanding and generating human-like language.
*   **Prompt Engineering**: The art and science of writing effective instructions (prompts) for an LLM to get the desired output. This is a critical skill in modern AI development.
*   **Intent Classification**: The task of identifying the user's goal or intention from their message (e.g., "check claim status").
*   **Entity Extraction**: The task of identifying and extracting specific pieces of information from text, like a claim number, a date, or a name.
*   **StateGraph**: A LangGraph concept where a central "state" object (a Python dictionary) is passed from node to node. Each node can read from and write to the state, progressively building up the information needed to answer the user's query.
*   **Checkpointing**: The process of saving the state of a conversation at each step. This provides memory, allowing the chatbot to remember previous turns in the conversation. It's also essential for debugging and auditing.
*   **Mocking**: Creating a simplified, fake version of a service (like an LLM). This is a standard software engineering practice that allows for rapid, cost-effective, and predictable testing.

This walkthrough should provide a solid foundation for understanding the project. As you explore the code, refer back to this document to see how each piece fits into the larger puzzle.
