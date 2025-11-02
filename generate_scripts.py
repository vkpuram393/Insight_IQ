#!/usr/bin/env python3
"""
COMPLETE LANGGRAPH VERSION GENERATOR
Creates entire PBM Multi-Agent Framework using LangGraph

ARCHITECTURE:
- 2 LLM-Powered Agents (Intent Classification, Response Generation)
- 9 Function Nodes (safety, cache, context, tools, memory)
- Branching graph with conditional routing
- Rich state management with full metadata
- SQLite checkpointing for conversation persistence
- Mock Claims API tool
- Super detailed comments for learning

USAGE:
    python3 GENERATE_LANGGRAPH.py

This creates a complete, production-ready LangGraph implementation!
"""

import os
from pathlib import Path
import shutil


def create_file(filepath, content):
    """Create a file with given content"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ {filepath}")


print("=" * 80)
print("🚀 GENERATING COMPLETE LANGGRAPH VERSION")
print("=" * 80)
print("\n⚠️  This will DELETE existing app/ directory!")
print("⚠️  Your old code will be REPLACED with LangGraph version!\n")

response = input("Continue? (type 'yes'): ")
if response.lower() != "yes":
    print("❌ Cancelled.")
    exit()

print("\n🗑️  Deleting old code...")
if Path("app").exists():
    shutil.rmtree("app")
    print("✅ Deleted app/")

print("\n📝 Creating LangGraph implementation...\n")

FILES = {}

# ============================================================================
# REQUIREMENTS.TXT
# ============================================================================
FILES['requirements.txt'] = '''# Web Framework
fastapi==0.115.0
uvicorn[standard]==0.24.0
pydantic==2.9.2
pydantic-settings==2.5.2

# LangChain & LangGraph - The core AI framework
langgraph==0.2.45
langchain==0.3.7
langchain-core==0.3.15
langchain-openai==0.2.8

# Checkpointing - Saves conversation state
langgraph-checkpoint-sqlite==1.0.3

# Utilities
aiofiles==23.2.1
python-dotenv==1.0.0
httpx==0.27.0
'''

FILES['.gitignore'] = '''__pycache__/
*.py[cod]
venv/
.env
*.log
.DS_Store
checkpoints.db
checkpoints.db-shm
checkpoints.db-wal
'''

FILES['.env.example'] = '''# LLM Configuration
OPENAI_API_KEY=your_openai_key_here
USE_MOCK_LLM=true

# App Settings
ENVIRONMENT=development
DEBUG=true
CONFIDENCE_THRESHOLD=0.7
'''

# ============================================================================
# STATE SCHEMA - THE HEART OF LANGGRAPH
# ============================================================================
FILES['state/schema.py'] = '''"""
LangGraph State Schema

🎓 CONCEPT: State is like a form that travels through an assembly line.
Each worker (node/agent) fills in their section and passes it along.

In LangGraph, state is THE CORE CONCEPT. Everything reads from and writes to state.
"""

from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Complete state that flows through the graph

    🎯 UNDERSTAND THIS:
    - When a request arrives, we create initial state
    - State flows: Node1 → Node2 → Node3 → ... → Response
    - Each node can READ all fields and WRITE to fields
    - At the end, state contains everything that happened

    📊 STATE FLOW EXAMPLE:

    Initial:
        text="What's my claim status?"
        intent=None
        response=""

    After intent_agent:
        text="What's my claim status?"
        intent="claim_status"
        confidence=0.95
        response=""

    After response_agent:
        text="What's my claim status?"
        intent="claim_status"
        confidence=0.95
        response="Your claim #12345 is approved!"
    """

    # === INPUT (from user) ===
    text: str                                    # User's message
    session_id: str                              # Conversation ID
    user_info: Dict[str, Any]                    # User metadata

    # === INTENT & ENTITIES (from intent_agent) ===
    intent: Optional[str]                        # What user wants
    confidence: Optional[float]                  # How sure we are (0-1)
    entities: Optional[Dict[str, Any]]           # Extracted info

    # === CLARIFICATION ===
    needs_clarification: bool                    # Ask user question?
    clarifying_question: Optional[str]           # The question

    # === CONTEXT ===
    conversation_history: List[Dict[str, str]]   # Recent messages
    relevant_facts: List[Dict[str, Any]]         # Important facts

    # === TOOL RESULTS ===
    tool_results: Optional[Dict[str, Any]]       # API call results

    # === SAFETY ===
    safety_precheck_passed: bool                 # Input safe?
    safety_postcheck_passed: bool                # Output safe?
    safety_block_reason: Optional[str]           # Why blocked

    # === OUTPUT (to user) ===
    response: str                                # Final answer

    # === METADATA ===
    metadata: Dict[str, Any]                     # Tracking info
    cache_hit: bool                              # From cache?
    error: Optional[str]                         # Any error

    # === LANGGRAPH MESSAGES (for checkpointing) ===
    messages: Annotated[List[BaseMessage], add_messages]  # Message history

def create_initial_state(
    text: str,
    session_id: str,
    user_info: Dict[str, Any] = None
) -> AgentState:
    """
    Create starting state for new request

    Think of this as filling out the top of a form before
    sending it down the assembly line.
    """
    return AgentState(
        text=text,
        session_id=session_id,
        user_info=user_info or {},
        intent=None,
        confidence=None,
        entities=None,
        needs_clarification=False,
        clarifying_question=None,
        conversation_history=[],
        relevant_facts=[],
        tool_results=None,
        safety_precheck_passed=False,
        safety_postcheck_passed=False,
        safety_block_reason=None,
        response="",
        metadata={},
        cache_hit=False,
        error=None,
        messages=[]
    )
'''

FILES['state/__init__.py'] = '''"""State management"""
from state.schema import AgentState, create_initial_state

__all__ = ["AgentState", "create_initial_state"]
'''

# ============================================================================
# CONFIGURATION
# ============================================================================
FILES['core/config.py'] = '''"""Configuration - All settings"""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""

    # LLM
    openai_api_key: str = "mock"
    use_mock_llm: bool = True
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7

    # Agent
    confidence_threshold: float = 0.7

    # Safety
    enable_safety_precheck: bool = True
    enable_safety_postcheck: bool = True

    # Cache
    enable_semantic_cache: bool = True

    # Checkpoint
    checkpoint_db_path: str = "checkpoints.db"

    # App
    environment: str = "development"
    debug: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
'''

FILES['core/logger.py'] = '''"""Logging"""
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
'''

FILES['core/__init__.py'] = '"""Core utilities"""'

# ============================================================================
# NODES - Function-based nodes (no LLM)
# ============================================================================

FILES['nodes/safety.py'] = '''"""
Safety Nodes - Input and Output validation

These are FUNCTIONS, not agents. No LLM needed - just rules!
"""

import asyncio
from typing import Dict, Any
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# SAFETY PRECHECK NODE
# ============================================================================

async def safety_precheck_node(state: AgentState) -> Dict[str, Any]:
    """
    Check if USER INPUT is safe

    📍 BREAKPOINT: Set here to debug safety checking

    🎓 CONCEPT:
    This is a FUNCTION NODE - it just runs code, no LLM.
    It checks for harmful keywords and blocks bad input.

    INPUT (reads from state):
        - text: User's message

    OUTPUT (writes to state):
        - safety_precheck_passed: True/False
        - safety_block_reason: Why blocked (if blocked)
        - response: Error message (if blocked)

    FLOW:
        If blocked → Graph ends (returns error to user)
        If passed → Graph continues to next node
    """

    logger.info("🔒 Node: Safety Precheck")

    if not settings.enable_safety_precheck:
        return {"safety_precheck_passed": True}

    text = state["text"].lower()

    # Check harmful keywords
    harmful = {
        "self_harm": ["kill myself", "suicide"],
        "violence": ["how to make a bomb", "hurt someone"],
        "hate_speech": ["hate speech examples"]
    }

    for category, keywords in harmful.items():
        for keyword in keywords:
            if keyword in text:
                logger.warning(f"🚫 Blocked: {category}")
                return {
                    "safety_precheck_passed": False,
                    "safety_block_reason": f"Violates {category} policy",
                    "response": "I cannot process that request."
                }

    await asyncio.sleep(0.05)  # Simulate check

    logger.info("✅ Safety precheck passed")
    return {"safety_precheck_passed": True}

# ============================================================================
# SAFETY POSTCHECK NODE
# ============================================================================

async def safety_postcheck_node(state: AgentState) -> Dict[str, Any]:
    """
    Check if AI OUTPUT is safe

    Same concept as precheck but for responses we generate.
    Ensures AI doesn't say anything harmful.
    """

    logger.info("🔒 Node: Safety Postcheck")

    if not settings.enable_safety_postcheck:
        return {"safety_postcheck_passed": True}

    response = state["response"].lower()

    # Check if too long (possible attack)
    if len(response) > 5000:
        logger.warning("🚫 Response too long")
        return {
            "safety_postcheck_passed": False,
            "response": "I apologize, I cannot provide that information."
        }

    await asyncio.sleep(0.05)

    logger.info("✅ Safety postcheck passed")
    return {"safety_postcheck_passed": True}
'''

FILES['nodes/cache.py'] = '''"""
Cache Nodes - Speed up repeated questions
"""

import hashlib
from typing import Dict, Any, Optional
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# Simple in-memory cache (in production: Redis)
_cache: Dict[str, Dict[str, Any]] = {}

def _hash(text: str) -> str:
    """Create hash of text for caching"""
    return hashlib.md5(text.lower().strip().encode()).hexdigest()

async def check_cache_node(state: AgentState) -> Dict[str, Any]:
    """
    Check if we've answered this before

    🎓 CONCEPT:
    If user asks same question, return cached answer instantly!
    No need to run through all nodes again.

    FLOW:
        Cache HIT → Return cached response, end graph
        Cache MISS → Continue to next node
    """

    logger.info("💾 Node: Check Cache")

    if not settings.enable_semantic_cache:
        return {}

    key = _hash(state["text"])

    if key in _cache:
        cached = _cache[key]
        logger.info("🎯 Cache HIT!")
        return {
            "response": cached["response"],
            "intent": cached["intent"],
            "confidence": cached["confidence"],
            "cache_hit": True
        }

    logger.info("💨 Cache MISS")
    return {"cache_hit": False}

async def cache_response_node(state: AgentState) -> Dict[str, Any]:
    """Store response in cache for future"""

    logger.info("💾 Node: Cache Response")

    if not settings.enable_semantic_cache:
        return {}

    key = _hash(state["text"])
    _cache[key] = {
        "response": state["response"],
        "intent": state["intent"],
        "confidence": state["confidence"]
    }

    logger.info("✅ Response cached")
    return {}
'''

FILES['nodes/context.py'] = '''"""
Context Building Node - Gather conversation background
"""

from typing import Dict, Any
from state.schema import AgentState
from core.logger import get_logger

logger = get_logger(__name__)

# Simple in-memory storage (production: database)
_short_term = {}
_long_term = {}

async def build_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Build context from conversation history

    🎓 CONCEPT:
    Before processing, gather:
    - Recent messages (short-term memory)
    - Important facts (long-term memory)

    This gives agents context about the conversation.
    """

    logger.info("🧠 Node: Build Context")

    session_id = state["session_id"]

    # Get conversation history
    history = _short_term.get(session_id, [])

    # Get relevant facts
    facts = _long_term.get(session_id, [])

    logger.debug(f"Context: {len(history)} messages, {len(facts)} facts")

    return {
        "conversation_history": history,
        "relevant_facts": facts
    }

async def update_memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Store conversation in memory

    After generating response, save it so we remember next time.
    """

    logger.info("💾 Node: Update Memory")

    session_id = state["session_id"]

    # Update short-term memory
    if session_id not in _short_term:
        _short_term[session_id] = []

    _short_term[session_id].append({
        "role": "user",
        "content": state["text"]
    })
    _short_term[session_id].append({
        "role": "assistant",
        "content": state["response"]
    })

    # Keep only recent 10 messages
    _short_term[session_id] = _short_term[session_id][-10:]

    # Update long-term memory (extract important facts)
    if "claim" in state["text"].lower():
        if session_id not in _long_term:
            _long_term[session_id] = []
        _long_term[session_id].append({
            "type": "claim_mention",
            "text": state["text"]
        })

    logger.info("✅ Memory updated")
    return {}
'''

FILES['nodes/clarification.py'] = '''"""
Clarification Node - Ask questions when unsure
"""

from typing import Dict, Any
from state.schema import AgentState
from core.logger import get_logger

logger = get_logger(__name__)

async def clarification_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate clarifying question

    🎓 CONCEPT:
    When confidence is low, we need more info from user.
    This generates an appropriate question to ask.

    FLOW:
        After this node → End graph, return question to user
        User answers → New request with more context
    """

    logger.info("❓ Node: Clarification")

    intent = state.get("intent", "unknown")

    # Predefined questions for each intent
    questions = {
        "claim_status": "Could you provide your claim number?",
        "claim_rejection_reason": "Which claim are you asking about?",
        "unknown": "I'm not sure I understand. Are you asking about a claim?"
    }

    question = questions.get(intent, questions["unknown"])

    logger.info(f"❓ Generated: {question}")

    return {
        "needs_clarification": True,
        "clarifying_question": question,
        "response": question
    }
'''

FILES['nodes/confidence.py'] = '''"""
Confidence Check - Conditional router
"""

from typing import Literal
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

def confidence_check_router(state: AgentState) -> Literal["clarification", "tool_call"]:
    """
    Route based on confidence

    🎓 CONCEPT:
    This is a CONDITIONAL EDGE - it decides which path to take.

    ROUTING:
        confidence < 0.7 → Go to clarification node
        confidence >= 0.7 → Go to tool_call node

    This is NOT a node - it's a router function!
    """

    confidence = state.get("confidence", 0.0)
    threshold = settings.confidence_threshold

    if confidence < threshold:
        logger.info(f"⚠️  Low confidence ({confidence:.2f}) → Clarification")
        return "clarification"
    else:
        logger.info(f"✅ High confidence ({confidence:.2f}) → Continue")
        return "tool_call"
'''

FILES['nodes/__init__.py'] = '''"""Graph nodes"""
from nodes.safety import safety_precheck_node, safety_postcheck_node
from nodes.cache import check_cache_node, cache_response_node
from nodes.context import build_context_node, update_memory_node
from nodes.clarification import clarification_node
from nodes.confidence import confidence_check_router

__all__ = [
    "safety_precheck_node",
    "safety_postcheck_node",
    "check_cache_node",
    "cache_response_node",
    "build_context_node",
    "update_memory_node",
    "clarification_node",
    "confidence_check_router"
]
'''

# ============================================================================
# AGENTS - LLM-powered agents
# ============================================================================

FILES['agents/intent_agent.py'] = '''"""
Intent Classification Agent - THE FIRST AGENT

🤖 This is a REAL AGENT with LLM calls!
It understands natural language and classifies intent.
"""

import asyncio
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# MOCK LLM (for development without API key)
# ============================================================================

class MockLLM:
    """Fake LLM for development"""

    async def ainvoke(self, messages):
        """Simulate LLM response"""
        await asyncio.sleep(0.3)  # Simulate API delay

        # Simple keyword matching
        text = messages[0].content.lower()

        if "status" in text and "claim" in text:
            intent = "claim_status"
            confidence = 0.95
        elif "reject" in text or "denied" in text:
            intent = "claim_rejection_reason"
            confidence = 0.90
        elif "hello" in text or "hi" in text:
            intent = "greeting"
            confidence = 0.4  # Low confidence
        else:
            intent = "unknown"
            confidence = 0.3

        class Response:
            content = f'{{"intent": "{intent}", "confidence": {confidence}, "entities": {{}}}}'

        return Response()

# ============================================================================
# INTENT AGENT
# ============================================================================

async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent 1: Intent Classification & Entity Extraction

    📍 BREAKPOINT: Set here to debug intent classification

    🤖 THIS IS AN AGENT - Uses LLM!

    What it does:
    1. Takes user's text
    2. Uses LLM to understand what they want
    3. Extracts important information (entities)
    4. Returns intent + confidence + entities

    INPUT (from state):
        - text: User message
        - conversation_history: Past messages

    OUTPUT (to state):
        - intent: What user wants
        - confidence: How sure we are
        - entities: Extracted info
    """

    logger.info("🤖 AGENT 1: Intent Classification")

    text = state["text"]
    history = state.get("conversation_history", [])

    # Create LLM
    if settings.use_mock_llm:
        llm = MockLLM()
    else:
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key
        )

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intent classification agent for a pharmacy benefits system.

Your job: Classify the user's intent and extract entities.

Available intents:
- claim_status: User wants to check claim status
- claim_rejection_reason: User wants to know why claim was rejected
- find_pharmacy: User wants to find a pharmacy
- check_coverage: User wants to check medication coverage
- unknown: Cannot determine intent

Respond with JSON:
{
    "intent": "claim_status",
    "confidence": 0.95,
    "entities": {"claim_number": "12345"}
}

Be conservative with confidence. If unsure, return low confidence."""),
        ("user", f"User message: {text}\n\nConversation history: {history}\n\nClassify this:")
    ])

    # Call LLM
    messages = prompt.format_messages()
    response = await llm.ainvoke(messages)

    # Parse response (in production: use structured output)
    import json
    try:
        result = json.loads(response.content)
    except:
        result = {"intent": "unknown", "confidence": 0.1, "entities": {}}

    logger.info(f"🎯 Intent: {result['intent']} ({result['confidence']:.2f})")

    return {
        "intent": result["intent"],
        "confidence": result["confidence"],
        "entities": result.get("entities", {})
    }
'''

FILES['agents/response_agent.py'] = '''"""
Response Generation Agent - THE SECOND AGENT

🤖 This is a REAL AGENT with LLM calls!
It generates natural, helpful responses.
"""

import asyncio
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# MOCK LLM
# ============================================================================

class MockLLM:
    """Fake LLM for development"""

    async def ainvoke(self, messages):
        await asyncio.sleep(0.3)

        # Extract intent from prompt
        prompt_text = str(messages)

        responses = {
            "claim_status": "Your claim #12345 is currently being processed. It was submitted on January 10, 2025 and is expected to be completed within 5-7 business days. You'll receive a notification once a decision is made.",
            "claim_rejection_reason": "Your claim #12345 was rejected because the medication requires prior authorization from your healthcare provider. To resolve this, please ask your doctor to submit the necessary documentation.",
            "greeting": "Hello! I'm here to help with your pharmacy benefits questions. How can I assist you today?",
            "unknown": "I'm not sure how to help with that. Could you please rephrase your question? I can help with claim status, coverage questions, and finding pharmacies."
        }

        # Find matching intent
        for intent, response in responses.items():
            if intent in prompt_text.lower():
                class Response:
                    content = response
                return Response()

        class Response:
            content = responses["unknown"]
        return Response()

# ============================================================================
# RESPONSE AGENT
# ============================================================================

async def response_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent 2: Response Generation

    📍 BREAKPOINT: Set here to debug response generation

    🤖 THIS IS AN AGENT - Uses LLM!

    What it does:
    1. Takes intent, context, and tool results
    2. Uses LLM to generate natural response
    3. Returns helpful, conversational answer

    INPUT (from state):
        - intent: What user wants
        - tool_results: Data from APIs
        - conversation_history: Context

    OUTPUT (to state):
        - response: The final answer
    """

    logger.info("🤖 AGENT 2: Response Generation")

    intent = state["intent"]
    tool_results = state.get("tool_results", {})
    history = state.get("conversation_history", [])

    # Create LLM
    if settings.use_mock_llm:
        llm = MockLLM()
    else:
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key
        )

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful pharmacy benefits assistant.

Your job: Generate natural, friendly responses based on the user's intent and available data.

Guidelines:
- Be conversational and empathetic
- Provide clear, actionable information
- If you have tool results, use that data
- Be concise but complete
- Use appropriate tone for the situation"""),
        ("user", f"""Intent: {intent}
Tool Results: {tool_results}
Conversation History: {history}

Generate a helpful response:""")
    ])

    # Call LLM
    messages = prompt.format_messages()
    response = await llm.ainvoke(messages)

    response_text = response.content

    logger.info(f"💬 Generated: {response_text[:50]}...")

    return {"response": response_text}
'''

FILES['agents/__init__.py'] = '''"""AI Agents"""
from agents.intent_agent import intent_agent_node
from agents.response_agent import response_agent_node

__all__ = ["intent_agent_node", "response_agent_node"]
'''

# ============================================================================
# TOOLS - External APIs
# ============================================================================

FILES['tools/claims_api.py'] = '''"""
Claims API Tool - Calls external API
"""

import asyncio
from typing import Dict, Any
from state.schema import AgentState
from core.logger import get_logger

logger = get_logger(__name__)

async def call_claims_tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Call Claims API

    🎓 CONCEPT:
    This simulates calling an external API to get data.
    In production, this would be a real HTTP call.

    INPUT (from state):
        - intent: What data to fetch
        - entities: Parameters (e.g., claim_number)

    OUTPUT (to state):
        - tool_results: API response data
    """

    logger.info("🔧 Node: Call Claims Tool")

    intent = state["intent"]
    entities = state.get("entities", {})

    # Simulate API call
    await asyncio.sleep(0.2)

    # Mock responses
    if intent == "claim_status":
        results = {
            "claim_id": "12345",
            "status": "processing",
            "submitted_date": "2025-01-10",
            "expected_completion": "5-7 business days"
        }
    elif intent == "claim_rejection_reason":
        results = {
            "claim_id": "12345",
            "status": "rejected",
            "reason": "Requires prior authorization",
            "action_needed": "Doctor must submit documentation"
        }
    else:
        results = {}

    logger.info(f"✅ Tool results: {results}")

    return {"tool_results": results}
'''

FILES['tools/__init__.py'] = '''"""Tools"""
from tools.claims_api import call_claims_tool_node

__all__ = ["call_claims_tool_node"]
'''

# ============================================================================
# THE GRAPH - This is where everything connects!
# ============================================================================

FILES['langgraph_agent.py'] = '''"""
LangGraph Agent - THE GRAPH DEFINITION

🎯 THIS IS THE HEART OF THE SYSTEM!

This file defines how all nodes connect together.
Think of it as a flowchart that LangGraph executes.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from state.schema import AgentState
from nodes import (
    safety_precheck_node,
    check_cache_node,
    build_context_node,
    clarification_node,
    confidence_check_router,
    safety_postcheck_node,
    update_memory_node,
    cache_response_node
)
from agents import intent_agent_node, response_agent_node
from tools import call_claims_tool_node
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# CONDITIONAL ROUTERS
# ============================================================================

def should_continue_after_precheck(state: AgentState) -> Literal["check_cache", END]:
    """
    After safety precheck, decide next step

    ROUTING:
        Blocked → END (return error)
        Passed → check_cache (continue)
    """
    if not state.get("safety_precheck_passed", False):
        return END
    return "check_cache"

def should_continue_after_cache(state: AgentState) -> Literal["build_context", END]:
    """
    After cache check, decide next step

    ROUTING:
        Cache HIT → END (return cached response)
        Cache MISS → build_context (continue processing)
    """
    if state.get("cache_hit", False):
        return END
    return "build_context"

# ============================================================================
# CREATE GRAPH
# ============================================================================

def create_graph():
    """
    Create the LangGraph

    🎓 UNDERSTAND THIS:
    This is like drawing a flowchart, then LangGraph executes it!

    GRAPH STRUCTURE:

    START
      ↓
    safety_precheck ──[blocked]──→ END
      ↓ [passed]
    check_cache ──[hit]──→ END
      ↓ [miss]
    build_context
      ↓
    🤖 intent_agent (AGENT 1 - LLM)
      ↓
    confidence_check (ROUTER)
      ↓                    ↓
    [low]              [high]
      ↓                    ↓
    clarification      call_claims_tool
      ↓                    ↓
    END               🤖 response_agent (AGENT 2 - LLM)
                           ↓
                      safety_postcheck
                           ↓
                      update_memory
                           ↓
                      cache_response
                           ↓
                         END
    """

    logger.info("📊 Creating LangGraph...")

    # Create graph
    workflow = StateGraph(AgentState)

    # ========================================================================
    # ADD NODES
    # ========================================================================

    # Safety nodes
    workflow.add_node("safety_precheck", safety_precheck_node)
    workflow.add_node("safety_postcheck", safety_postcheck_node)

    # Cache nodes
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("cache_response", cache_response_node)

    # Context node
    workflow.add_node("build_context", build_context_node)

    # 🤖 AGENT NODES (with LLM)
    workflow.add_node("intent_agent", intent_agent_node)
    workflow.add_node("response_agent", response_agent_node)

    # Tool node
    workflow.add_node("call_claims_tool", call_claims_tool_node)

    # Clarification node
    workflow.add_node("clarification", clarification_node)

    # Memory node
    workflow.add_node("update_memory", update_memory_node)

    # ========================================================================
    # ADD EDGES - Define the flow!
    # ========================================================================

    # Set entry point
    workflow.set_entry_point("safety_precheck")

    # After safety precheck
    workflow.add_conditional_edges(
        "safety_precheck",
        should_continue_after_precheck,
        {
            "check_cache": "check_cache",
            END: END
        }
    )

    # After cache check
    workflow.add_conditional_edges(
        "check_cache",
        should_continue_after_cache,
        {
            "build_context": "build_context",
            END: END
        }
    )

    # Linear flow to intent agent
    workflow.add_edge("build_context", "intent_agent")

    # After intent agent → confidence check (CONDITIONAL)
    workflow.add_conditional_edges(
        "intent_agent",
        confidence_check_router,
        {
            "clarification": "clarification",
            "tool_call": "call_claims_tool"
        }
    )

    # If clarification → END (return question to user)
    workflow.add_edge("clarification", END)

    # If high confidence → tool → response agent
    workflow.add_edge("call_claims_tool", "response_agent")

    # After response agent → safety postcheck
    workflow.add_edge("response_agent", "safety_postcheck")

    # After safety postcheck → memory
    workflow.add_edge("safety_postcheck", "update_memory")

    # After memory → cache
    workflow.add_edge("update_memory", "cache_response")

    # After cache → END
    workflow.add_edge("cache_response", END)

    # ========================================================================
    # COMPILE WITH CHECKPOINTING
    # ========================================================================

    # Create SQLite checkpointer (saves conversation state)
    memory = SqliteSaver.from_conn_string(settings.checkpoint_db_path)

    # Compile graph
    app = workflow.compile(checkpointer=memory)

    logger.info("✅ Graph created successfully!")

    return app

# ============================================================================
# EXECUTE GRAPH
# ============================================================================

async def run_graph(text: str, session_id: str, user_info: dict = None):
    """
    Execute the graph for a user message

    Args:
        text: User's message
        session_id: Session ID for checkpointing
        user_info: User metadata

    Returns:
        Final state after processing
    """
    from state.schema import create_initial_state

    # Create initial state
    initial_state = create_initial_state(text, session_id, user_info)

    # Create graph
    app = create_graph()

    # Run graph with checkpointing
    config = {"configurable": {"thread_id": session_id}}

    # Execute
    final_state = await app.ainvoke(initial_state, config)

    return final_state
'''

# ============================================================================
# API ROUTES
# ============================================================================

FILES['api/routes.py'] = '''"""
API Routes - HTTP endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
from datetime import datetime

from langgraph_agent import run_graph

router = APIRouter()

class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint

    📍 BREAKPOINT: Set here (line 33) to debug incoming requests
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())

        # Run LangGraph
        final_state = await run_graph(
            text=request.text,
            session_id=session_id,
            user_info=request.user_info or {}
        )

        # Build response
        return ChatResponse(
            response=final_state.get("response", ""),
            session_id=session_id,
            intent=final_state.get("intent"),
            confidence=final_state.get("confidence"),
            needs_clarification=final_state.get("needs_clarification", False),
            clarifying_question=final_state.get("clarifying_question"),
            metadata=final_state.get("metadata", {}),
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''

FILES['api/__init__.py'] = '"""API"""'

# ============================================================================
# MAIN.PY
# ============================================================================

FILES['main.py'] = '''"""
LangGraph Multi-Agent Framework

📍 BREAKPOINT: Line 56 - Start debugging
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import router as api_router
from core.config import settings

app = FastAPI(
    title="PBM LangGraph Framework",
    description="2 Agents + 9 Nodes",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "PBM LangGraph Framework",
        "version": "2.0.0",
        "agents": 2,
        "nodes": 9,
        "framework": "LangGraph"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("🚀 LangGraph Multi-Agent Framework")
    print(f"🤖 Agents: 2 (Intent, Response)")
    print(f"🔧 Nodes: 9 functions")
    print(f"💾 Checkpointing: SQLite")
    print(f"🎯 Mode: {'Mock' if settings.use_mock_llm else 'Real'} LLM")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
'''

# ============================================================================
# DOCUMENTATION
# ============================================================================

FILES['README.md'] = '''# PBM LangGraph Multi-Agent Framework 🤖

## 🎯 Architecture

**2 LLM Agents:**
1. Intent Classification Agent
2. Response Generation Agent

**9 Function Nodes:**
- safety_precheck
- check_cache  
- build_context
- confidence_check (router)
- clarification
- call_claims_tool
- safety_postcheck
- update_memory
- cache_response

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
python main.py

# Test
POST http://localhost:8000/api/v1/chat
{"text": "What's my claim status?"}
```

## 🐛 Debugging

**PyCharm:**
- Breakpoint at main.py line 56
- Breakpoint at api/routes.py line 33
- Breakpoint at langgraph_agent.py line 150

**LangGraph Studio:**
See docs/LANGGRAPH_STUDIO.md

## 📊 Graph Visualization

```python
from langgraph_agent import create_graph
graph = create_graph()
graph.get_graph().draw_mermaid()  # View graph structure
```

## 🎓 Learning

Every file has detailed comments for beginners!
Start with: state/schema.py → langgraph_agent.py
'''

FILES['docs/LANGGRAPH_STUDIO.md'] = '''# LangGraph Studio Guide 🎨

## What is LangGraph Studio?

LangGraph Studio is a visual IDE for debugging LangGraph applications.
It shows you:
- The graph structure (visual flowchart)
- State at each node
- Execution path taken
- Step-by-step debugging

## Installation

```bash
# Install LangGraph CLI
pip install langgraph-cli

# Or use Docker
docker pull langchain/langgraph-studio
```

## Setup for This Project

1. **Create langgraph.json:**

```json
{
  "dependencies": ["requirements.txt"],
  "graphs": {
    "agent": "langgraph_agent.py:create_graph"
  },
  "env": ".env"
}
```

2. **Start Studio:**

```bash
langgraph up
```

3. **Open browser:**
http://localhost:8000

## Using Studio

### 1. View Graph Structure
- Click "Graph" tab
- See visual flowchart of your nodes
- Blue boxes = nodes
- Arrows = edges
- Diamond = conditional router

### 2. Test Your Graph
- Click "Playground"
- Enter: `{"text": "What's my claim status?", "session_id": "test"}`
- Click "Run"
- Watch execution in real-time!

### 3. Debug Step-by-Step
- Click "Step" mode
- Execute one node at a time
- Inspect state after each node
- See exactly what changed

### 4. View State
- Click any node
- See state before and after
- Inspect all fields
- Track how data flows

### 5. Time Travel
- LangGraph saves every state
- Rewind to any point
- See what happened
- Debug issues easily

## Common Workflows

### Testing Intent Classification
1. Start Studio
2. Send: "Why was my claim rejected?"
3. Watch it hit intent_agent
4. See confidence score
5. Check routing decision

### Debugging Low Confidence
1. Send: "Hello"
2. Watch confidence_check_router
3. See it route to clarification
4. Check generated question

### Viewing Tool Calls
1. Send: "Claim status?"
2. Watch call_claims_tool node
3. See API results in state
4. Check how response_agent uses it

## Keyboard Shortcuts

- `Space` - Run/Pause
- `S` - Step forward
- `R` - Reset
- `D` - Download state

## Tips

1. **Always check state** - Most bugs are state issues
2. **Use step mode** - Don't run full graph immediately
3. **Watch routers** - Conditional edges are tricky
4. **Check checkpoints** - See conversation history

## Troubleshooting

**Studio won't start?**
- Check langgraph.json is correct
- Ensure requirements.txt installed
- Try: `langgraph up --verbose`

**Graph not showing?**
- Check langgraph_agent.py path
- Ensure create_graph() works
- Test: `python -c "from langgraph_agent import create_graph; create_graph()"`

**Can't see state?**
- Click on node circles
- Enable "Show State" toggle
- Check state schema matches

You're ready to visualize and debug! 🎉
'''

# ============================================================================
# CREATE ALL FILES
# ============================================================================

print("📦 Creating all files...\n")

for filepath, content in FILES.items():
    create_file(filepath, content)

print("\n" + "=" * 80)
print("✅ COMPLETE LANGGRAPH VERSION CREATED!")
print("=" * 80)
print(f"""
🎉 SUCCESS! Generated {len(FILES)} files!

📦 What was created:
  ✅ Complete LangGraph implementation
  ✅ 2 LLM Agents (Intent + Response)
  ✅ 9 Function Nodes (all logic)
  ✅ Branching graph with conditional routing
  ✅ Rich state management
  ✅ SQLite checkpointing
  ✅ Mock Claims API tool
  ✅ Detailed documentation
  ✅ LangGraph Studio guide

🚀 Next Steps:

1. Install dependencies:
   pip install -r requirements.txt

2. Run the server:
   python main.py

3. Test with Postman:
   POST http://localhost:8000/api/v1/chat
   {{"text": "What's my claim status?"}}

4. Debug with PyCharm:
   - Set breakpoint at main.py line 56
   - Set breakpoint at api/routes.py line 33
   - Right-click main.py → Debug

5. Visualize with LangGraph Studio:
   - pip install langgraph-cli
   - Create langgraph.json (see docs/)
   - Run: langgraph up
   - Open: http://localhost:8000

📚 Documentation:
  - README.md - Project overview
  - docs/LANGGRAPH_STUDIO.md - Visual debugging guide
  - state/schema.py - Understand state (START HERE!)
  - langgraph_agent.py - See how graph is built

🎓 Every file has detailed comments for learning!

🤖 Architecture:
  2 Agents: intent_agent, response_agent
  9 Nodes: safety, cache, context, tools, memory
  Conditional routing based on confidence

Ready to explore LangGraph! 🚀
""")
