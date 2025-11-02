"""
Intent Classification Agent - THE FIRST AGENT

🤖 This is a REAL AGENT with LLM calls!
It understands natural language and classifies intent.
"""

import asyncio
import re
import json
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from state.schema import AgentState
from core.config import settings
from core.logger import get_logger

# Use base logger
logger = get_logger(__name__)

# ============================================================================
# MOCK LLM (for development without API key)
# ============================================================================

class MockLLM:
    """Fake LLM for development"""

    async def ainvoke(self, messages: List[Any]):
        await asyncio.sleep(0.05)  # Simulate small latency

        # Extract last user/human message content
        user_msg = ""
        for m in messages:
            role = getattr(m, "role", getattr(m, "type", ""))
            if role in ("user", "human"):
                user_msg = m.content
        text = user_msg.lower()

        # Simple keyword matching for mock intent
        if "status" in text and "claim" in text:
            intent = "claim_status"
            confidence = 0.92
        elif "reject" in text or "denied" in text:
            intent = "claim_rejection_reason"
            confidence = 0.88
        elif any(greet in text for greet in ["hello", "hi", "hey"]):
            intent = "greeting"
            confidence = 0.45
        else:
            intent = "unknown"
            confidence = 0.30

        # Naive entity extraction (claim number: 4-10 digits)
        entities: Dict[str, Any] = {}
        claim_match = re.search(r"\b\d{4,10}\b", text)
        if claim_match and "claim" in text:
            entities["claim_number"] = claim_match.group(0)

        class Response:
            content = json.dumps({
                "intent": intent,
                "confidence": confidence,
                "entities": entities
            })
        return Response()

# ============================================================================
# INTENT AGENT NODE
# ============================================================================

async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    """Classify user intent and extract entities."""
    logger.info("🤖 AGENT 1: Intent Classification")

    text = state["text"]
    history = state.get("conversation_history", [])

    # Normalize history into readable string
    if isinstance(history, list):
        if history and isinstance(history[0], dict):
            history_str = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in history)
        else:
            history_str = "\n".join(str(h) for h in history)
    else:
        history_str = str(history)

    # Select LLM (mock vs real)
    if settings.use_mock_llm:
        llm = MockLLM()
    else:
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
        )

    # Build raw system prompt then escape braces to avoid KeyError during format
    raw_system_prompt = (
        "You are an intent classification agent for a pharmacy benefits system.\n\n"
        "Your job: Classify the user's intent and extract entities.\n\n"
        "Available intents:\n"
        "- claim_status: User wants to check claim status\n"
        "- claim_rejection_reason: User wants to know why claim was rejected\n"
        "- find_pharmacy: User wants to find a pharmacy\n"
        "- check_coverage: User wants to check medication coverage\n"
        "- unknown: Cannot determine intent\n\n"
        "Respond ONLY with JSON like:\n"
        '{"intent": "claim_status", "confidence": 0.95, "entities": {"claim_number": "12345"}}\n\n'
        "Be conservative with confidence; if unsure use lower confidence."
    )
    # Escape braces for Python .format safety
    system_prompt = raw_system_prompt.replace('{', '{{').replace('}', '}}')

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", (
            "User message: {user_text}\n\n"
            "Conversation history:\n{conversation_history}\n\n"
            "Classify this."
        )),
    ])

    # Format messages
    messages = prompt.format_messages(
        user_text=text,
        conversation_history=history_str if history_str.strip() else "(none)"
    )

    # Invoke LLM
    response = await llm.ainvoke(messages)

    # Parse JSON safely
    try:
        result = json.loads(response.content)
    except Exception:
        result = {"intent": "unknown", "confidence": 0.1, "entities": {}}

    intent = result.get("intent", "unknown")
    confidence = float(result.get("confidence", 0.1))
    entities = result.get("entities") or {}

    logger.info(f"🎯 Intent: {intent} ({confidence:.2f}) | Entities: {entities}")

    return {
        "intent": intent,
        "confidence": confidence,
        "entities": entities,
    }
