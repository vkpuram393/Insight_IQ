"""
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
        prompt_text = str(messages).lower()

        responses = {
            "claim_status": "Your claim #12345 is currently being processed. It was submitted on January 10, 2025 and is expected to be completed within 5-7 business days.",
            "claim_rejection_reason": "Your claim #12345 was rejected because the medication requires prior authorization from your healthcare provider.",
            "greeting": "Hello! I'm here to help with your pharmacy benefits questions. How can I assist you today?",
            "unknown": "I'm not sure how to help with that. Could you please rephrase your question?"
        }

        # Find matching intent
        for intent, response in responses.items():
            if intent in prompt_text:
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

    intent = state.get("intent", "unknown")  # guard
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
        ("system", "You are a helpful pharmacy benefits assistant."),
        ("user", "Intent: {intent}\nTool Results: {tool_results}\nConversation History: {history}\n\nGenerate a helpful response:")
    ])

    # Call LLM
    messages = prompt.format_messages(
        intent=intent,
        tool_results=tool_results,
        history=history
    )
    response = await llm.ainvoke(messages)

    response_text = response.content

    logger.info(f"💬 Generated: {response_text[:50]}...")

    return {"response": response_text}
