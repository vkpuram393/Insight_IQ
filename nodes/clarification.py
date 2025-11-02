"""
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
        "response": question,
        "metadata": {**state.get("metadata", {}), "clarification": True}
    }
