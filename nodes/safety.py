"""
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
    # we will be using Gemini filter here
    harmful = {
        "self_harm": ["kill", "suicide"],
        "violence": ["bomb", "hurt"],
        "hate_speech": ["hate speech examples here"]
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
