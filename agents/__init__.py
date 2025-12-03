"""AI Agents"""

# CVS Production Classifier
from agents.extended_intent_agent_node import extended_intent_agent_node as intent_agent_node

try:
    from agents.response_agent import response_agent_node
except ImportError:
    # Create a simple mock response agent if dependencies not available
    async def response_agent_node(state):
        return {"response": f"Mock response for intent: {state.get('intent')}"}

__all__ = ["intent_agent_node", "response_agent_node"]

# Intent Classifier can be imported directly:
#   from classifiers.keyword_classifier import get_cvs_intent_classifier
#   from classifiers.embedded_classifier import CVSIntentEmbedded
#   from classifiers.intent_classifier_wrapper import classify_intent_unified
#   from utils.entity_extractor import get_entity_extractor
