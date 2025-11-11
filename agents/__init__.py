"""AI Agents"""

from core.config import settings

# Use CVS Classifier or Original based on config
if settings.use_cvs_intent_classifier:
    # CVS Production Classifier (NO LangChain required!)
    from agents.cvs_intent_agent_node import cvs_intent_agent_node as intent_agent_node
    try:
        from agents.response_agent import response_agent_node
    except ImportError:
        # Create a simple mock response agent if LangChain not available
        async def response_agent_node(state):
            return {"response": f"Mock response for intent: {state.get('intent')}"}
    
    __all__ = ["intent_agent_node", "response_agent_node"]
    
else:
    # Original agents (require LangChain dependencies)
    try:
        from agents.intent_agent import intent_agent_node
        from agents.response_agent import response_agent_node
        __all__ = ["intent_agent_node", "response_agent_node"]
    except ImportError:
        # If LangChain not installed, skip (CVS classifier doesn't need it)
        __all__ = []

# CVS Intent Classifier (NO dependencies required)
# These can be imported directly:
#   from agents.cvs_intent_classifier import get_cvs_intent_classifier
#   from agents.entity_extractor import get_entity_extractor
#   from agents.intent_classifier_wrapper import classify_intent_unified
#   from agents.cvs_intent_agent_node import cvs_intent_agent_node
