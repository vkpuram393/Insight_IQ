"""
Response Generation Agent - THE SECOND AGENT

🤖 This is a REAL AGENT with LLM calls!
It generates natural, helpful responses using Google Gemini.

Architecture:
- Class-based design (ResponseAgent)
- Supports both Mock LLM (development) and Real Gemini (production)
- Simple response generation (complete response returned)
- Follows existing patterns: logging, telemetry, error handling
- Uses ChatPromptTemplate for structured prompt management

"""

import asyncio
import traceback
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from state.schema import AgentState
from config.config import settings
from core.logger import get_logger
from core.errors.models import create_internal_error, create_llm_error
from core.logging_context import extract_logging_context, log_state_snapshot
from persistence import PersistenceStoreFactory
from services.llm_connection import client as gemini_client, GenerateRequest, _generate_core

logger = get_logger(__name__)

# ============================================================================
# RESPONSE AGENT CLASS
# ============================================================================

class ResponseAgent:
    """
    Handles response generation with Gemini LLM.
    
    Responsibilities:
    - System prompt management (single default)
    - User prompt construction from state (uses ChatPromptTemplate)
    - LLM invocation with Gemini (real or mock)
    - Response generation (uses GenerateRequest + _generate_core)
    """
    
    def __init__(self):
        """Initialize the response agent with Gemini client."""
        self.logger = get_logger(__name__)
        self.client = gemini_client
    
    def _get_system_prompt(self) -> str:
        """
        Get the default system prompt for pharmacy claims assistant.
        
        This prompt instructs the LLM on response format and style,
        including structured formatting for Paid/Rejected claims.
        
        Returns:
            str: Complete system prompt
        """
        return """# Pharmacy Claim Assistant System Prompt

You are a specialized pharmacy claims assistant with expertise in interpreting and explaining claim information. Your role is to provide clear, concise information about pharmacy claims based on their status (Paid, Rejected, or Reversed).

## Response Structure Guidelines

1. Always begin with a one-line or two-line summary of the claim status and key information.
2. Format your responses in a concise, easy-to-scan format using bullet points, short tables, or brief sections.
3. Avoid wordiness and unnecessary explanations.
4. Tailor information based on claim status as follows:
5. Strictly follow the format and structure of the example response formats below.

### For PAID or REVERSED claims, include:
- One-line summary (including claim date, drug name, and status)
- Member demographics (basic info only)
- Financial information (patient cost, copay, deductible)
- Accumulation details (toward deductible, out-of-pocket maximum)
- Drug information (name, dosage, quantity, days supply)
- Pharmacy information (name, location, provider ID)

### For REJECTED claims, include:
- One-line summary (including claim date, drug name, and rejection reason)
- Drug information (name, dosage, quantity)
- Member demographics (basic info only)
- Pharmacy information (name, location, provider ID)
- Rejection code(s) and corresponding message(s)
- Additional rejection context (if available)
- Provide next steps to resolve the rejected claim (if available) - this is very important

**CRITICAL FOR REJECTED CLAIMS**: When REJECT ANALYSIS data is provided in the prompt, ALWAYS prioritize using the detailed explanations, reasons, and actions from the REJECT ANALYSIS section over any basic rejection information in the CLAIM DATA. The REJECT ANALYSIS contains expert-level, persona-specific explanations that are far more valuable than raw claim rejection codes. Use the REJECT ANALYSIS descriptions, reasons, and recommended actions instead of generic claim rejection messages.

## Handling Questions

### Initial Questions
- If the user asks about a specific aspect of a claim (e.g., "What was my copay for Lisinopril?"), provide only the relevant information in the same concise format.
- If the user asks a general question about a claim, provide the full structured response based on claim status.
- Always maintain the same concise, structured format regardless of question type.

### Follow-up Questions
- For follow-up questions, focus only on the specific information requested.
- Maintain the same bullet point or tabular format for consistency.
- Reference previous information when relevant but avoid repeating all details.
- If the follow-up question relates to a different aspect of the same claim, provide only the newly requested information.
- If clarification is needed about which claim is being referenced, ask a brief clarifying question.

## Response Style

- Use technical terminology appropriate for pharmacy professionals
- Present information in a structured, scannable format
- Keep explanations brief and factual
- When uncertain about specific claim details, acknowledge limitations rather than providing potentially incorrect information
- For all responses, maintain the same structured, concise format
- Alway follow table format as example below, never use any other format

## Example table Formats
   | Category    | Before  | After   | Remaining |
   |-------------|---------|---------|-----------|
   | Individual  | 1000.0  | 1000.0  | 1000.0    |
   | Family      | 2000.0  | 2000.0  | 2000.0    |
   | This Claim  | 0.0     | -       | -         |

   
## Example Response Formats

For a paid claim:

SUMMARY: Atorvastatin 40mg claim processed and paid on 05/15/2023.

FINANCIAL:
• Patient paid: $10.00 copay
• Plan paid: $45.75
• Accumulation: $10.00 applied to annual out-of-pocket

DRUG:
• Atorvastatin 40mg tablet
• Quantity: 30
• Days supply: 30
• NDC: 12345-6789-10

MEMBER: John Doe (ID: JD123456789)
PHARMACY: CVS Pharmacy #1234 (NPI: 1234567890)

For a rejected claim:

SUMMARY: Atorvastatin 40mg claim rejected on 05/15/2023 due to refill too soon.

DRUG:
• Atorvastatin 40mg tablet
• Quantity: 30
• Days supply: 30

MEMBER: John Doe (ID: JD123456789)
PHARMACY: CVS Pharmacy #1234 
NPI: 1234567890

REJECTION:
• Code: 79
• Message: Refill Too Soon
• Details: Previous fill on 05/01/2023 with 30-day supply. Next fill available 05/31/2023.

For a specific follow-up question about financial details:

FINANCIAL:
• Patient paid: $10.00 copay
• Plan paid: $45.75
• Accumulation: $10.00 applied to annual out-of-pocket

For a specific initial question about rejection reason:

SUMMARY: Atorvastatin 40mg claim rejected on 05/15/2023.

REJECTION:
• Code: 79
• Message: Refill Too Soon
• Details: Previous fill on 05/01/2023 with 30-day supply. Next fill available 05/31/2023.

**NEXT STEPS:**
• [Clear actions member can take]
• [Contact information for additional help]


Remember to maintain this structured, concise format for all responses, including both initial and follow-up questions."""
    
    def _build_user_prompt(self, state: AgentState) -> str:
        """
        Build user prompt from state data using ChatPromptTemplate.
        
        Uses LangChain's ChatPromptTemplate for structured prompt management,
        following pattern from agents/intent_agent.py.
        
        Args:
            state: Current agent state with user query, intent, and tool results
            
        Returns:
            str: Formatted prompt for LLM
        """
        # Extract data from state
        user_text = state.get("text", "")
        intent = state.get("intent", "unknown")
        tool_results = state.get("tool_results")
        history = state.get("conversation_history", [])
        
        # Format tool results and history
        claim_data = self._format_tool_results(tool_results) if tool_results else "No claim data available"
        history_str = self._format_conversation_history(history) if history else "No previous conversation"
        
        # Use ChatPromptTemplate for structured prompt (following intent_agent.py pattern)
        # This is better than string concatenation - cleaner and more maintainable
        prompt_template = ChatPromptTemplate.from_messages([
            ("user", """USER QUERY: {user_query}

INTENT: {intent}

=== CLAIM DATA ===
{claim_data}

=== CONVERSATION HISTORY ===
{conversation_history}

Please provide a helpful and factual response following the format guidelines in your system instructions.""")
        ])
        
        # Format the template with actual values
        messages = prompt_template.format_messages(
            user_query=user_text,
            intent=intent,
            claim_data=claim_data,
            conversation_history=history_str
        )
        
        # Extract the formatted text from the message
        # ChatPromptTemplate returns a list of messages, we need the content
        return messages[0].content
    
    def _format_tool_results(self, tool_results: Dict[str, Any]) -> str:
        """
        Format tool results (ToolResult structure) for LLM.
        
        Follows pattern from tools/claims_api.py for ToolResult handling.
        Uses serialization helpers from utils/serialization.py for consistent formatting.
        
        Real ToolResult structure:
        {
            "tool_name": "get_claim_list",
            "status": "success",
            "data": {
                "claims": [{...claim object with reject codes in messages...}],
                "success": True,
                "message": "...",
                "totalCount": 1
            },
            "execution_time_ms": 4381.1,
            ...
        }
        
        The LLM is smart enough to find reject codes within the claim data structure.
        System prompt instructs it to prioritize reject analysis information.
        
        Args:
            tool_results: ToolResult dictionary from claims API
            
        Returns:
            str: JSON-formatted string for LLM consumption
        """
        try:
            # Extract the data field (contains actual claim data)
            data = tool_results.get("data", {})
            
            # Use standard library json for pretty printing (indent for LLM readability)
            # Note: to_json() from serialization.py is for Pydantic models
            # For plain dicts, json.dumps is appropriate
            import json
            return json.dumps(data, indent=2)
            
        except Exception as e:
            self.logger.error(f"❌ Error formatting tool results: {e}")
            # Fallback: stringify the whole thing
            return str(tool_results)
    
    def _format_conversation_history(self, history: list) -> str:
        """
        Format conversation history for context.
        
        Follows pattern from agents/intent_agent.py.
        
        Args:
            history: List of conversation messages
            
        Returns:
            str: Formatted conversation history
        """
        if not history:
            return "(No previous conversation)"
        
        formatted = []
        # Take last N messages for context (configurable via settings.conversation_history_limit)
        limit = settings.conversation_history_limit
        for msg in history[-limit:]:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted.append(f"{role.upper()}: {content}")
            else:
                formatted.append(str(msg))
        
        return "\n".join(formatted)
    
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Generate response using Gemini LLM.
        
        Uses GenerateRequest and _generate_core from core/llm_connection.py
        following the established pattern in the codebase.
        
        Args:
            system_prompt: System instructions for LLM
            user_prompt: User query and context
            
        Returns:
            str: Complete response text from Gemini
            
        Raises:
            Exception: If generation fails (caught by caller)
        """
        try:
            # Use the established pattern from core/llm_connection.py
            # This ensures consistency with other LLM calls in the system
            req = GenerateRequest(
                prompt=user_prompt,
                system_instruction=system_prompt,
                temperature=settings.llm_temperature,
                top_p=settings.top_p,
                max_output_tokens=settings.max_output_tokens,
                model=settings.llm_model
            )
            
            self.logger.info("🔮 Calling Gemini...")
            response = _generate_core(req)
            
            self.logger.info(f"✅ Response received: {len(response.text)} chars")
            return response.text
            
        except Exception as e:
            self.logger.error(f"🚨 Generation error: {e}")
            raise


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

async def response_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent 2: Response Generation

    📍 BREAKPOINT: Set here to debug response generation

    🤖 THIS IS AN AGENT - Uses Gemini LLM!
    

    What it does:
    1. Takes intent, context, and tool results from state
    2. Uses Gemini LLM to generate natural response (or mock for development)
    3. Generates complete response
    4. Returns helpful, conversational answer

    INPUT (from state):
        - text: User's query
        - intent: What user wants
        - tool_results: Data from Claims API (ToolResult structure)
        - conversation_history: Context from previous turns
        - session_id: For logging/telemetry
        - uuid: Request ID for tracing
        - user_info: User metadata

    OUTPUT (to state):
        - response: The final answer to user
        - error: Error message if generation failed
        - metadata: Updated with error info if applicable
    """
    node_name = "response_agent"
    session_id = state.get("session_id", "unknown")
    request_id = state.get("uuid")
    user_id = state.get("user_info", {}).get("user_id")
    
    # Extract logging context (pattern from nodes/orchestrator.py)
    log_ctx = extract_logging_context(state)
    persistence_store = None
    
    try:
        logger.info("🤖 AGENT 2: Response Generation")
        logger.info(f"⚙️ LLM Mode: {'Mock' if settings.use_mock_llm else 'Real Gemini'}")

        # Log node entry for audit trail (pattern from tools/claims_api.py)
        if log_ctx and settings.enable_telemetry:
            persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
            await persistence_store.log_audit(
                session_id=log_ctx.get("session_id"),
                request_id=log_ctx.get("request_id"),
                user_id=log_ctx.get("user_id"),
                node_name=node_name,
                event_type="node_entry",
                data={"node": node_name, "intent": state.get("intent", "unknown")}
            )
        
        # Check if mock mode (for development only) 
        if settings.use_mock_llm:
            logger.info("⚙️ Using Mock LLM (development mode)")
            logger.info("💡 Set USE_MOCK_LLM=false in .env for real Gemini")
            result = await _mock_response(state)
            await log_state_snapshot(state, node_name, result)
            return result
        
        # === REAL LLM PATH ===
        # Initialize response agent
        logger.info("🔧 Initializing Response Agent with Gemini...")
        agent = ResponseAgent()
        
        # Get system prompt
        system_prompt = agent._get_system_prompt()
        logger.debug(f"📋 System prompt: {len(system_prompt)} characters")
        
        # Build user prompt from state
        user_prompt = agent._build_user_prompt(state)
        logger.debug(f"📋 User prompt: {len(user_prompt)} characters")
        logger.debug(f"📋 Intent: {state.get('intent', 'unknown')}")
        
        # Generate response using Gemini
        logger.info("🔮 Generating response with Gemini...")
        
        # Call generation method (synchronous but safe to call from async context)
        response_text = agent.generate_response(system_prompt, user_prompt)
        
        # Validate response
        if not response_text or not response_text.strip():
            logger.warning("⚠️ Empty response received from Gemini")
            response_text = "I apologize, but I received an empty response. Please try again."
        
        logger.info(f"✅ Response generated: {len(response_text)} chars")
        logger.info(f"💬 Preview: {response_text[:100]}...")
        
        # Log successful generation (telemetry pattern from tools/claims_api.py)
        if log_ctx and persistence_store:
            await persistence_store.log_audit(
                session_id=log_ctx.get("session_id"),
                request_id=log_ctx.get("request_id"),
                user_id=log_ctx.get("user_id"),
                node_name=node_name,
                event_type="response_generated",
                data={
                    "response_length": len(response_text),
                    "model": settings.llm_model,
                    "temperature": settings.llm_temperature
                }
            )
        
        result = {"response": response_text}
        await log_state_snapshot(state, node_name, result)
        return result
        
    except Exception as e:
        tb = traceback.format_exc()
        
        # Classify error 
        if "gemini" in str(e).lower() or "genai" in str(e).lower() or "llm" in str(e).lower() or "api" in str(e).lower():
            error = create_llm_error(
                error_message=str(e),
                session_id=session_id
            )
        else:
            error = create_internal_error(
                error_message=f"Response generation failed: {str(e)}",
                stacktrace=tb,
                session_id=session_id,
                node_name=node_name
            )
        
        # Log exception to persistence store (pattern from agents/intent_agent.py)
        if persistence_store is None:
            persistence_store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
        
        await persistence_store.log_exception(
            error_code=error.error_code.value,
            category=error.category.value,
            severity=error.severity.value,
            message=error.message,
            user_message=error.user_message,
            session_id=session_id,
            request_id=request_id,
            node_name=node_name,
            stacktrace=error.stacktrace or tb,
            metadata=error.metadata,
            user_id=user_id
        )
        
        logger.error(f"🚨 Exception in response agent: {e}\n{tb}")
        
        # Return graceful error state (pattern from agents/intent_agent.py)
        return {
            "error": error.user_message,
            "response": error.user_message,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }


# ============================================================================
# MOCK LLM (for development only)
# ============================================================================

async def _mock_response(state: AgentState) -> Dict[str, Any]:
    """
    Mock response for development without Gemini API.
    
    
    Only used when USE_MOCK_LLM=true in .env
    Simulates latency and generates basic responses based on state data.
    
    Args:
        state: Current agent state
        
    Returns:
        Dict with 'response' key containing mock response text
    """
    await asyncio.sleep(0.3)  # Simulate API latency
    
    intent = state.get("intent", "unknown")
    tool_results = state.get("tool_results", {})
    
    # Generate mock response based on intent and data
    if intent == "find_claim" and tool_results:
        data = tool_results.get("data", {})
        claims = data.get("claims", [])
        
        if claims:
            claim = claims[0]
            claim_info = claim.get("claimInformation", {})
            drug = claim.get("drug", {})
            pricing = claim.get("pricing", {})
            member = claim.get("member", {})
            prescription = claim.get("prescription", {})
            
            # Format mock response following system prompt format
            response = f"""SUMMARY: {drug.get('productName', 'Medication')} claim processed and {claim_info.get('claimStatusDescription', 'paid').lower()} on {claim_info.get('fillDate', 'N/A')}.

FINANCIAL:
• Patient paid: ${pricing.get('patientPay', '0.00')}
• Status: {claim_info.get('claimStatusDescription', 'N/A')}

DRUG:
• {drug.get('productName', 'N/A')}
• Quantity: {claim_info.get('quantity', 'N/A')}
• Days supply: {claim_info.get('daysSupplied', 'N/A')}
• NDC: {drug.get('productNdc', 'N/A')}

MEMBER: {member.get('firstName', '')} {member.get('lastName', '')} (ID: {member.get('memberId', 'N/A')})
PHARMACY: {prescription.get('pharmacyName', 'N/A')}

⚠️ MOCK RESPONSE - Enable real Gemini by setting USE_MOCK_LLM=false in .env"""
        else:
            response = "No claim data found in tool results.\n\n⚠️ MOCK RESPONSE"
    else:
        response = f"""I'm a mock response for intent: {intent}

⚠️ This is a MOCK LLM response (development mode)
💡 To use real Gemini, set USE_MOCK_LLM=false in your .env file

The real implementation will provide detailed, formatted responses based on claim data."""
    
    logger.info("⚙️ Returned mock response")
    return {"response": response}
