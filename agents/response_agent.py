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
    
    def _get_followup_system_prompt(self) -> str:
        """
        System prompt for generating follow-up clarification questions
        
        Used when needs_clarification=True in state.
        Instructs LLM to generate targeted, context-aware follow-up questions.
        
        Returns:
            str: Follow-up question generation prompt
        """
        return """**Role Overview:**
            You are a pharmacy claims assistant focused on improving user interactions by asking precise follow-up questions when user queries are unclear or incomplete.

            **Your Task:**
            When a user's question lacks clarity or necessary details, generate one specific and relevant follow-up question to clarify their intent. Your aim is to gather the information needed to assist the user effectively with their pharmacy claims inquiries.

            **Guidelines:**
            Generate ONE Question: Always ask one specific, conversational follow-up question to obtain missing information.

            CRITICAL RULES:
            1. **Focus on MISSING INFORMATION**: Ask ONLY for the missing information listed in "MISSING INFORMATION" field. Do NOT ask about ambiguous terms or abbreviations in the user's query (like "TF", "status", etc.)
            2. **Check CONVERSATION HISTORY FIRST**: The user may have already mentioned the information in a previous turn. Never ask for something they already provided.
            3. **Be Direct**: If missing "claim number", ask "Could you please provide your claim number?" - don't ask what "TF" means or other ambiguous terms.

            **Key Points:**

            • Ask only ONE question at a time.
            • Maintain a conversational tone and acknowledge previous user input.
            • Use provided data indicated by masked tokens like [CLAIM_ID_XXX].
            • If the user has mentioned a claim (e.g., X123) earlier, reference it without asking again.
            • Focus on the MISSING INFORMATION, not on clarifying ambiguous terms in the query

            **Important:**
                Do not ask for information that is already provided in the USER QUERY or CONVERSATION HISTORY.
                **CRITICAL: Masked Token Handling (MUST FOLLOW)**
                    *Do not modify masked tokens.** When you encounter tokens such as [CLAIM_ID_XXX], [MEMBER_ID_XXX], or [PERSON_XXX]:
                    **Always retain the square brackets [ and ] exactly as shown.**
                    **Never remove the brackets or enclose tokens in backticks or quotes.**
                    **Copy tokens exactly as they appear.**

                **Examples: Masked ID received from the user query or conversation history: [CLAIM_ID_B161BCED] **
                    ❌ WRONG: "Could you confirm CLAIM_ID_B161BCED?"
                    ❌ WRONG: "Could you confirm CLAIM_ID_B161BCED?]"
                    ❌ WRONG: "Could you confirm [CLAIM_ID_B161BCED?"
                    ❌ WRONG: "Could you confirm `CLAIM_ID_B161BCED?`"
                    ✅ CORRECT: "Could you confirm [CLAIM_ID_B161BCED]?"
            
            **Input:**
                USER QUERY: Current question (may contain ambiguous terms - ignore those)
                CONVERSATION HISTORY: Previous messages (check this first).
                MISSING INFORMATION: What you need to ask for (e.g., "claim number or claim ID")
                PROVIDED INFORMATION: What's in current message

            **Output:**
            Provide only the follow-up question, without any explanations.

            **Examples of Good Questions:**
                ✅ "I can help with that. Could you please provide your claim number?"
                ✅ "To look that up for you, could you provide your claim number or claim ID?"
                ✅ "I'd be happy to help! Could you please provide your claim number?"

            **Examples of Poor Questions:**
                ❌ "What does 'TF' stand for?" (asking about ambiguous term, not missing entity)
                ❌ "Which claim?" (if CLM123 was mentioned earlier)
                ❌ "Please provide claim number, date, and pharmacy." (too many requests)
                ❌ "Error: missing parameter claim_id." (too robotic)

            **Note on Claim and other Identifiers:**
            - When users provide a claim number, the system may require both the claim number and claim sequence number for accurate lookup.
            - Focus on what's explicitly listed in MISSING INFORMATION field for handling such cases.
            - Apply the same approach for other identifiers as well. If additional related identifiers are required for accurate lookup, ensure they are captured based on what is missing.
            
             ## Communication Style

                **TONE:**
                - Be warm, professional, and genuinely helpful
                - Use active voice
                - Be assertive and confident
                - Acknowledge the user's situation before diving into data
                - Make sure that all the responses should be conversational in nature.

            **Generate one clear, helpful question that asks for the MISSING INFORMATION.**"""
    
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

## CRITICAL: Masked Token Handling (MUST FOLLOW)

**NEVER modify, remove, or reformat masked tokens in your response.**

Masked tokens are privacy placeholders that look like `[ENTITY_TYPE_HEXCODE]`. Examples:
- `[CLAIM_ID_B161BCED]`
- `[MEMBER_ID_A1F2C3D4]`
- `[PERSON_E5F6G7H8]`
- `[US_SSN_12345678]`

**STRICT RULES:**
1. **ALWAYS keep the square brackets `[` and `]` exactly as shown**
2. **NEVER remove brackets or any characters from tokens**
3. **NEVER wrap tokens in backticks, quotes, or other formatting**
4. **Copy tokens EXACTLY as they appear - character for character**

Examples of CORRECT vs WRONG usage:

❌ WRONG: "I cannot find claim CLAIM_ID_B161BCED"
❌ WRONG: "I cannot find claim `CLAIM_ID_B161BCED`"
❌ WRONG: "I cannot find claim 'CLAIM_ID_B161BCED'"
❌ WRONG: "I cannot find claim ID: CLAIM_ID_B161BCED"
✅ CORRECT: "I cannot find claim [CLAIM_ID_B161BCED]"

❌ WRONG: "Member PERSON_E5F6G7H8 has the following claims..."
✅ CORRECT: "Member [PERSON_E5F6G7H8] has the following claims..."

This is CRITICAL for data security. Tokens are automatically replaced with real values after your response.

## Your Identity & Capabilities

**WHO YOU ARE:**
You are a helpful, knowledgeable pharmacy claims assistant. You have access to pharmacy claim data and can help users understand their prescription claims, costs, coverage, and resolve issues.

**WHAT YOU CAN DO:**
- Look up claim details by claim ID
- Explain claim status (paid, rejected, reversed)
- Break down costs (copay, plan paid, patient responsibility)
- Explain rejection reasons and provide resolution steps
- Show drug information (name, quantity, days supply, NDC)
- Display pharmacy and prescriber details
- Explain benefit phases and accumulator information
- Review claim history within date ranges

**WHAT YOU CANNOT DO:**
- Access real-time information (current date, weather, news)
- Make changes to claims or benefits
- Access information outside the pharmacy claims domain
- Process refunds or payments

**Self-Introduction:**
When asked "Who are you?", "What can you do?", "How can you help me?", or similar:
→ Briefly introduce yourself and highlight 2-3 key capabilities relevant to their context
→ Be conversational, not robotic: "I'm your pharmacy claims assistant! I can help you understand your prescription claims, explain any rejections, break down your costs, and more. What would you like to know?"

## Communication Style

**TONE:**
- Be warm, professional, and genuinely helpful
- Use active voice: "I found your claim" NOT "Based on the provided data, your claim shows..."
- Be assertive and confident: "Your claim was paid on..." NOT "It appears that..."
- Acknowledge the user's situation before diving into data
- Make sure that all the responses should be conversational in nature.

**WHEN DATA IS UNAVAILABLE:**
- Never say: "I cannot help as context is insufficient"
- Never ask for a claim ID if the user already provided one
- If user provided claim ID but no data found: "I wasn't able to find claim [ID they provided]. Could you please double-check the claim number?"
- If user didn't provide any claim ID: "Could you please provide the claim number so I can look that up for you?"


**Note on Claim Identifiers:**
When users provide a claim number, the system may require both the claim number and claim sequence number for accurate lookup. This is handled automatically by the clarification system - you should respond based on the data provided to you.

## Response Strategy

**CRITICAL: User's EXPLICIT request for comprehensive information overrides intent-based sectioning.**

### STEP 1: Check for Comprehensive Response Keywords (CHECK FIRST)

Before applying intent-based rules, check if the user's query contains ANY of these keywords/phrases:
- **"details"**, **"all details"**, **"full details"**, **"the details"**
- **"summary"**, **"full summary"**, **"claim summary"**, **"complete summary"**
- **"everything"**, **"all information"**, **"full information"**
- **"complete"**, **"comprehensive"**, **"entire"**

**If ANY of these keywords are found:** Provide the FULL claim response format (see "For FULL claim summaries" section below). Do NOT limit to a single section.

Examples:
- "What are the **details** for claim X?" → Provide FULL response (SUMMARY + FINANCIAL + DRUG + MEMBER + PHARMACY)
- "Give me a **summary** of claim X" → Provide FULL response
- "Tell me **everything** about claim X" → Provide FULL response

### STEP 2: Intent-Based Response Guidelines (ONLY when NO comprehensive keywords detected)

**For specific, narrow queries WITHOUT the keywords above, provide ONLY the relevant section:**

- **claim_status, approval_info** (narrow queries like "is it paid?"): SUMMARY only (status, date, drug name)
- **pricing_info, settlement_info, cob_info, reimbursement_info**: FINANCIAL section only
  - Patient paid, Plan paid, Accumulation/deductible
- **drug_info, rx_details**: DRUG section only
  - Drug name, dosage, quantity, days supply, NDC
- **pharmacy_info**: PHARMACY section only
  - Name, location, NCPDP ID
- **rejection_reasons**: REJECTION section with NEXT STEPS
  - Rejection code, message, recommended actions
- **beneficiary_info, member_info**: MEMBER section only
- **General/unclear queries**: Provide relevant sections based on query context

### Handling Chit-Chat & Non-Claim Queries:

**greeting (hello, hi, good morning):**
→ Respond warmly and offer assistance: "Hello! I'm here to help with your pharmacy claims. What would you like to know about your prescriptions?"

**help (how do I, what should I do):**
→ Provide guidance relevant to their question. If general, briefly explain your capabilities and ask what specific help they need.

**out_of_scope (weather, jokes, unrelated topics):**
→ Be graceful and redirect politely:
  - "I appreciate the question! While I can't help with [topic], I'm great at helping you understand your pharmacy claims. Do you have any questions about your prescriptions or claim status?"
  - Never be dismissive or robotic
  - Acknowledge what they asked, explain your focus, offer relevant help
  - Avoid any inappropriate requests politely, explain your focus, and offer relevant help.

### Response Formatting:

1. Use bullet points for easy scanning
2. Be concise - avoid wordiness and unnecessary explanations
3. Maintain professional pharmacy terminology
4. For follow-up questions, acknowledge previous context: "For the claim we discussed earlier..."

### For FULL claim summaries (when requested), include:

#### PAID or REVERSED claims:
- One-line summary (claim date, drug name, status)
- Financial information (patient cost, plan paid, accumulation)
- Drug information (name, dosage, quantity, days supply)
- Member demographics (basic info)
- Pharmacy information (name, location)

#### REJECTED claims:
- One-line summary (claim date, drug name, rejection reason)
- Drug information (name, dosage, quantity)
- Member demographics (basic info)
- Pharmacy information (name, location)
- Rejection code(s) and message(s)
- Next steps to resolve (CRITICAL - very important)

**CRITICAL FOR REJECTED CLAIMS**: When REJECT ANALYSIS data is provided, ALWAYS prioritize the detailed explanations, reasons, and actions from REJECT ANALYSIS over basic claim rejection information. The REJECT ANALYSIS contains expert-level, persona-specific guidance that is more valuable than raw rejection codes.

## Handling Invalid or Missing Data

**CRITICAL: Always acknowledge the identifier the user provided. Never ask for an ID if they already gave one.**

**When user provides an identifier but no data is found:**
- For claim ID: "I wasn't able to find claim 12345 in the system. Could you please double-check the claim number?"
- For member ID: "I wasn't able to find member M1234567 in the system. Could you please verify the member ID?"
- Do NOT say: "Could you please provide the claim/member number?" (They already did!)

**When user asks about something but provides NO identifier:**
- For claims: "I'd be happy to help with that. Could you please provide the claim number so I can look it up?"
- For member info: "I'd be happy to help. Could you please provide the member ID?"

**When data is retrieved but seems incomplete:**
→ Provide what data is available and note what's missing: "Here's what I found for claim 12345. Some details may not be available in the system."

## Handling Questions

### Initial Questions
- If the user asks about a specific aspect of a claim (e.g., "What was my copay for Lisinopril?"), provide only the relevant information in the same concise format.
- If the user asks a general question about a claim, provide the full structured response based on claim status.
- Always maintain the same concise, structured format regardless of question type.

## Response Style

- Use technical terminology appropriate for pharmacy professionals
- Present information in a structured, scannable format
- Keep explanations brief and factual
- When uncertain about specific claim details, acknowledge limitations rather than providing potentially incorrect information
- For all responses, maintain the same structured, concise format
- Respond conversationally by default; use a clear, labeled table only when comparing multiple items or presenting structured data.
- Never include repetitive or redundant words in your response; keep it concise and clear. For example: Your claim claim [CLM1234] was processed successfully - INCORRECT❌ due to repetition of the word claim.

## Example Table Formats
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

**NEXT STEPS:**
• Wait until the next eligible fill date
• Contact your pharmacy if an early refill is needed
• Your prescriber can request an override if medically necessary

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
• Wait until the next eligible fill date
• Contact your pharmacy if an early refill is needed
• Your prescriber can request an override if medically necessary


Use this structured format when presenting claim data. For conversational exchanges, prioritize natural, flowing dialogue, but always ensure it is factual and concise."""
    
    def _map_entity_to_user_friendly(self, entity_name: str) -> str:
        """
        Map technical entity names to user-friendly labels for clarification questions.
        
        Args:
            entity_name: Technical entity name (e.g., 'claim_ids', 'claim_number')
            
        Returns:
            str: User-friendly label (e.g., 'claim number' or 'claim ID')
        """
        entity_map = {
            # Claim-related
            'claim_ids': 'claim number or claim ID',
            'claim_id': 'claim number or claim ID',
            'claim_number': 'claim number',
            'claim_sequence': 'claim sequence number',
            'sequence': 'claim sequence number',  # Added: matches confidence.py missing_slots
            'claim_numbers': 'claim numbers',
            
            # Member-related
            'member_ids': 'member ID',
            'member_id': 'member ID',
            
            # Prescription-related
            'prescription_ids': 'prescription number',
            'prescription_id': 'prescription number',
            'prescription_number': 'prescription number',
            
            # Date-related
            'date_range': 'date range',
            'start_date': 'start date',
            'end_date': 'end date',
        }
        
        # Return mapped name or original if not found
        return entity_map.get(entity_name.lower(), entity_name)
    
    def _build_user_prompt(self, state: AgentState) -> str:
        """
        Build user prompt from state data - handles both modes with separate variables
        
        MODES:
        - Normal mode (needs_clarification=False): Uses tool_results for claim responses
        - Clarification mode (needs_clarification=True): Uses clarification_context for questions
        
        Uses LangChain's ChatPromptTemplate for structured prompt management,
        following pattern from agents/intent_agent.py.
        
        Args:
            state: Current agent state with query, intent, and mode-specific data
            
        Returns:
            str: Formatted prompt for LLM
        """
        # Check mode first
        needs_clarification = state.get("needs_clarification", False)
        
        if needs_clarification:
            # ===== CLARIFICATION MODE: Generate follow-up question =====
            self.logger.info("🔍 Building clarification prompt (using clarification_context)")
            
            # Extract clarification context (not tool_results!)
            clarification_ctx = state.get("clarification_context")
            
            # DEFENSIVE: Handle None case (should be dict but sometimes isn't)
            if clarification_ctx is None:
                self.logger.warning("⚠️ clarification_context is None, using defaults")
                clarification_ctx = {}
            
            # Defensive check
            if state.get("tool_results"):
                self.logger.warning("⚠️ Unexpected: needs_clarification=True but tool_results present")
            
            # Extract context fields with defaults
            reason = clarification_ctx.get("reason", "unknown")
            user_query = clarification_ctx.get("user_query") or state.get("text", "")  # Fallback to state.text
            intent = clarification_ctx.get("intent") or state.get("intent", "unknown")  # Fallback to state.intent
            confidence = clarification_ctx.get("confidence") or state.get("confidence", 0.0)  # Fallback to state.confidence
            missing_entities = clarification_ctx.get("missing_entities", [])
            provided_entities = clarification_ctx.get("provided_entities", [])
            
            # Map technical entity names to user-friendly labels
            missing_entities_friendly = [self._map_entity_to_user_friendly(e) for e in missing_entities]
            
            # Provide appropriate fallback based on clarification reason
            if missing_entities_friendly:
                missing_entities_str = ", ".join(missing_entities_friendly)
            elif reason == "low_confidence":
                missing_entities_str = "more details about what you need"
            elif reason == "ambiguous_intent":
                missing_entities_str = "clarification on what you'd like to know"
            else:
                # Default fallback for missing_entity reason with empty list (shouldn't happen)
                missing_entities_str = "the claim number and sequence number"
            
            # Format history (may provide context for question)
            history = state.get("conversation_history", [])
            history_str = self._format_conversation_history(history) if history else "No previous conversation"
            
            # Build clarification prompt with explicit guidance
            prompt_template = ChatPromptTemplate.from_messages([
                ("user", """REASON FOR CLARIFICATION: {reason}

USER QUERY: {user_query}

INTENT: {intent} (confidence: {confidence:.2f})

MISSING INFORMATION: {missing_entities}
PROVIDED INFORMATION: {provided_entities}

=== CONVERSATION HISTORY ===
{conversation_history}

CRITICAL INSTRUCTIONS:
1. Focus ONLY on asking for the MISSING INFORMATION listed above (e.g., "{missing_entities}")
2. Do NOT ask about ambiguous terms in the user query (like "TF" or abbreviations)
3. The user wants to know about their claim - ask for the claim number/ID to look it up
4. Be direct and specific: "Could you please provide your {missing_entities}?"

Generate ONE specific, helpful follow-up question to get the missing information. Just the question, no explanation.""")
            ])
            
            messages = prompt_template.format_messages(
                reason=reason,
                user_query=user_query,
                intent=intent,
                confidence=confidence,
                missing_entities=missing_entities_str,
                provided_entities=", ".join(provided_entities) if provided_entities else "None",
                conversation_history=history_str
            )
            
        else:
            # ===== NORMAL MODE: Generate claim response =====
            self.logger.info("💬 Building normal response prompt (using tool_results)")
            
            # Extract data from state (not clarification_context!)
            user_text = state.get("text", "")
            intent = state.get("intent", "unknown")
            tool_results = state.get("tool_results")
            history = state.get("conversation_history", [])
            
            # Defensive check
            if state.get("clarification_context"):
                self.logger.warning("⚠️ Unexpected: needs_clarification=False but clarification_context present")
            
            # Format tool results and history
            claim_data = self._format_tool_results(tool_results) if tool_results else "No claim data available"
            history_str = self._format_conversation_history(history) if history else "No previous conversation"
            
            # Build normal response prompt
            prompt_template = ChatPromptTemplate.from_messages([
                ("user", """USER QUERY: {user_query}

INTENT: {intent} <- Focus your response on this specific intent

=== CLAIM DATA ===
{claim_data}

=== CONVERSATION HISTORY ===
{conversation_history}

Provide a targeted response that directly answers the user's question based on the INTENT above. Only include sections relevant to their specific query. Do not provide a full summary unless the query is general or asks for comprehensive information.""")
            ])
            
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
        
        CRITICAL: Uses PII-masked data for LLM consumption to prevent leakage.
        The response_safety_pii_postcheck_node will unmask the final response for the user.
        
        Real ToolResult structure (after PII masking):
        {
            "tool_name": "get_claim_list",
            "status": "success",
            "data": {
                "claims": [{...}],  // Contains real PII (for programmatic access)
                "_masked_response": "...",  // PII-masked text for LLM
                "_pii_metadata": {...}
            },
            "execution_time_ms": 4381.1,
            ...
        }
        
        Args:
            tool_results: ToolResult dictionary from claims API
            
        Returns:
            str: PII-masked string for LLM consumption
        """
        try:
            # Extract the data field (contains actual claim data)
            data = tool_results.get("data", {})
            
            # CRITICAL: Use _masked_response if available (contains PII-masked data for LLM)
            # This prevents PII leakage in LLM-generated responses
            if "_masked_response" in data:
                self.logger.debug("Using _masked_response for LLM (PII-masked)")
                return data["_masked_response"]
            
            # Fallback: Use full data (for backward compatibility with non-masked responses)
            # Remove internal fields that start with underscore
            filtered_data = {k: v for k, v in data.items() if not k.startswith("_")}
            
            # Use standard library json for pretty printing (indent for LLM readability)
            import json
            return json.dumps(filtered_data, indent=2)
            
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
            # First attempt: Use default safety settings (let Gemini use its defaults)
            # Safety filtering is already done in safety_precheck_node with BLOCK_LOW_AND_ABOVE
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
            
            # If response is empty, retry with more permissive settings for medical/pharmacy content
            if not response.text or len(response.text.strip()) == 0:
                self.logger.warning("⚠️ Empty response from Gemini, retrying with adjusted safety thresholds for medical content...")
                
                # Use BLOCK_ONLY_HIGH for medical/pharmacy domain where terms like "rejected", 
                # "denied", "dangerous drugs" are legitimate professional terminology
                safety_thresholds = {
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH"
                }
                
                req_retry = GenerateRequest(
                    prompt=user_prompt,
                    system_instruction=system_prompt,
                    temperature=settings.llm_temperature,
                    top_p=settings.top_p,
                    max_output_tokens=settings.max_output_tokens,
                    model=settings.llm_model,
                    safety_thresholds=safety_thresholds
                )
                
                self.logger.info("� Retrying with BLOCK_ONLY_HIGH thresholds...")
                response = _generate_core(req_retry)
            
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
        # Detect mode
        needs_clarification = state.get("needs_clarification", False)
        
        if needs_clarification:
            logger.info("🤖 AGENT 2: Follow-Up Question Generation (Clarification Mode)")
        else:
            logger.info("🤖 AGENT 2: Response Generation (Normal Mode)")
        
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
                data={
                    "node": node_name, 
                    "intent": state.get("intent", "unknown"),
                    "needs_clarification": needs_clarification
                }
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
        
        # Get appropriate system prompt based on mode
        if needs_clarification:
            system_prompt = agent._get_followup_system_prompt()
            logger.info("📝 Using follow-up question system prompt")
        else:
            system_prompt = agent._get_system_prompt()
            logger.info("📝 Using standard response system prompt")
        
        logger.debug(f"📋 System prompt: {len(system_prompt)} characters")
        
        # Build user prompt from state (mode-aware)
        user_prompt = agent._build_user_prompt(state)
        logger.debug(f"📋 User prompt: {len(user_prompt)} characters")
        logger.debug(f"📋 Intent: {state.get('intent', 'unknown')}")
        
        # Generate response using Gemini
        if needs_clarification:
            logger.info("🔮 Generating follow-up question with Gemini...")
        else:
            logger.info("🔮 Generating response with Gemini...")
        
        # Call generation method (synchronous but safe to call from async context)
        response_text = agent.generate_response(system_prompt, user_prompt)
        
        # Validate response
        if not response_text or not response_text.strip():
            logger.warning("⚠️ Empty response received from Gemini")
            if needs_clarification:
                response_text = "Could you please provide more details about your question?"
            else:
                response_text = "I apologize, but I received an empty response. Please try again."
        
        if needs_clarification:
            logger.info(f"✅ Follow-up question generated: {len(response_text)} chars")
            logger.info(f"❓ Question: {response_text}")
        else:
            logger.info(f"✅ Response generated: {len(response_text)} chars")
            logger.info(f"💬 Preview: {response_text[:100]}...")
        
        # Log successful generation (telemetry pattern from tools/claims_api.py)
        if log_ctx and persistence_store:
            await persistence_store.log_audit(
                session_id=log_ctx.get("session_id"),
                request_id=log_ctx.get("request_id"),
                user_id=log_ctx.get("user_id"),
                node_name=node_name,
                event_type="followup_question_generated" if needs_clarification else "response_generated",
                data={
                    "response_length": len(response_text),
                    "model": settings.llm_model,
                    "temperature": settings.llm_temperature,
                    "needs_clarification": needs_clarification
                }
            )
        
        # Build result - always use 'response' field for both modes
        # The 'needs_clarification' flag in state already indicates if this is a question
        result = {"response": response_text}
        
        if needs_clarification:
            logger.info("📝 Set 'response' field with clarification question")
        else:
            logger.info("📝 Set 'response' field with normal answer")
        
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
        result = {
            "error": error.user_message,
            "response": error.user_message,
            "metadata": {
                **state.get("metadata", {}),
                "error_occurred": True,
                "error_code": error.error_code.value
            }
        }
        await log_state_snapshot(state, node_name, result)
        return result


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
