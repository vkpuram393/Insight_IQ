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
from tools.claims_api import normalize_entities
import uuid

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
            • Use masked tokens exactly as they appear in the input data.
            • If the user has mentioned a claim earlier, reference it without asking again.
            • Focus on the MISSING INFORMATION, not on clarifying ambiguous terms in the query

            **Important:**
                Do not ask for information that is already provided in the USER QUERY or CONVERSATION HISTORY.
                **CRITICAL: Masked Token Handling (MUST FOLLOW)**
                    **Do not modify masked tokens.** When you encounter tokens in the input:
                    **Always retain the square brackets [ and ] exactly as shown.**
                    **Never remove the brackets or enclose tokens in backticks, quotes, or parentheses.**
                    **Copy tokens exactly as they appear in the input.**

                **Examples of WRONG token handling (using placeholder [CLAIM_ID_XXXXXXXX] for illustration):**
                    ❌ WRONG: "Could you confirm CLAIM_ID_XXXXXXXX?"
                    ❌ WRONG: "Could you confirm CLAIM_ID_XXXXXXXX?]"
                    ❌ WRONG: "Could you confirm [CLAIM_ID_XXXXXXXX?"
                    ❌ WRONG: "Could you confirm `CLAIM_ID_XXXXXXXX?`"
                    ❌ WRONG: "Could you confirm (CLAIM_ID: XXXXXXXX)?"
                    ✅ CORRECT: "Could you confirm [CLAIM_ID_XXXXXXXX]?"
                
                ⚠️ **CRITICAL WARNING - DO NOT COPY PLACEHOLDER EXAMPLES**: 
                    The placeholder patterns above (containing XXXXXXXX) are ONLY for demonstrating correct vs incorrect formatting.
                    These are NOT real tokens! Real tokens follow the pattern [ENTITY_TYPE_HASH] where HASH is exactly 8 uppercase hexadecimal characters (0-9 and A-F), generated by the system.
                    Real tokens will ONLY appear in the USER QUERY or CLAIM DATA provided to you.
                    NEVER output placeholder patterns (with X's) in your response!
                    If you need to reference a claim but no token exists in the input, use natural language like "the claim" or "your claim".
                
                **STRICT RULES:**
                1. ONLY use tokens that ACTUALLY APPEAR in the USER QUERY or CLAIM DATA provided below
                2. If no token exists for a value, use natural language (e.g., "the claim", "your prescription")
                3. Plain claim numbers provided by user should be used directly as-is
                4. NEVER fabricate, invent, or copy placeholder tokens from these instructions
            
            **Input:**
                USER QUERY: Current question (may contain ambiguous terms - ignore those)
                CONVERSATION HISTORY: Previous messages (check this first).
                MISSING INFORMATION: What you need to ask for (e.g., "claim number", "3-digit sequence number")
                PROVIDED INFORMATION: What's in current message

            **Output:**
            Provide only the follow-up question, without any explanations.

            **Examples of Good Questions:**
                ✅ "I can help with that. Could you please provide your claim number?"
                ✅ "To look that up for you, could you provide your claim number?"
                ✅ "I'd be happy to help! Could you please provide your claim number and 3-digit sequence number?"

            **Examples of Poor Questions:**
                ❌ "What does 'TF' stand for?" (asking about ambiguous term, not missing entity)
                ❌ "Which claim?" (if CLM123 was mentioned earlier)
                ❌ "Please provide claim number, date, and pharmacy." (too many requests)
                ❌ "Error: missing parameter claim_id." (too robotic)

            **Note on Claim and other Identifiers:**
            - When users provide a claim number, the system may require both the claim number and sequence number for accurate lookup.
            - **CRITICAL: When asking for a sequence number, ALWAYS specify it as "3-digit sequence number"** (never just "sequence number").
            - Focus on what's explicitly listed in MISSING INFORMATION field for handling such cases.
            
            **⚠️ HARD CONSTRAINT - VALID ENTITIES TO ASK FOR:**
            This system ONLY supports looking up information by **claim number** and **3-digit sequence number**.
            - ✅ VALID to ask for: claim number, 3-digit sequence number
            - ❌ NEVER ask for: member ID, member number, patient ID, prescription ID, or any other identifier
            - Even if the user's query seems to require member-level data (like "Medicare accumulation"), still ask for the claim number - the system will extract member info from the claim data.
            - This is a pharmacy claims system - ALL lookups are claim-based.
            
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

Masked tokens are privacy placeholders with format `[ENTITY_TYPE_HASH]`. Example patterns for illustration:
- `[CLAIM_ID_XXXXXXXX]`
- `[MEMBER_ID_XXXXXXXX]`
- `[PERSON_XXXXXXXX]`
- `[US_SSN_XXXXXXXX]`

⚠️ **CRITICAL WARNING - ABOUT THESE EXAMPLE PATTERNS**: 
The patterns above (containing XXXXXXXX) are ONLY for demonstrating the token format structure.
These are NOT real tokens and must NEVER appear in your output!
Real tokens follow the pattern [ENTITY_TYPE_HASH] where HASH is exactly 8 uppercase hexadecimal characters (0-9 and A-F), generated by the system.
Real tokens will ONLY be present in the USER QUERY or CLAIM DATA sections provided to you.
If you need to reference a claim or member but no token exists in the input, use natural language like "the claim", "your prescription", or "the member".

**IMPORTANT: Only reference tokens that ACTUALLY APPEAR in the input data.**
If the user provides a plain claim number, use it directly as-is - do NOT create tokens.

**STRICT RULES:**
1. **ALWAYS keep the square brackets `[` and `]` exactly as shown**
2. **NEVER remove brackets or any characters from tokens**
3. **NEVER wrap tokens in backticks, quotes, parentheses, or other formatting**
4. **Copy tokens EXACTLY as they appear in the input - character for character**
5. **NEVER fabricate, invent, or copy the placeholder patterns from these instructions**
6. **If no token exists in input, use natural language** (e.g., "the claim", "your prescription")

Examples of CORRECT vs WRONG token handling (using placeholder [CLAIM_ID_XXXXXXXX] for illustration):

❌ WRONG: "I cannot find claim CLAIM_ID_XXXXXXXX"
❌ WRONG: "I cannot find claim `CLAIM_ID_XXXXXXXX`"
❌ WRONG: "I cannot find claim 'CLAIM_ID_XXXXXXXX'"
❌ WRONG: "I cannot find claim (CLAIM_ID: XXXXXXXX)"
❌ WRONG: "I cannot find claim ID: CLAIM_ID_XXXXXXXX"
✅ CORRECT: "I cannot find claim [CLAIM_ID_XXXXXXXX]" (only if this exact token exists in input)

❌ WRONG: "Member PERSON_XXXXXXXX has the following claims..."
✅ CORRECT: "Member [PERSON_XXXXXXXX] has the following claims..." (only if this exact token exists in input)

⚠️ **FINAL REMINDER**: 
The XXXXXXXX placeholder patterns in these instructions demonstrate the token FORMAT only.
Real tokens have 8 uppercase hex characters (like A1B2C3D4), not X's.
You must NEVER output placeholder patterns - only use actual tokens from the USER QUERY or CLAIM DATA below.
If no token is available for a value, use natural descriptive language instead (e.g., "the claim", "this prescription").

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
When users provide a claim number, the system may require both the claim number and sequence number for accurate lookup. **CRITICAL: When asking the user for a sequence number, ALWAYS specify it as "3-digit sequence number"** (never just "sequence number"). This is handled automatically by the clarification system - you should respond based on the data provided to you.

**CRITICAL - Always Include Identifiers in Response:**
When providing claim information, always include the claim number and sequence number for absolute clarity so the user knows exactly which claim the data belongs to.

## CRITICAL - DATA SOURCE RULES (MUST FOLLOW STEP BY STEP)

You receive information in THREE sections. Each section has a specific purpose and format:

### STEP 1: ENTITIES Section (PLAIN NUMERIC IDENTIFIERS)
This section contains PLAIN NUMBERS - specifically the 15-digit claim number and 3-digit sequence number.
- These are the AUTHORITATIVE identifiers for the CURRENT request
- You MUST cite these EXACT numbers at the BEGINNING of your response
- Write them as plain numbers exactly as shown - do not modify or substitute them
- Do NOT replace these with masked tokens from other sections
- The current response belongs to these identifiers

### STEP 2: CLAIM DATA Section (ANSWER SOURCE - MAY CONTAIN MASKED TOKENS)
This section contains the actual claim information retrieved for the current request.
- This is the ONLY section you should use to answer the user's question
- Data values may appear as masked tokens in square brackets - use them exactly as they appear
- Do not modify, remove brackets, or alter masked tokens in any way
- These masked tokens get automatically replaced with real values after your response
- All financial amounts, drug details, pharmacy info, and member info come from here

### STEP 3: CONVERSATION HISTORY (CONTEXT ONLY - NEVER USE DATA OR TOKENS FROM HERE)
This section shows previous conversation turns to help you understand context.
- It contains identifiers and data from COMPLETELY DIFFERENT claims discussed in earlier turns
- NEVER copy any claim numbers, sequence numbers, or masked tokens from history into your response
- NEVER answer questions using any data values from history
- Using identifiers from history WILL cause your response to show information for the WRONG claim
- Only use history to understand the conversational flow and what the user is asking about

### HOW TO CONSTRUCT YOUR RESPONSE:
1. FIRST: Begin with a natural conversational sentence starting with 'For claim' followed by the exact plain numeric 15-digit claim number from ENTITIES, then 'sequence' followed by the exact plain numeric 3-digit sequence number from ENTITIES, then a comma or colon before continuing
2. THEN: Answer the user's question using ONLY data from CLAIM DATA section
3. If CLAIM DATA contains masked tokens, use them exactly as shown - they will be replaced automatically
4. NEVER reference, copy, or use any identifiers or data values from CONVERSATION HISTORY

### WHY THIS MATTERS:
- ENTITIES contains plain numbers because they identify THIS specific request
- CLAIM DATA contains the correct answer data for the current claim (masked tokens unmask correctly)
- HISTORY contains tokens and data for OLD claims - using them shows WRONG claim information to the user

## CRITICAL: Data Exploration and Reasoning Strategy (READ THIS FIRST)

**The INTENT provided is primarily for API routing purposes and may not necessarily capture the complete essence of the current user question correctly. Do NOT let it limit your data exploration or reasoning - always explore complete claim data with logical reasoning.**

### GOLDEN RULE: Understand First, Answer Second

Before generating ANY response, you MUST:

1. **FULLY UNDERSTAND THE USER'S ACTUAL QUESTION**
   - Read the USER QUERY carefully - what is the user ACTUALLY asking?
   - If the query mentions "final", "after", "remaining", "total", or "net" amounts - these often require looking at MULTIPLE data sources
   - If the query is ambiguous, use CONVERSATION HISTORY only to understand context (NEVER use data/tokens from history - they belong to different claims)

2. **EXPLORE ALL RELEVANT FIELDS IN CLAIM DATA**
   - Do NOT assume where data lives based on the intent name
   - Scan through ALL sections of CLAIM DATA before deciding which values to use
   - Different fields may contain similar-looking data with DIFFERENT meanings
   - When multiple fields contain patient pay amounts, determine WHICH ONE actually answers the user's question

3. **REASON ABOUT DATA SELECTION**
   - If you find the same type of value in multiple places, ask yourself: "Which one is correct for THIS specific question?"
   - Consider the CONTEXT of the field (e.g., `primary` vs `linkedClaim.stcob` mean different things)
   - The most obvious or first-found field is NOT always the correct answer

---

### Example: Applying These Principles to Coordination of Benefits (COB)

The following example demonstrates how to apply the above principles to a common scenario where rushing leads to wrong answers:

**Scenario:** User asks "What was the final patient pay after primary and secondary coverage?"

**COMMON MISTAKE (Rushing):** 
Seeing "patient pay" and immediately grabbing `claimDetails.primary.approvedPatientPayAmount`. 
This is WRONG because `primary` contains the patient pay BEFORE secondary coverage was applied!

**CORRECT APPROACH (Following the principles):**

1. **Understand the question:** User wants "FINAL" pay "AFTER" secondary coverage - This is a COB question
2. **Explore all fields:** Look for COB-related sections - Find `linkedClaim.stcob`
3. **Reason about selection:** 
   - `primary.approvedPatientPayAmount` = Patient pay BEFORE secondary
   - `linkedClaim.stcob.responsePatientPayAmount` = Patient pay AFTER secondary - This answers "final"!

**Field Reference for COB Questions:**

| User Asks About | USE This Field | DO NOT Use |
|-----------------|----------------|------------|
| "Final patient pay after secondary" | `linkedClaim.stcob.responsePatientPayAmount` | `primary.approvedPatientPayAmount` |
| "What did secondary coverage pay?" | `linkedClaim.stcob.responseTotalAmountPaid` | - |
| "Patient pay before secondary" | `primary.approvedPatientPayAmount` | `linkedClaim.stcob.*` |

**Key Insight:** When CLAIM DATA contains `linkedClaim.stcob` AND the user asks about "final" or "after secondary" amounts, ALWAYS prefer `stcob` values over `primary` values.

---

### Before Finalizing Your Response - Quick Verification

Ask yourself:
- Did I understand what the user is ACTUALLY asking (not just the intent label)?
- Did I explore the relevant sections of CLAIM DATA (not just the obvious ones)?
- If multiple fields have similar values, am I using the RIGHT one for this question?

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

**IMPORTANT:** These guidelines specify which SECTIONS to include in your response (formatting). They do NOT limit which fields to explore in CLAIM DATA. You must STILL understand the actual user question and explore ALL relevant CLAIM DATA fields to find the correct answer - then present it in the appropriate section format below.

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
- For claim ID: "I wasn't able to find claim 12345 in the system. Could you please double-check the claim number and make sure it's valid?"
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
            
            # Check for invalid claim ID format (user tried to provide ID but wrong format)
            claim_id_format_invalid = clarification_ctx.get("claim_id_format_invalid", False)
            potential_claim_ids = clarification_ctx.get("potential_claim_ids", [])
            
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
            
            # Build special instruction for invalid claim ID format
            # Make it part of CRITICAL INSTRUCTIONS so LLM prioritizes it
            format_hint_instruction = ""
            if claim_id_format_invalid and potential_claim_ids:
                invalid_id = potential_claim_ids[0] if potential_claim_ids else "unknown"
                format_hint_instruction = f"""**PRIORITY**: The user provided '{invalid_id}' as a claim ID, but it's NOT VALID. You MUST acknowledge this invalid ID in your response.
Example: "I see you mentioned claim {invalid_id}, but that doesn't appear to be a valid claim number. Could you please provide a valid claim number?"
"""
            else:
                format_hint_instruction = "Be direct and specific in your question."
            
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
{format_hint_instruction}
1. Focus ONLY on asking for the MISSING INFORMATION listed above (e.g., "{missing_entities}")
2. Do NOT ask about ambiguous terms in the user query (like "TF" or abbreviations)
3. The user wants to know about their claim - ask for the claim number/ID to look it up

Generate ONE specific, helpful follow-up question to get the missing information. Just the question, no explanation.""")
            ])
            
            messages = prompt_template.format_messages(
                reason=reason,
                user_query=user_query,
                intent=intent,
                confidence=confidence,
                missing_entities=missing_entities_str,
                provided_entities=", ".join(provided_entities) if provided_entities else "None",
                format_hint_instruction=format_hint_instruction,
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
            
            # CRITICAL FIX: Merge extracted_slots (from history) with current entities
            # Normalize keys first so claim_ids and claim_number both become claimNumber
            # This ensures current claim properly overwrites old claim from history
            # Priority: current entities overwrite history entities (for recency)
            extracted_slots = state.get("extracted_slots", {})
            current_entities = state.get("entities", {})
            normalized_extracted = normalize_entities(extracted_slots) if extracted_slots else {}
            normalized_current = normalize_entities(current_entities) if current_entities else {}
            entities = {**normalized_extracted, **normalized_current}
            self.logger.debug(f"🔗 Merged entities (normalized): extracted={list(normalized_extracted.keys())}, current={list(normalized_current.keys())}, merged={list(entities.keys())}")
            
            # Defensive check
            if state.get("clarification_context"):
                self.logger.warning("⚠️ Unexpected: needs_clarification=False but clarification_context present")
            
            # Format tool results, history, and entities
            claim_data = self._format_tool_results(tool_results) if tool_results else "No claim data available"
            history_str = self._format_conversation_history(history) if history else "No previous conversation"
            entities_str = self._format_entities(entities) if entities else "No entities extracted"
            
            # Build normal response prompt
            prompt_template = ChatPromptTemplate.from_messages([
                ("user", """USER QUERY: {user_query}

INTENT: {intent}

=== ENTITIES (PLAIN NUMERIC IDENTIFIERS - CITE THESE EXACT NUMBERS AT BEGINNING OF YOUR RESPONSE) ===
{entities}

=== CLAIM DATA (ANSWER FROM THIS SECTION ONLY - MAY CONTAIN MASKED TOKENS, USE THEM AS-IS) ===
{claim_data}

=== CONVERSATION HISTORY (FOR UNDERSTANDING CONTEXT ONLY - DO NOT USE ANY DATA OR TOKENS FROM HERE) ===
{conversation_history}

RESPONSE INSTRUCTIONS:
1. Provide a targeted response that directly addresses the user's question
2. BEGIN with a natural conversational sentence starting with 'For claim' followed by the exact plain numeric 15-digit claim number from ENTITIES section above, then 'sequence' followed by the exact plain numeric 3-digit sequence number from ENTITIES section above, then a comma or colon before continuing
3. ANSWER the user's question using ONLY information from CLAIM DATA section
4. If you see masked tokens in CLAIM DATA, use them exactly as shown - they get replaced automatically
5. NEVER use any claim numbers, tokens, or data from CONVERSATION HISTORY - it belongs to different claims
6. Only include sections relevant to the specific query""")
            ])
            
            messages = prompt_template.format_messages(
                user_query=user_text,
                intent=intent,
                entities=entities_str,
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
    
    def _format_entities(self, entities: Dict[str, Any]) -> str:
        """
        Format extracted entities for LLM context.
        
        This ensures the LLM knows what identifiers the user provided,
        especially important for error messages when data is not found.
        
        Example: When user provides only "999" (sequence), but claim_number
        was extracted from history, both should be shown to the LLM so error
        messages correctly say "claim 123456789012345 and sequence 999".
        
        Args:
            entities: Dictionary of extracted entities from state
            
        Returns:
            str: Formatted entities string for LLM prompt
        """
        if not entities:
            return "(No entities extracted)"
        
        formatted = []
        for key, value in entities.items():
            if value:  # Only include non-empty values
                # Format lists as comma-separated (e.g., multiple claim_ids)
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                else:
                    value_str = str(value)
                
                # FIX: Strip "claim" prefix from claim-related entity values
                # Prevents duplication like "claim claim 233211748898001" in LLM responses
                # HIPAA Note: Claim IDs are business identifiers, not PHI
                if key in ('claim_number', 'claim_id', 'claim_ids', 'claimNumber', 'claimId'):
                    import re
                    numeric_match = re.search(r'\d+$', value_str)
                    if numeric_match:
                        value_str = numeric_match.group(0)
                
                # Map technical names to user-friendly labels
                friendly_key = key.replace("_", " ").title()
                formatted.append(f"- {friendly_key}: {value_str}")
        
        return "\n".join(formatted) if formatted else "(No entities extracted)"
    
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> tuple:
        """
        Generate response using Gemini LLM.
        
        Uses GenerateRequest and _generate_core from core/llm_connection.py
        following the established pattern in the codebase.
        
        Args:
            system_prompt: System instructions for LLM
            user_prompt: User query and context
            
        Returns:
            tuple: (response_text: str, llm_metadata: dict)
            
            llm_metadata contains:
            - finish_reason: str (STOP, MAX_TOKENS, SAFETY, etc.)
            - is_truncated: bool (True if truncated)
            - response_length: int
            - appears_incomplete: bool (heuristic for logging only)
            - thoughts: Optional[str] (chain of thought if enabled)
            - retry_attempted: bool (True if retry was needed)
            
        Raises:
            Exception: If generation fails (caught by caller)
        """
        # Initialize metadata dict to track LLM response details (Issue 1 & 2)
        llm_metadata = {
            "finish_reason": "UNKNOWN",
            "is_truncated": False,
            "response_length": 0,
            "appears_incomplete": False,
            "thoughts": None,
            "retry_attempted": False
        }
        
        try:
            # Build request with thinking mode if enabled (Issue 2)
            include_thoughts = getattr(settings, 'enable_thinking_mode', False)
            
            req = GenerateRequest(
                prompt=user_prompt,
                system_instruction=system_prompt,
                temperature=settings.llm_temperature,
                top_p=settings.top_p,
                max_output_tokens=settings.max_output_tokens,
                model=settings.llm_model,
                include_thoughts=include_thoughts
            )
            
            self.logger.info("🔮 Calling Gemini...")
            response = _generate_core(req)
            
            # Capture metadata from response (Issue 1: finish_reason, is_truncated)
            llm_metadata["finish_reason"] = response.finish_reason
            llm_metadata["is_truncated"] = response.is_truncated
            llm_metadata["response_length"] = len(response.text or "")
            llm_metadata["thoughts"] = response.thoughts  # Issue 2
            # Issue 3: Token usage metrics
            llm_metadata["prompt_tokens"] = response.prompt_tokens
            llm_metadata["completion_tokens"] = response.completion_tokens
            llm_metadata["total_tokens"] = response.total_tokens
            
            # Log truncation warning if detected
            if response.is_truncated:
                self.logger.warning(
                    f"⚠️ Response truncated: finish_reason={response.finish_reason}, "
                    f"length={llm_metadata['response_length']} chars"
                )
                # Check if truncated response appears incomplete (for logging only)
                llm_metadata["appears_incomplete"] = self._response_appears_incomplete(response.text)
            
            # RETRY ONLY IF RESPONSE IS COMPLETELY EMPTY (not for truncated responses)
            # Truncated responses may still be valid/useful, so we don't retry them
            if not response.text or len(response.text.strip()) == 0:
                self.logger.warning("⚠️ Empty response from Gemini, retrying with adjusted safety thresholds...")
                llm_metadata["retry_attempted"] = True
                
                # Use BLOCK_ONLY_HIGH for medical/pharmacy domain
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
                    safety_thresholds=safety_thresholds,
                    include_thoughts=include_thoughts
                )
                
                self.logger.info("🔄 Retrying with BLOCK_ONLY_HIGH thresholds...")
                response = _generate_core(req_retry)
                
                # Update metadata with retry response
                llm_metadata["finish_reason"] = response.finish_reason
                llm_metadata["is_truncated"] = response.is_truncated
                llm_metadata["response_length"] = len(response.text or "")
                if response.thoughts:
                    llm_metadata["thoughts"] = response.thoughts
            
            self.logger.info(f"✅ Response received: {len(response.text)} chars")
            return response.text, llm_metadata
            
        except Exception as e:
            self.logger.error(f"🚨 Generation error: {e}")
            raise
    
    def _response_appears_incomplete(self, text: str) -> bool:
        """
        Check if truncated response appears incomplete (for logging only).
        
        NOTE: This is ONLY called when is_truncated=True.
        It does NOT block responses - only adds metadata for debugging.
        
        Args:
            text: Response text to check
            
        Returns:
            bool: True if response appears incomplete
        """
        if not text or len(text.strip()) < 20:
            return True
        
        text = text.rstrip()
        
        # Common indicators of mid-sentence truncation
        incomplete_indicators = [
            text.endswith(':'),       # Ends with section header
            text.endswith('['),       # Ends with bracket (masked token cut off)
            text.endswith('•'),       # Ends with bullet point (kept per user request - for logging only)
            text.endswith(','),       # Ends with comma (mid-list)
            text.endswith('('),       # Ends with open parenthesis
        ]
        
        return any(incomplete_indicators)


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
        
        # Run in executor to avoid blocking (Gemini client is sync)
        # Using get_running_loop() - recommended for Python 3.10+ inside async functions
        loop = asyncio.get_running_loop()
        response_text, llm_metadata = await loop.run_in_executor(None, agent.generate_response, system_prompt, user_prompt)
        
        # =====================================================================
        # Issue 1: Log truncation warnings for observability
        # =====================================================================
        if llm_metadata.get("is_truncated"):
            logger.warning(
                f"⚠️ LLM response truncated: finish_reason={llm_metadata.get('finish_reason')}, "
                f"length={llm_metadata.get('response_length')} chars, "
                f"appears_incomplete={llm_metadata.get('appears_incomplete')}"
            )
        
        # =====================================================================
        # Issue 2: Fire-and-forget thinking log to MongoDB (ZERO LATENCY)
        # Thoughts logged to MongoDB, NOT stored in state for memory efficiency
        # =====================================================================
        thoughts = llm_metadata.get("thoughts")
        if thoughts and getattr(settings, 'log_thoughts_to_mongo', False):
            async def _log_thoughts_async():
                try:
                    store = PersistenceStoreFactory.get_instance(settings.persistence_store_type)
                    await store.log_thinking_process(
                        session_id=session_id,
                        request_id=request_id or str(uuid.uuid4()),
                        user_query=state.get("text", ""),
                        intent=state.get("intent", "unknown"),
                        thinking_content=thoughts,
                        final_response=response_text,
                        model=settings.llm_model,
                        user_id=user_id,
                        metadata={
                            "is_truncated": llm_metadata.get("is_truncated"),
                            "finish_reason": llm_metadata.get("finish_reason"),
                            "needs_clarification": needs_clarification
                        }
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Thought logging failed (non-fatal): {e}")
            
            # Fire-and-forget: don't await, just schedule
            asyncio.create_task(_log_thoughts_async())
            logger.debug("🧠 Thought logging scheduled (async)")
        
        # FIX: Monitor for remaining token-like patterns (helps debugging)
        # These will be cleaned up by postcheck's cleanup_remaining_tokens()
        import re
        remaining_tokens = re.findall(r'\[[A-Z_]+_[A-Za-z0-9]+\]', response_text or "")
        if remaining_tokens:
            logger.warning(f"⚠️ Response contains {len(remaining_tokens)} token-like patterns: {remaining_tokens[:3]}...")
            logger.warning("   These will be cleaned up by postcheck")
        
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
                    "needs_clarification": needs_clarification,
                    "finish_reason": llm_metadata.get("finish_reason"),  # Issue 1: Added
                    "is_truncated": llm_metadata.get("is_truncated")     # Issue 1: Added
                }
            )
        
        # Generate unique response_id for feedback tracking
        response_id = str(uuid.uuid4())
        logger.info(f"🆔 Generated response_id: {response_id}")
        
        # =====================================================================
        # Create SLIM metadata for state (EXCLUDE large thoughts for memory efficiency)
        # Thoughts are logged to MongoDB, not stored in AgentState
        # =====================================================================
        slim_llm_metadata = {
            "finish_reason": llm_metadata.get("finish_reason"),
            "is_truncated": llm_metadata.get("is_truncated"),
            "response_length": llm_metadata.get("response_length"),
            "appears_incomplete": llm_metadata.get("appears_incomplete"),
            "retry_attempted": llm_metadata.get("retry_attempted"),
            # Issue 3: Token usage (small integers, OK to include in state)
            "prompt_tokens": llm_metadata.get("prompt_tokens", 0),
            "completion_tokens": llm_metadata.get("completion_tokens", 0),
            "total_tokens": llm_metadata.get("total_tokens", 0),
            # NOTE: thoughts NOT included - logged to MongoDB instead
        }
        
        # Build result - always use 'response' field for both modes
        # The 'needs_clarification' flag in state already indicates if this is a question
        result = {
            "response": response_text,
            "response_id": response_id,
            "metadata": {
                **state.get("metadata", {}),
                "llm_metadata": slim_llm_metadata  # Issue 1: Small metadata only
            }
        }
        
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

        # Generate response_id even for errors (for tracking/feedback)
        error_response_id = str(uuid.uuid4())
        logger.info(f"🆔 Generated response_id for error: {error_response_id}")
        
        # Return graceful error state (pattern from agents/intent_agent.py)
        result = {
            "error": error.user_message,
            "response": error.user_message,
            "response_id": error_response_id,
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
    response_id = str(uuid.uuid4())
    
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
    return {
        "response": response,
        "response_id": response_id
    }
