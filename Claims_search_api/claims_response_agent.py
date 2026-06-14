"""
Claims_search_api.claims_response_agent

Lightweight response agent that takes filtered + formatted claims context
and generates a natural-language answer using the existing Gemini LLM connection.

This does NOT modify or import the existing ResponseAgent in agents/.
It reuses only the shared LLM infrastructure (services.llm_connection).
"""

from typing import Any, Dict, Optional
from core.logger import get_logger

logger = get_logger(__name__)


def _get_llm_components():
    """Lazy-import LLM components to avoid hard dependency on google.genai at module load."""
    from services.llm_connection import (
        client as gemini_client,
        GenerateRequest,
        _generate_core,
    )
    return gemini_client, GenerateRequest, _generate_core


def _get_settings():
    """Lazy-import settings."""
    from config.config import settings
    return settings


# ---------------------------------------------------------------------------
# System prompt for the claims-search domain
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a pharmacy claims assistant. You have been given a filtered set of claims data based on the user's question.

YOUR TASK:
Analyze the claims data provided and answer the user's question in a clear, conversational, and helpful way.

RULES:
1. Answer ONLY from the data provided — do not invent or assume information.
2. Be conversational: "The last fill for Levothyroxine was on 2026-02-05 at CVS Pharmacy 00610..." — NOT "Based on the provided data, it appears that..."
3. When the user asks "when was X taken last", look at the FIRST claim in the list (claims are sorted newest-first) and cite its Fill Date.
4. When listing multiple claims, present them clearly with key details: claim number, fill date, drug name, status, and patient pay (if available).
5. If no claims matched the filter, say so clearly and suggest the user rephrase or check details.
6. Include relevant details the user didn't explicitly ask for if they add context (e.g., mention the claim was Rejected and the reject reason if asked about a drug's last fill).
7. Use plain conversational text. Do NOT use markdown formatting (no bold, italic, headings, or bullet points).
8. Keep responses concise but complete — typically 2-5 sentences for single-claim answers, more for multi-claim summaries.
9. Always mention the claim number when referencing specific claims so the user can follow up.
10. For pricing questions, always state the patient pay amount and mention if other pricing fields are available.
11. If the data shows a rejected claim, mention the reject reason and any resolution messages from the messages field.
12. Never say "I cannot help" — if the data doesn't answer the question, explain what data IS available and suggest how to refine the query.

IMPORTANT:
- Claims are sorted newest-first (by fill date). The first claim is the most recent.
- The "Showing: N" count tells you how many claims matched the filter.
- Member info is at the top if available.
"""


class ClaimsResponseAgent:
    """
    Generates natural-language responses for claims-search queries
    using the shared Gemini LLM connection.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def generate_response(
        self,
        user_query: str,
        claims_context: str,
        member_info: Optional[Dict[str, str]] = None,
        total_claims: int = 0,
        filtered_claims: int = 0,
    ) -> str:
        """
        Build prompt and call Gemini to generate a response.

        Args:
            user_query:      The user's original question.
            claims_context:  Pre-formatted claims text from llm_formatter.
            member_info:     Member CAGM dict (optional, for extra context).
            total_claims:    Total claims fetched from API.
            filtered_claims: Number of claims after filtering.

        Returns:
            LLM-generated response string.
        """
        # Build the user prompt
        user_prompt = self._build_user_prompt(
            user_query, claims_context, member_info,
            total_claims, filtered_claims,
        )

        self.logger.info(
            f"[ClaimsResponseAgent] Generating response for query: {user_query!r} "
            f"({filtered_claims}/{total_claims} claims in context)"
        )

        # Use mock response in mock mode
        settings = _get_settings()
        if settings.use_mock_llm:
            return self._mock_response(user_query, claims_context)

        # Call Gemini via the shared _generate_core
        try:
            _, GenerateRequest, _generate_core = _get_llm_components()

            request = GenerateRequest(
                prompt=user_prompt,
                system_instruction=_SYSTEM_PROMPT,
                temperature=settings.llm_temperature,
                top_p=settings.top_p,
                max_output_tokens=4096,  # Responses for this domain are short
            )
            result = _generate_core(request)
            response_text = result.text.strip()

            if not response_text:
                response_text = (
                    "I found the claims data but couldn't generate a clear answer. "
                    "Could you try rephrasing your question?"
                )

            self.logger.info(
                f"[ClaimsResponseAgent] Response generated "
                f"({result.completion_tokens} tokens, "
                f"truncated={result.is_truncated})"
            )
            return response_text

        except Exception as e:
            self.logger.error(f"[ClaimsResponseAgent] LLM error: {e}")
            return (
                "I encountered an issue generating a response. "
                "Please try again shortly."
            )

    def _build_user_prompt(
        self,
        user_query: str,
        claims_context: str,
        member_info: Optional[Dict[str, str]],
        total_claims: int,
        filtered_claims: int,
    ) -> str:
        """Construct the prompt sent to the LLM."""
        parts = []

        parts.append(f"USER QUERY: {user_query}")
        parts.append("")

        if member_info and "error" not in member_info:
            parts.append(
                f"MEMBER CONTEXT: "
                f"Member={member_info.get('member', 'N/A')}, "
                f"Carrier={member_info.get('carrierId', 'N/A')}, "
                f"Account={member_info.get('accountId', 'N/A')}, "
                f"Group={member_info.get('groupId', 'N/A')}"
            )
            parts.append("")

        parts.append(
            f"FILTER SUMMARY: {filtered_claims} of {total_claims} total claims "
            f"matched the user's query."
        )
        parts.append("")

        parts.append("CLAIMS DATA:")
        parts.append(claims_context)

        return "\n".join(parts)

    def _mock_response(self, user_query: str, claims_context: str) -> str:
        """Return a mock response for development/testing without LLM."""
        # Count claims in the context
        claim_count = claims_context.count("--- Claim ")
        if claim_count == 0:
            return "No claims were found matching your query."
        return (
            f"[MOCK] Found {claim_count} claim(s) matching your query: "
            f'"{user_query}". In production, the LLM would provide a '
            f"detailed conversational answer based on the claims data."
        )
