"""
Multidomain Intent Detection — LLM Fallback
=============================================

When the ensemble classifier confidence is below threshold, this module
sends the top-5 candidates to Gemini Flash for disambiguation.

Used by classifier.py when:
  confidence < confidence_threshold  OR  margin < margin_threshold
"""

import os
import re
import json
import logging
from typing import List, Optional

from multidomain_intent_detection.config import INTENT_DESCRIPTIONS, INTENT_TO_DOMAIN

logger = logging.getLogger(__name__)


def llm_classify(
    query: str,
    candidates: List[str],
    ensemble_intent: Optional[str] = None,
    ensemble_confidence: float = 0.0,
) -> str:
    """LLM fallback — sends top-5 candidates to Gemini for disambiguation.

    Args:
        query:               Raw user query (with claim numbers).
        candidates:          Top-5 intent names from the ensemble.
        ensemble_intent:     The ensemble's top prediction (for context).
        ensemble_confidence: The ensemble's confidence (for context).

    Returns:
        Predicted intent name (guaranteed to be a valid intent).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=os.getenv("PROJECT_ID", "pbm-poc-coderev-genai-poc"),
        location=os.getenv("LOCATION", "us-central1"),
    )

    candidate_desc = "\n".join(
        f"- {name}: {INTENT_DESCRIPTIONS.get(name, name)}" for name in candidates
    )

    system_instruction = f"""You are an intent classification system for a Pharmacy Benefit Manager (PBM).
Classify the user query into exactly ONE of these candidate intents:

{candidate_desc}

KEY RULES:
- cap_api intents = details about ONE specific claim (references "this claim" or a claim number)
- claim_history_search intents = SEARCH/FILTER across MULTIPLE claims
- member_domain intents = member demographics, eligibility, coverage
- override_domain intents = PA management, override analysis
- benefits_api intents = benefit plan summary, history, finder

OUTPUT FORMAT (JSON only):
{{"intent": "<one of the candidates>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}}
"""

    user_prompt = f"Query: {query}\n"
    if ensemble_intent:
        user_prompt += (
            f"Note: Primary classifier suggested '{ensemble_intent}' "
            f"({ensemble_confidence:.0%} confidence) but was uncertain.\n"
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=100,
                system_instruction=system_instruction,
            ),
        )
        text = response.text.strip()
        # Strip markdown fences
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        json_match = re.search(r'\{[^{}]*"intent"[^{}]*\}', text, re.DOTALL)
        if json_match:
            llm_result = json.loads(json_match.group(0))
        else:
            llm_result = json.loads(text)

        predicted = llm_result.get("intent", "")

        # Validate against candidates (case-insensitive)
        for c in candidates:
            if c.lower() == predicted.lower():
                return c
        for c in candidates:
            if c.lower() in predicted.lower() or predicted.lower() in c.lower():
                return c
        if predicted in INTENT_TO_DOMAIN:
            return predicted

        logger.warning(f"LLM returned unknown intent '{predicted}', using ensemble pick")
        return candidates[0]

    except Exception as e:
        logger.error(f"LLM fallback failed: {e}")
        return candidates[0]
