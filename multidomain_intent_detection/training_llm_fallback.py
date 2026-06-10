"""
Training-only LLM fallback wrapper
====================================

This module is **scoped exclusively to multidomain_intent_detection/training.py**.
It re-implements the domain-aware LLM fallback path with two training-specific
differences from the production `prompt_templates.domain_prompts.llm_fallback`:

  1. Model: **Gemini 2.5 Flash** (instead of gemini-2.0-flash)
  2. Thinking mode: **enabled with a thinking budget of 6000 tokens**
     → the chain-of-thought (`thoughts`) is captured and returned

It is NOT imported anywhere except `training.py`, so production behavior is
unchanged.

Returns a tuple: (predicted_intent, llm_confidence, thoughts_text)
"""

import os
import json
import logging
from typing import List, Optional, Tuple

from multidomain_intent_detection.config import INTENT_TO_DOMAIN

logger = logging.getLogger(__name__)

# Lazy singletons
_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(
            vertexai=True,
            project=os.getenv("PROJECT_ID", "pbm-poc-coderev-genai-poc"),
            location=os.getenv("LOCATION", "us-central1"),
        )
    return _client


def training_llm_classify_with_thoughts(
    query: str,
    candidates: List[str],
    ensemble_intent: Optional[str] = None,
    ensemble_confidence: float = 0.0,
) -> Tuple[str, float, str]:
    """
    Training-only LLM fallback.

    Mirrors the production domain-aware flow (system prompt, candidate
    restriction, JSON parsing, intent validation) but:
      • model       = gemini-2.5-flash
      • thinking    = ON with thinking_budget = 6000 tokens
      • thoughts    = captured and returned as the third tuple element

    Returns
    -------
    (intent, confidence, thoughts_text)
    """
    from google.genai import types
    from prompt_templates.domain_prompts.llm_fallback import (
        _build_system_prompt,
        _build_user_prompt,
        _parse_llm_response,
    )
    from prompt_templates.domain_prompts.base_prompt import get_candidate_domains

    # Build top5 (decreasing dummy probs since we only have names)
    top5_intents = [
        (c, max(0.05, ensemble_confidence - i * 0.08))
        for i, c in enumerate(candidates)
    ]
    candidate_domains = get_candidate_domains(top5_intents)

    system_prompt = _build_system_prompt(candidates, candidate_domains, top5_intents)
    user_prompt = _build_user_prompt(query, ensemble_intent, ensemble_confidence, None)

    # ─── Thinking config: 6000-token budget (training-only) ────────────────
    # google-genai SDK ≥ 1.2.0 supports `thinking_budget`; if the installed
    # SDK only accepts `include_thoughts`, gracefully degrade.
    try:
        thinking_config = types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=6000,
        )
    except Exception:
        thinking_config = types.ThinkingConfig(include_thoughts=True)
        logger.warning(
            "Installed google-genai SDK does not support thinking_budget; "
            "falling back to include_thoughts only."
        )

    try:
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # training-only override
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048,  # leave headroom; thinking_budget=6000 is internal
                system_instruction=system_prompt,
                thinking_config=thinking_config,
            ),
        )
    except Exception as e:
        logger.error(f"[training] LLM 2.5 Flash call failed: {e}")
        fallback = candidates[0] if candidates else (ensemble_intent or "unknown")
        return fallback, 0.5, ""

    # ─── Extract thoughts + answer text ────────────────────────────────────
    thoughts_text = ""
    answer_text = ""
    try:
        if getattr(response, "candidates", None):
            cand = response.candidates[0]
            if getattr(cand, "content", None) and cand.content.parts:
                for part in cand.content.parts:
                    is_thought = getattr(part, "thought", False) or getattr(part, "thinking", False)
                    txt = getattr(part, "text", "") or ""
                    if not txt:
                        continue
                    if is_thought:
                        thoughts_text += txt
                    else:
                        answer_text += txt
        if not answer_text:
            answer_text = (response.text or "").strip()
    except Exception as e:
        logger.warning(f"[training] Failed to parse response parts: {e}")
        answer_text = (getattr(response, "text", "") or "").strip()

    # ─── Parse + validate ─────────────────────────────────────────────────
    try:
        parsed = _parse_llm_response(answer_text, candidates)
    except Exception as exc:
        logger.warning(f"[training] JSON parse failed: {exc} — using first candidate")
        return (candidates[0] if candidates else "unknown"), 0.5, thoughts_text
    predicted = parsed.get("intent", "")
    llm_confidence = float(parsed.get("confidence", 0.5))

    # Validate against candidates / known intents
    for c in candidates:
        if c.lower() == predicted.lower():
            return c, llm_confidence, thoughts_text
    if predicted in INTENT_TO_DOMAIN:
        return predicted, llm_confidence, thoughts_text

    logger.warning(f"[training] LLM returned unknown intent '{predicted}' — using first candidate")
    return (candidates[0] if candidates else "unknown"), llm_confidence, thoughts_text
