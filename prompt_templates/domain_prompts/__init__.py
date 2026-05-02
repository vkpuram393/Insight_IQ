"""
Domain-specific LLM fallback prompts for intent classification.

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │  Embedding Classifier (~91% accuracy, <1ms)                  │
  │  Confident → fast path                                       │
  ├──────────────────────────────────────────────────────────────┤
  │  Confidence < threshold?                                     │
  │  ↓                                                           │
  │  Domain Router → selects domain-specific prompt              │
  │  ↓                                                           │
  │  LLM Fallback with domain-expert prompt (~99%+ on fallback)  │
  └──────────────────────────────────────────────────────────────┘

Each domain module contains:
  - Exhaustive intent definitions with examples
  - Decision trees for disambiguation
  - Confusion pair resolution rules
  - Entity extraction guidance
"""

from prompt_templates.domain_prompts.llm_fallback import (
    llm_fallback_classify,
    get_domain_prompt,
    DOMAIN_PROMPT_MAP,
)

__all__ = [
    "llm_fallback_classify",
    "get_domain_prompt",
    "DOMAIN_PROMPT_MAP",
]
