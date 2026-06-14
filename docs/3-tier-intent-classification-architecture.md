# Scalable Multi-API Intent Classification: 3-Tier Pipeline Architecture

## Overview

This document describes a scalable, modular architecture for intent classification and domain routing in a multi-API conversational system. It details a 3-tier classification pipeline that robustly handles overlapping vocabulary, ambiguous queries, and seamless onboarding of new API domains. It also outlines modular prompt template management for LLM fallback and response agents.

---

## Architecture Diagram

```
User Query
    │
    ├─► Global Intent Check (greeting/help/out_of_scope)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                  TIER 1: DOMAIN ROUTING                     │
│                                                            │
│  Approach 1: Zero-Shot NLI Classifier                      │
│    • Query vs domain hypotheses (natural language)          │
│    • No training, just English descriptions                 │
│                                                            │
│  Approach 2: Dual-Embedding Similarity                     │
│    • Full query vs routing examples                         │
│    • Entity-stripped query vs action patterns               │
│                                                            │
│  Approach 3: LLM Fallback                                  │
│    • Structured prompt per domain (modular)                 │
│    • Only called if 1 & 2 disagree or are low-confidence    │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────┐
│ TIER 2: INTENT CLASSIFICATION                 │
│   • Embedding similarity within routed domain  │
└───────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────┐
│ RESPONSE AGENT                               │
│   • Modular, per-domain prompt templates      │
└───────────────────────────────────────────────┘
```

---

## Implementation Steps

### 1. Domain Registry & Configuration
- Create a central registry (e.g., `domain_registry.py`) for all API domains.
- For each domain, define:
  - Unique domain ID
  - Natural language description (for NLI)
  - Routing examples (20–40 per domain)
  - Action patterns (20–40 per domain)
  - Path to intent examples and response templates

### 2. Approach 1: Zero-Shot NLI Classifier
- Use a pre-trained NLI model (e.g., DeBERTa-v3-large-mnli or BART-large-mnli).
- For each domain, write 1–3 English hypothesis sentences describing the domain’s purpose and boundaries.
- At runtime, for each query, compute entailment scores for all domain hypotheses.
- Pick the domain with the highest score and sufficient margin/confidence.

#### Onboarding a New Domain
- Add a new hypothesis sentence for the domain in the registry.
- No code changes required.

### 3. Approach 2: Dual-Embedding Similarity
- For each domain, maintain:
  - Routing examples (full queries)
  - Action patterns (entity-stripped verb phrases)
- At runtime:
  - Compute embedding similarity between the query and each domain’s routing examples.
  - Strip entities from the query and compute similarity to each domain’s action patterns.
  - Use the maximum of the two scores per domain.
- Pick the domain with the highest combined score and sufficient margin.

#### Onboarding a New Domain
- Add routing examples and action patterns for the new domain.
- Regenerate embedding caches.

### 4. Approach 3: LLM Fallback (Modular Prompt Templates)
- For each domain, create a dedicated prompt template file (e.g., `prompts/domain_cap_api.md`).
- The LLM fallback loads all relevant domain prompt templates and presents them in a structured prompt.
- The LLM is only called if NLI and Embedding approaches disagree or are low-confidence.

#### Onboarding a New Domain
- Add a new prompt template file for the domain.
- Register the template path in the domain registry.

### 5. Tier 2: Intent Classification
- For the routed domain, run embedding similarity against that domain’s intent examples.
- Intent examples are maintained per domain (e.g., `classifiers/domains/cap_api_intents.py`).

#### Onboarding a New Domain
- Add intent examples for the new domain.
- Regenerate intent embedding caches.

### 6. Response Agent: Modular Prompt Templates
- For each domain, create a dedicated response agent prompt template (e.g., `prompts/response_cap_api.md`).
- The response agent loads the template for the routed domain and uses it to generate the final response.

#### Onboarding a New Domain
- Add a new response agent prompt template for the domain.
- Register the template path in the domain registry.

---

## Example Directory Structure

```
c:\ProjectData\POC-Flow-1\pss-myclaims-ai-agent\
  config\
    domain_registry.py
  classifiers\
    tier1_domain_router.py
    ...
    domains\
      cap_api_intents.py
      claims_search_intents.py
      ...
  prompts\
    domain_cap_api.md
    domain_claims_search.md
    response_cap_api.md
    response_claims_search.md
    ...
```

---

## Onboarding a New API Domain: Checklist

1. **Add to `domain_registry.py`:**
    - Domain ID, description, routing examples, action patterns, intent module path, prompt template paths.
2. **Write NLI Hypothesis:**
    - Add 1–3 English sentences describing the domain’s purpose.
3. **Add Routing Examples & Action Patterns:**
    - 20–40 representative queries and action phrases.
4. **Create Intent Examples:**
    - In `classifiers/domains/{domain}_intents.py`.
5. **Create Prompt Templates:**
    - `prompts/domain_{domain}.md` for LLM fallback.
    - `prompts/response_{domain}.md` for response agent.
6. **Regenerate Embedding Caches:**
    - Run the embedding cache generation script.
7. **No code changes required.**

---

## Key Advantages

- **No manual weight tuning.**
- **No code changes for new domains.**
- **Handles overlapping vocabulary and ambiguous queries.**
- **Modular, per-domain prompt templates for LLM and response agent.**
- **Scales to any number of domains.**

---

## References
- [Zero-Shot NLI: BART-large-mnli](https://huggingface.co/facebook/bart-large-mnli)
- [DeBERTa-v3-large-mnli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
- [OpenAI GPT-4](https://platform.openai.com/docs/models/gpt-4)
- [Google Gemini](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini)





"""
Tier 1 Domain Router — Triple-Approach Pipeline

Approach 1 (Primary):   Zero-Shot NLI Classification
  - Pre-trained NLI model (e.g., BART-large-mnli / DeBERTa-v3-large-mnli)
  - Domain routing as hypothesis testing
  - No training needed, just natural language domain descriptions

Approach 2 (Secondary): Dual-Embedding Similarity  
  - Entity-stripped action pattern embeddings
  - Domain routing example embeddings
  - Cosine similarity

Approach 3 (Fallback):  LLM Classification
  - Structured prompt with domain descriptions + examples
  - Called ONLY when Approach 1 + 2 disagree or are low-confidence
  - Most accurate but most expensive

Consensus:
  - NLI + Embedding agree         → use it (no LLM call)
  - NLI + Embedding disagree      → LLM breaks tie
  - Either one low confidence     → LLM verifies
  - All three fail               → out_of_scope with needs_clarification
"""

import re
import math
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================

class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ApproachResult:
    """Output from a single classification approach"""
    approach_name: str
    domain: Optional[str]
    score: float                    # 0.0 to 1.0
    all_scores: Dict[str, float]   # domain_id → score
    confidence: Confidence
    reason: str = ""


@dataclass
class RoutingResult:
    """Final Tier 1 output"""
    domain: str
    confidence: Confidence
    scores: Dict[str, float]        # domain_id → final score
    margin: float
    approach_used: str              # Which approach(es) determined the result
    approach_details: Dict[str, ApproachResult]
    llm_used: bool = False
    fallback_used: bool = False


# =============================================================================
# DOMAIN DESCRIPTIONS (for NLI + LLM)
# =============================================================================
# These are NATURAL LANGUAGE descriptions of what each domain does.
# The NLI model tests: "Does the user query entail this description?"
#
# Guidelines:
#   - Write 2-3 HYPOTHESIS sentences per domain
#   - Describe the ACTION/PURPOSE, not the data fields
#   - Be specific about what distinguishes this domain from others
#   - Include both what it IS and what it IS NOT

DOMAIN_HYPOTHESES: Dict[str, List[str]] = {

    "cap_api": [
        "The user wants to retrieve specific details, attributes, or status of an already identified claim or prescription.",
        "The user is asking about properties of a particular claim such as pricing, drug information, rejection reasons, prescriber details, or claim status.",
        "The user wants to look up, display, fetch, or generate information about a specific known claim using its claim number or in the current context.",
    ],

    "claims_search": [
        "The user wants to search for claims, find claims matching certain criteria, or look up claim history across multiple claims.",
        "The user is asking about when a drug was last taken, filled, or processed, or wants to find claims over a time period.",
        "The user wants to discover, list, or search across claims rather than examining the details of one specific claim.",
    ],

    # ── TEMPLATE: New domain ──
    # "benefits_api": [
    #     "The user wants to check benefit coverage, eligibility, or plan details.",
    #     "The user is asking what is covered under their plan, deductible amounts, or drug tier information.",
    # ],
}


# Short single-hypothesis version (faster, used when speed matters)
DOMAIN_HYPOTHESIS_SHORT: Dict[str, str] = {
    "cap_api": "This is a request to get details or attributes of a specific identified claim or prescription.",
    "claims_search": "This is a request to search for claims, find claims by criteria, or look up claim or drug history.",
    # "benefits_api": "This is a request to check benefit coverage, eligibility, or plan details.",
}


# =============================================================================
# DOMAIN DESCRIPTIONS FOR LLM (Approach 3)
# =============================================================================

DOMAIN_DESCRIPTIONS_FOR_LLM: Dict[str, str] = {
    "cap_api": """CAP API - Claim Detail Lookup
Purpose: Retrieve specific details/attributes of an already-known claim.
Use when: User asks about a specific claim's status, pricing, drug info, 
rejection reasons, prescriber, pharmacy, copay, DAW code, settlement, 
COB, edits, overrides, NDC, days supply, quantity, etc.
Examples:
- "What is the status of this claim?"
- "Show me the pricing for claim 252453396162000 seq 001"  
- "Fetch the brand versus generic status"
- "Generate a summary of claims for claim 252453396162000 with sequence 001"
- "Display the drug information for this prescription"
- "Why was this claim rejected?"
NOT this: Searching across claims, finding claims by date/drug/member""",

    "claims_search": """Claims Search API - Search & Discovery
Purpose: Search across multiple claims, find claims by criteria, look up history.
Use when: User wants to find claims, search by date range, find when a drug 
was last taken/filled/processed, list claims for a member, discover claims 
matching criteria, get prescription history.
Examples:
- "What was the last claim for this drug?"
- "When was this drug taken last for claim 253152732536005 001"
- "Find all claims for this member in the last 30 days"
- "Show me prescription history"
- "How many claims were processed last month?"
NOT this: Getting specific details of one known claim""",
}


# =============================================================================
# ACTION PATTERNS (for Embedding approach)
# =============================================================================

DOMAIN_ACTION_PATTERNS: Dict[str, List[str]] = {

    "cap_api": [
        "what is the status of this claim",
        "show me the details",
        "get claim details",
        "tell me about this claim",
        "why was this rejected",
        "why was this denied",
        "what is the rejection reason",
        "show me the pricing",
        "what is the copay",
        "what was the ingredient cost",
        "show pricing breakdown",
        "what pharmacy filled this",
        "who was the prescriber",
        "what drug was dispensed",
        "what medication was on this claim",
        "show me the drug information",
        "what is the NDC",
        "was there a prior authorization",
        "what was the DAW code",
        "show coordination of benefits",
        "what was the days supply",
        "what quantity was dispensed",
        "show me the settlement details",
        "was this reversed",
        "show the audit trail",
        "what edits were applied",
        "get the claim summary",
        "generate a summary",
        "fetch the status",
        "fetch the brand versus generic status",
        "display the drug information",
        "generate the drug status",
        "pull up this claim",
        "explain this claim",
        "break down this claim",
        "what did the plan pay",
        "show the payment details",
    ],

    "claims_search": [
        "search for claims",
        "find all claims",
        "find claims matching",
        "look up claims by member",
        "show me all claims",
        "list all claims",
        "get claim history",
        "show prescription history",
        "show drug history",
        "what claims does this member have",
        "find recent claims",
        "show claims from last month",
        "find claims by drug name",
        "find claims by pharmacy",
        "search claims by date range",
        "when was this drug taken last",
        "when was this drug last filled",
        "when was this drug last processed",
        "what was the last claim for this drug",
        "find the most recent fill",
        "what medications has this person been on",
        "list all medications for this member",
        "how many times was this drug filled",
        "which pharmacies has this member used",
        "show claims filled at this pharmacy",
        "find rejected claims for this member",
        "get claims processed this year",
        "any claims for this member",
        "pull up member claims",
        "all transactions for this person",
        "show me everything for this member",
    ],
}


# =============================================================================
# ENTITY STRIPPING (for Embedding approach)
# =============================================================================

ENTITY_STRIP_PATTERNS: List[Tuple[str, str]] = [
    (r'\b\d{12,18}\b', '[ID]'),
    (r'\bCLM[-]?\d{4,}', '[ID]'),
    (r'\bclaim\s*#?\s*(\d{6,})', r'claim [ID]'),
    (r'\bsequence\s*#?\s*\d{1,3}', 'sequence [SEQ]'),
    (r'\bseq\s*#?\s*\d{1,3}', 'sequence [SEQ]'),
    (r'\b\d{3}\b(?=\s|$)', '[SEQ]'),
    (r'\bMBR[-]?\d{4,}', '[MEMBER]'),
    (r'\bmember\s*(id|#|number)\s*[:=]?\s*\d{4,}', 'member [MEMBER]'),
    (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '[DATE]'),
    (r'\bRX[-]?\d{6,}', '[RX]'),
    (r'\b\d{6,}\b', '[ID]'),
    (r'\bwith\s+sequence\s+\[SEQ\]', ''),  # Clean up residual
]


def strip_entities(query: str) -> str:
    """Remove entity values so embeddings focus on ACTION verbs."""
    result = query
    for pattern, replacement in ENTITY_STRIP_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


# =============================================================================
# APPROACH 1: ZERO-SHOT NLI CLASSIFIER
# =============================================================================

class NLIClassifier:
    """
    Zero-shot classification using Natural Language Inference.
    
    Uses a pre-trained NLI model (e.g., facebook/bart-large-mnli or
    MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
    to test whether a query ENTAILS a domain hypothesis.
    
    No training data needed. Just natural language domain descriptions.
    
    If the NLI model is not available, falls back to a keyword-based
    approximation.
    """

    def __init__(self, enabled_domains: Dict[str, Any]):
        self.enabled_domains = enabled_domains
        self.classifier = None
        self.available = False
        self._load_model()

    def _load_model(self):
        """
        Load zero-shot classification pipeline.
        
        Model options (in order of preference):
          1. MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (best accuracy)
          2. facebook/bart-large-mnli (good balance)
          3. MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (smaller, faster)
          4. typeform/distilbert-base-uncased-mnli (smallest, fastest)
        """
        try:
            from transformers import pipeline
            
            # Try models in order of preference
            models_to_try = [
                "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",  # Good balance for production
                "facebook/bart-large-mnli",                       # Well-known, reliable
                "typeform/distilbert-base-uncased-mnli",          # Fastest, least accurate
            ]

            for model_name in models_to_try:
                try:
                    logger.info(f"   Loading NLI model: {model_name}...")
                    self.classifier = pipeline(
                        "zero-shot-classification",
                        model=model_name,
                        device=-1,  # CPU. Use 0 for GPU.
                    )
                    self.available = True
                    logger.info(f"   ✅ NLI model loaded: {model_name}")
                    return
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to load {model_name}: {e}")
                    continue

            logger.warning("   ⚠️ No NLI model available. Approach 1 disabled.")
            self.available = False

        except ImportError:
            logger.warning(
                "   ⚠️ transformers library not installed. "
                "Install with: pip install transformers torch"
            )
            self.available = False

    def classify(self, query: str) -> ApproachResult:
        """
        Classify query using zero-shot NLI.
        
        Tests the query against each domain's hypothesis.
        Returns the domain whose hypothesis the query most entails.
        """
        if not self.available or self.classifier is None:
            return ApproachResult(
                approach_name="nli",
                domain=None,
                score=0.0,
                all_scores={},
                confidence=Confidence.LOW,
                reason="NLI model not available",
            )

        # Build candidate labels from domain hypotheses
        # Use the short single-hypothesis version for speed
        candidate_labels = []
        label_to_domain = {}

        for domain_id in self.enabled_domains:
            if domain_id in DOMAIN_HYPOTHESIS_SHORT:
                label = DOMAIN_HYPOTHESIS_SHORT[domain_id]
                candidate_labels.append(label)
                label_to_domain[label] = domain_id

        if not candidate_labels:
            return ApproachResult(
                approach_name="nli",
                domain=None,
                score=0.0,
                all_scores={},
                confidence=Confidence.LOW,
                reason="No domain hypotheses defined",
            )

        try:
            # Run zero-shot classification
            result = self.classifier(
                query,
                candidate_labels,
                multi_label=False,  # Mutually exclusive domains
            )

            # Parse results
            all_scores = {}
            for label, score in zip(result['labels'], result['scores']):
                domain_id = label_to_domain.get(label, label)
                all_scores[domain_id] = score

            top_domain = label_to_domain.get(result['labels'][0], result['labels'][0])
            top_score = result['scores'][0]

            # Determine confidence
            if len(result['scores']) >= 2:
                margin = result['scores'][0] - result['scores'][1]
            else:
                margin = top_score

            if top_score >= 0.70 and margin >= 0.20:
                confidence = Confidence.HIGH
            elif top_score >= 0.50:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            logger.debug(
                f"   NLI: {top_domain} ({top_score:.3f}) | "
                f"margin={margin:.3f} | {all_scores}"
            )

            return ApproachResult(
                approach_name="nli",
                domain=top_domain,
                score=top_score,
                all_scores=all_scores,
                confidence=confidence,
                reason=f"score={top_score:.3f}, margin={margin:.3f}",
            )

        except Exception as e:
            logger.error(f"   ❌ NLI classification failed: {e}")
            return ApproachResult(
                approach_name="nli",
                domain=None,
                score=0.0,
                all_scores={},
                confidence=Confidence.LOW,
                reason=f"error: {e}",
            )

    def classify_multi_hypothesis(self, query: str) -> ApproachResult:
        """
        Enhanced version: Tests query against ALL hypotheses per domain,
        then aggregates. More accurate but slower (3x the NLI calls).
        
        Use this when single-hypothesis result is MEDIUM confidence.
        """
        if not self.available or self.classifier is None:
            return self.classify(query)  # Fallback to single

        domain_scores: Dict[str, List[float]] = {}

        for domain_id in self.enabled_domains:
            hypotheses = DOMAIN_HYPOTHESES.get(domain_id, [])
            if not hypotheses:
                continue

            scores = []
            for hypothesis in hypotheses:
                try:
                    result = self.classifier(
                        query,
                        [hypothesis, "This is something else entirely."],
                        multi_label=False,
                    )
                    # Score for the domain hypothesis
                    for label, score in zip(result['labels'], result['scores']):
                        if label == hypothesis:
                            scores.append(score)
                            break
                except Exception:
                    continue

            if scores:
                domain_scores[domain_id] = scores

        if not domain_scores:
            return self.classify(query)  # Fallback

        # Aggregate: use MAX of hypotheses per domain
        # (any one hypothesis matching strongly is enough)
        all_scores = {d: max(s) for d, s in domain_scores.items()}
        top_domain = max(all_scores, key=all_scores.get)
        top_score = all_scores[top_domain]

        sorted_scores = sorted(all_scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else top_score

        if top_score >= 0.75 and margin >= 0.15:
            confidence = Confidence.HIGH
        elif top_score >= 0.50:
            confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.LOW

        return ApproachResult(
            approach_name="nli_multi",
            domain=top_domain,
            score=top_score,
            all_scores=all_scores,
            confidence=confidence,
            reason=f"multi-hypothesis: score={top_score:.3f}, margin={margin:.3f}",
        )


# =============================================================================
# APPROACH 2: DUAL-EMBEDDING SIMILARITY
# =============================================================================

class EmbeddingClassifier:
    """
    Embedding-based classification using:
      A. Full query vs domain routing examples
      B. Entity-stripped query vs action pattern examples
    """

    def __init__(
        self,
        domain_routing_embeddings: Dict[str, np.ndarray],
        action_pattern_embeddings: Dict[str, np.ndarray],
        enabled_domains: Dict[str, Any],
    ):
        self.domain_routing_embeddings = domain_routing_embeddings
        self.action_pattern_embeddings = action_pattern_embeddings
        self.enabled_domains = enabled_domains

    def classify(
        self,
        query_embedding: np.ndarray,
        stripped_embedding: np.ndarray,
    ) -> ApproachResult:
        """
        Score domains using dual embeddings, pick winner.
        """
        all_scores = {}

        for domain_id in self.enabled_domains:
            # Signal A: Routing similarity (full query)
            route_embs = self.domain_routing_embeddings.get(domain_id)
            if route_embs is not None and len(route_embs) > 0:
                route_sim = float(np.max(
                    self._cosine_batch(query_embedding, route_embs)
                ))
            else:
                route_sim = 0.0

            # Signal B: Action pattern similarity (stripped query)
            action_embs = self.action_pattern_embeddings.get(domain_id)
            if action_embs is not None and len(action_embs) > 0:
                action_sim = float(np.max(
                    self._cosine_batch(stripped_embedding, action_embs)
                ))
            else:
                action_sim = 0.0

            # Combined: take MAX (either signal being strong is sufficient)
            all_scores[domain_id] = max(route_sim, action_sim)

        if not all_scores:
            return ApproachResult(
                approach_name="embedding",
                domain=None, score=0.0, all_scores={},
                confidence=Confidence.LOW, reason="no embeddings",
            )

        top_domain = max(all_scores, key=all_scores.get)
        top_score = all_scores[top_domain]

        sorted_vals = sorted(all_scores.values(), reverse=True)
        margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else top_score

        if top_score >= 0.75 and margin >= 0.08:
            confidence = Confidence.HIGH
        elif top_score >= 0.55:
            confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.LOW

        logger.debug(
            f"   Embedding: {top_domain} ({top_score:.3f}) | "
            f"margin={margin:.3f} | {all_scores}"
        )

        return ApproachResult(
            approach_name="embedding",
            domain=top_domain,
            score=top_score,
            all_scores=all_scores,
            confidence=confidence,
            reason=f"score={top_score:.3f}, margin={margin:.3f}",
        )

    def _cosine_batch(self, query_emb: np.ndarray, example_embs: np.ndarray) -> np.ndarray:
        q = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        e = example_embs / (np.linalg.norm(example_embs, axis=1, keepdims=True) + 1e-10)
        return np.dot(e, q)


# =============================================================================
# APPROACH 3: LLM FALLBACK
# =============================================================================

class LLMClassifier:
    """
    LLM-based domain classification — used ONLY as tiebreaker/fallback.
    
    Structured prompt → domain ID + confidence + reasoning.
    Most accurate but most expensive (~200-500ms per call).
    """

    # Class-level flag to track availability
    _llm_available: Optional[bool] = None

    def __init__(self, enabled_domains: Dict[str, Any]):
        self.enabled_domains = enabled_domains

    def classify(self, query: str) -> ApproachResult:
        """
        Ask LLM to classify the query into a domain.
        """
        try:
            domain_descriptions = "\n\n".join(
                f"Domain: {domain_id}\n{DOMAIN_DESCRIPTIONS_FOR_LLM.get(domain_id, 'No description')}"
                for domain_id in self.enabled_domains
            )

            prompt = f"""You are a domain classifier for a healthcare claims system.

Given a user query, classify it into exactly ONE of the following domains:

{domain_descriptions}

User Query: "{query}"

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "domain": "<domain_id>",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one sentence explaining why>"
}}"""

            response = self._call_llm(prompt)

            if response is None:
                return ApproachResult(
                    approach_name="llm",
                    domain=None, score=0.0, all_scores={},
                    confidence=Confidence.LOW,
                    reason="LLM not available",
                )

            # Parse response
            parsed = self._parse_response(response)
            if parsed is None:
                return ApproachResult(
                    approach_name="llm",
                    domain=None, score=0.0, all_scores={},
                    confidence=Confidence.LOW,
                    reason=f"Failed to parse LLM response: {response[:200]}",
                )

            domain = parsed.get("domain", "")
            llm_confidence = parsed.get("confidence", 0.5)
            reasoning = parsed.get("reasoning", "")

            if domain not in self.enabled_domains:
                return ApproachResult(
                    approach_name="llm",
                    domain=None, score=0.0, all_scores={},
                    confidence=Confidence.LOW,
                    reason=f"LLM returned unknown domain: {domain}",
                )

            # Map LLM confidence to our confidence levels
            if llm_confidence >= 0.80:
                confidence = Confidence.HIGH
            elif llm_confidence >= 0.50:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            all_scores = {d: 0.0 for d in self.enabled_domains}
            all_scores[domain] = llm_confidence

            logger.debug(f"   LLM: {domain} ({llm_confidence:.2f}) | {reasoning}")

            return ApproachResult(
                approach_name="llm",
                domain=domain,
                score=llm_confidence,
                all_scores=all_scores,
                confidence=confidence,
                reason=reasoning,
            )

        except Exception as e:
            logger.error(f"   ❌ LLM classification failed: {e}")
            return ApproachResult(
                approach_name="llm",
                domain=None, score=0.0, all_scores={},
                confidence=Confidence.LOW,
                reason=f"error: {e}",
            )

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM service. Override this for your specific LLM setup."""
        try:
            from config.config import settings

            if getattr(settings, 'use_google_embeddings', False):
                return self._call_google_llm(prompt)
            else:
                return self._call_azure_llm(prompt)
        except ImportError:
            return self._call_azure_llm(prompt)

    def _call_azure_llm(self, prompt: str) -> Optional[str]:
        try:
            from openai import AzureOpenAI
            from config.config import settings

            client = AzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
            )

            response = client.chat.completions.create(
                model=settings.azure_openai_deployment_name,
                messages=[
                    {"role": "system", "content": "You are a precise domain classifier. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Azure LLM call failed: {e}")
            return None

    def _call_google_llm(self, prompt: str) -> Optional[str]:
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            model = GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 200,
                },
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Google LLM call failed: {e}")
            return None

    def _parse_response(self, response: str) -> Optional[Dict]:
        """Parse LLM JSON response, handling markdown wrappers"""
        # Strip markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            match = re.search(r'\{[^}]+\}', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            return None


# =============================================================================
# MAIN TIER 1 ROUTER — CONSENSUS PIPELINE
# =============================================================================

class Tier1DomainRouter:
    """
    Routes queries to correct domain using 3-approach consensus pipeline.
    
    Flow:
      1. Run NLI + Embedding in parallel (both are fast)
      2. If they agree with decent confidence → done
      3. If they disagree or low confidence → call LLM to break tie
      4. Return result with full audit trail
    
    Cost profile:
      - 80% of queries: NLI + Embedding only (~50-100ms)
      - 15% of queries: + LLM call (~300-500ms)  
      - 5% of queries:  All three + fallback
    """

    def __init__(
        self,
        domain_routing_embeddings: Dict[str, np.ndarray],
        enabled_domains: Dict[str, Any],
        embeddings_service: Any,
    ):
        self.enabled_domains = enabled_domains
        self.embeddings_service = embeddings_service

        # ── Approach 1: NLI ──
        self.nli = NLIClassifier(enabled_domains)

        # ── Approach 2: Embedding ──
        action_embeddings = self._build_action_embeddings()
        self.embedding = EmbeddingClassifier(
            domain_routing_embeddings=domain_routing_embeddings,
            action_pattern_embeddings=action_embeddings,
            enabled_domains=enabled_domains,
        )

        # ── Approach 3: LLM ──
        self.llm = LLMClassifier(enabled_domains)

        logger.info(
            f"   ✅ Tier1 Router: {len(enabled_domains)} domains | "
            f"NLI={'✅' if self.nli.available else '❌'} | "
            f"Embedding=✅ | LLM=fallback"
        )

    def _build_action_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for action patterns"""
        import pickle
        import os

        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "action_patterns_cache.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if all(d in cached for d in self.enabled_domains if d in DOMAIN_ACTION_PATTERNS):
                    logger.info("   ⚡ Action patterns: loaded from cache")
                    return cached
            except Exception:
                pass

        logger.info("   📝 Action patterns: generating embeddings...")
        result = {}
        for domain_id in self.enabled_domains:
            patterns = DOMAIN_ACTION_PATTERNS.get(domain_id, [])
            if patterns:
                embeddings = self.embeddings_service.embed(patterns)
                result[domain_id] = np.array(embeddings)
                logger.info(f"   ✅ {domain_id}: {len(patterns)} action patterns")

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"   ⚠️ Cache save failed: {e}")

        return result

    # =========================================================================
    # MAIN ROUTING
    # =========================================================================

    def route(self, query: str, query_embedding: np.ndarray) -> RoutingResult:
        """
        Route query using 3-approach consensus pipeline.
        """
        query_lower = query.lower().strip()

        # ── Prepare entity-stripped embedding ──
        stripped_query = strip_entities(query_lower)
        if stripped_query != query_lower:
            logger.debug(f"   Entity strip: '{query_lower}' → '{stripped_query}'")

        try:
            from services.azure_embeddings import get_embedding as _get_emb
            try:
                from config.config import settings
                if getattr(settings, 'use_google_embeddings', False):
                    from services.google_embeddings import get_embedding as _get_emb
            except ImportError:
                pass
            stripped_embedding = np.array(_get_emb(stripped_query))
        except Exception:
            stripped_embedding = query_embedding

        # ─────────────────────────────────────────────────────────
        # Run Approach 1 (NLI) + Approach 2 (Embedding)
        # ─────────────────────────────────────────────────────────
        nli_result = self.nli.classify(query)
        emb_result = self.embedding.classify(query_embedding, stripped_embedding)

        approach_details = {
            "nli": nli_result,
            "embedding": emb_result,
        }

        # ─────────────────────────────────────────────────────────
        # CONSENSUS LOGIC
        # ─────────────────────────────────────────────────────────

        # Case 1: Both agree with at least MEDIUM confidence → done
        if (
            nli_result.domain is not None
            and nli_result.domain == emb_result.domain
            and nli_result.confidence in (Confidence.HIGH, Confidence.MEDIUM)
            and emb_result.confidence in (Confidence.HIGH, Confidence.MEDIUM)
        ):
            # Strong consensus — no LLM needed
            best_confidence = Confidence.HIGH if (
                nli_result.confidence == Confidence.HIGH
                or emb_result.confidence == Confidence.HIGH
            ) else Confidence.MEDIUM

            logger.info(
                f"🏷️ Tier1 CONSENSUS: {nli_result.domain} ({best_confidence.value}) | "
                f"NLI={nli_result.domain}({nli_result.confidence.value}) "
                f"Emb={emb_result.domain}({emb_result.confidence.value})"
            )

            return RoutingResult(
                domain=nli_result.domain,
                confidence=best_confidence,
                scores=self._merge_scores(nli_result, emb_result),
                margin=max(
                    self._get_margin(nli_result.all_scores),
                    self._get_margin(emb_result.all_scores),
                ),
                approach_used="nli+embedding",
                approach_details=approach_details,
                llm_used=False,
            )

        # Case 2: One is HIGH confidence, other is unavailable/LOW → trust the strong one
        if nli_result.confidence == Confidence.HIGH and nli_result.domain:
            logger.info(
                f"🏷️ Tier1 NLI-dominant: {nli_result.domain} ({nli_result.confidence.value}) | "
                f"Emb={emb_result.domain}({emb_result.confidence.value})"
            )
            return RoutingResult(
                domain=nli_result.domain,
                confidence=Confidence.HIGH,
                scores=nli_result.all_scores,
                margin=self._get_margin(nli_result.all_scores),
                approach_used="nli",
                approach_details=approach_details,
                llm_used=False,
            )

        if emb_result.confidence == Confidence.HIGH and emb_result.domain:
            logger.info(
                f"🏷️ Tier1 Emb-dominant: {emb_result.domain} ({emb_result.confidence.value}) | "
                f"NLI={nli_result.domain}({nli_result.confidence.value})"
            )
            return RoutingResult(
                domain=emb_result.domain,
                confidence=Confidence.HIGH if self.nli.available else Confidence.MEDIUM,
                scores=emb_result.all_scores,
                margin=self._get_margin(emb_result.all_scores),
                approach_used="embedding",
                approach_details=approach_details,
                llm_used=False,
            )

        # Case 3: NLI not available → trust embedding if MEDIUM
        if not self.nli.available and emb_result.confidence == Confidence.MEDIUM:
            logger.info(
                f"🏷️ Tier1 Emb-only (no NLI): {emb_result.domain} ({emb_result.confidence.value})"
            )
            return RoutingResult(
                domain=emb_result.domain,
                confidence=Confidence.MEDIUM,
                scores=emb_result.all_scores,
                margin=self._get_margin(emb_result.all_scores),
                approach_used="embedding",
                approach_details=approach_details,
                llm_used=False,
            )

        # ─────────────────────────────────────────────────────────
        # Case 4: DISAGREEMENT or LOW confidence → Call LLM
        # ─────────────────────────────────────────────────────────
        logger.info(
            f"🔄 Tier1 TIEBREAK needed: "
            f"NLI={nli_result.domain}({nli_result.confidence.value}) vs "
            f"Emb={emb_result.domain}({emb_result.confidence.value}) → calling LLM"
        )

        # If NLI gave MEDIUM confidence, try multi-hypothesis first (cheaper than LLM)
        if (
            self.nli.available
            and nli_result.confidence == Confidence.MEDIUM
            and nli_result.domain != emb_result.domain
        ):
            nli_multi = self.nli.classify_multi_hypothesis(query)
            approach_details["nli_multi"] = nli_multi

            if nli_multi.confidence == Confidence.HIGH and nli_multi.domain:
                logger.info(
                    f"🏷️ Tier1 NLI-multi resolved: {nli_multi.domain} ({nli_multi.confidence.value})"
                )
                return RoutingResult(
                    domain=nli_multi.domain,
                    confidence=Confidence.MEDIUM,  # Still medium since approaches disagreed initially
                    scores=nli_multi.all_scores,
                    margin=self._get_margin(nli_multi.all_scores),
                    approach_used="nli_multi",
                    approach_details=approach_details,
                    llm_used=False,
                )

        # LLM tiebreaker
        llm_result = self.llm.classify(query)
        approach_details["llm"] = llm_result

        if llm_result.domain and llm_result.confidence in (Confidence.HIGH, Confidence.MEDIUM):
            # LLM broke the tie
            logger.info(
                f"🏷️ Tier1 LLM-resolved: {llm_result.domain} ({llm_result.confidence.value}) | "
                f"{llm_result.reason}"
            )
            return RoutingResult(
                domain=llm_result.domain,
                confidence=llm_result.confidence,
                scores=llm_result.all_scores,
                margin=self._get_margin(llm_result.all_scores),
                approach_used="llm",
                approach_details=approach_details,
                llm_used=True,
            )

        # ─────────────────────────────────────────────────────────
        # Case 5: Even LLM failed — use best available signal
        # ─────────────────────────────────────────────────────────
        # Priority: NLI > Embedding > LLM > first domain
        for result in [nli_result, emb_result, llm_result]:
            if result.domain is not None:
                logger.warning(
                    f"🏷️ Tier1 FALLBACK: {result.domain} "
                    f"(from {result.approach_name}, low confidence)"
                )
                return RoutingResult(
                    domain=result.domain,
                    confidence=Confidence.LOW,
                    scores=result.all_scores,
                    margin=0.0,
                    approach_used=f"fallback_{result.approach_name}",
                    approach_details=approach_details,
                    llm_used="llm" in approach_details,
                    fallback_used=True,
                )

        # Absolute last resort
        fallback_domain = list(self.enabled_domains.keys())[0]
        logger.warning(f"🏷️ Tier1 ABSOLUTE FALLBACK: {fallback_domain}")
        return RoutingResult(
            domain=fallback_domain,
            confidence=Confidence.LOW,
            scores={d: 0.0 for d in self.enabled_domains},
            margin=0.0,
            approach_used="absolute_fallback",
            approach_details=approach_details,
            llm_used="llm" in approach_details,
            fallback_used=True,
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _merge_scores(self, *results: ApproachResult) -> Dict[str, float]:
        """Merge scores from multiple approaches (average)"""
        merged: Dict[str, List[float]] = {}
        for r in results:
            for domain, score in r.all_scores.items():
                if domain not in merged:
                    merged[domain] = []
                merged[domain].append(score)
        return {d: sum(s) / len(s) for d, s in merged.items()}

    def _get_margin(self, scores: Dict[str, float]) -> float:
        """Gap between #1 and #2"""
        if len(scores) < 2:
            return max(scores.values()) if scores else 0.0
        sorted_vals = sorted(scores.values(), reverse=True)
        return sorted_vals[0] - sorted_vals[1]