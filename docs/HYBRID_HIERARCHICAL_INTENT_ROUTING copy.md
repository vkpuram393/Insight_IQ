# Design Document: Hybrid Hierarchical Intent Routing

## Domain-Aware Cascading Router for Scalable, High-Accuracy Intent Detection

| Field           | Value                                    |
|-----------------|------------------------------------------|
| **Status**      | Proposed                                 |
| **Author**      | Insight_IQ Team                          |
| **Created**     | April 2026                               |
| **Supersedes**  | `3-tier-intent-classification-architecture.md` |
| **Applies To**  | `classifiers/`, `nodes/`, `langgraph_agent.py` |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Core Problem](#2-the-core-problem)
3. [Domain & Intent Registry](#3-domain--intent-registry)
4. [Proposed Architecture: The Cascading Router](#4-proposed-architecture-the-cascading-router)
5. [Detailed Tier Design](#5-detailed-tier-design)
6. [End-to-End Workflow](#6-end-to-end-workflow)
7. [Concrete Examples](#7-concrete-examples)
8. [Data Preparation & Centroid Generation](#8-data-preparation--centroid-generation)
9. [Implementation Plan](#9-implementation-plan)
10. [Directory Structure](#10-directory-structure)
11. [LangGraph Integration](#11-langgraph-integration)
12. [Adding a New Domain (Onboarding Playbook)](#12-adding-a-new-domain-onboarding-playbook)
13. [Performance & Cost Analysis](#13-performance--cost-analysis)
14. [Migration Strategy](#14-migration-strategy)
15. [Appendix](#appendix)

---

## 1. Executive Summary

This document proposes replacing the current flat embedding classifier (`embedded_classifier.py`) with a **domain-aware Cascading Router** that organizes intents into **4 API domains** and uses a three-tier classification pipeline:

- **Tier 1** — Sub-millisecond **centroid-based vector retrieval** first identifies the target **domain** (Cap-API, Claim History Search, Benefits API, General), then narrows to a Top-5 intent shortlist within that domain.
- **Tier 2** — A lightweight **cross-encoder reranker** validates the shortlist with word-level attention.
- **Tier 3** — **Gemini 2.5 Flash tool-calling** acts as the final arbiter for ambiguous cases only.

**Domain structure** (4 domains, 24 intents):

| Domain | API | Intents |
|--------|-----|---------|
| **Cap-API** | `/myclaims/claims/v1/claim/byclaimnumber` | `claim_status`, `multi_claim_summary`, `pharmacy_info`, `prescriber_info`, `pricing_info`, `reimbursement_info`, `rejection_reasons`, `settlement_info`, `rx_details`, `reversal_info`, `cob_info`, `generic_availability` |
| **Claim History Search** | `/myclaims/claims/v1/claim/history` | `claim_status`, `date_range_claims`, `drug_info`, `fill_date_info`, `drug_interaction_info`, `compound_info` |
| **Benefits API** | `/myclaims/benefits/v1/member` | `beneficiary_info`, `audit_info`, `approval_info` |
| **General** | None (handled locally) | `greeting`, `help`, `out_of_scope` |

> **Note**: `claim_status` is a **cross-domain intent** — it appears in both Cap-API (single claim lookup) and Claim History Search (claim search/filter context). The domain router determines which API to call based on query context.

The result is a system where **80–90% of requests never touch the LLM**, new domains/APIs can be onboarded in minutes without code changes, and semantic overlap between similar intents is resolved with near-human accuracy.

---

## 2. The Core Problem

### 2.1 Domain Overlap in Pharmacy Claims

In the current system, the same vocabulary appears across intents that belong to **different API domains**:

| Ambiguous Term | Intents (Domain) |
|----------------|-----------------|
| `"summary"` | `claim_status` (Cap-API + History), `multi_claim_summary` (Cap-API), `pricing_info` (Cap-API), `approval_info` (Benefits) |
| `"status"` | `claim_status` (Cap-API + History), `approval_info` (Benefits) |
| `"history"` | `audit_info` (Benefits), `date_range_claims` (History), `reversal_info` (Cap-API), `fill_date_info` (History) |
| `"paid"` / `"pay"` | `pricing_info` (Cap-API), `settlement_info` (Cap-API), `reimbursement_info` (Cap-API) |
| `"details"` | Nearly every intent across all domains |
| `"dispensed"` | `pharmacy_info` (Cap-API), `fill_date_info` (History), `rx_details` (Cap-API) |
| `"medication"` | `drug_info` (History), `compound_info` (History), `generic_availability` (Cap-API), `drug_interaction_info` (History) |

The overlap is worst **across domain boundaries** — the current flat classifier has no concept of domains, so it cannot distinguish whether "Show me the claim summary" should route to Cap-API (single claim) or Claim History Search (search results).

### 2.2 Current System Weaknesses

**Problem 1: Flat Classification — No Domain Awareness**
The current `CVSIntentEmbedded` classifier computes cosine similarity between a query embedding and 480 example embeddings (20 per intent × 24 intents), picking the single highest match. It treats all intents as equal peers, ignoring that they belong to **different APIs with different endpoints, parameters, and response schemas**.

```
Query: "Show me the pricing summary for this claim"

Current System Results (illustrative):
  pricing_info:        0.91  ← correct (Cap-API)
  claim_status:        0.89  ← confused by "summary" (Cap-API or History?)
  settlement_info:     0.87  ← confused by "pricing" (Cap-API)
  reimbursement_info:  0.86  ← confused by pricing context (Cap-API)
```

The margin between the correct intent and the runner-up is only **0.02** — a single word change could flip the result. Worse, the system doesn't know all four candidates are in the **same domain** (Cap-API), which is useful context for disambiguation.

**Problem 2: Manual Weight Tuning**
The keyword classifier (`keyword_classifier.py`) contains hundreds of lines of manual weight adjustments:

```python
# From keyword_classifier.py (actual code comments):
# 'member' LOWERED to 0.3 — was causing beneficiary_info to steal from drug_info
# Removed 'medication' — it was causing drug_info to win over compound_info
# 'why' removed — too generic, was stealing from multiple intents
```

Every new intent or example change risks breaking other intents.

**Problem 3: Cross-Domain Misrouting**
When a user asks "Show me the claim history," the flat classifier might route to `audit_info` (Benefits API) instead of `date_range_claims` (Claim History Search). These are **different APIs** with different endpoints — a wrong domain means the wrong API is called entirely.

**Problem 4: LLM Judge Overhead**
When the confidence is low, the system currently sends the full query to Gemini for re-classification across all intents. This is expensive ($0.075/1M input tokens) and adds 500–1500ms latency. With domain-first routing, the LLM only needs to disambiguate between 3–12 intents within a single domain.

---

## 3. Domain & Intent Registry

### 3.1 Domain Definitions

The system organizes all intents into **4 API domains**. Each domain maps to a specific backend API with its own endpoint, parameters, and response schema.

```python
DOMAIN_REGISTRY = {
    "cap_api": {
        "name": "Cap-API",
        "description": (
            "Capability API for single-claim operations: claim status, pricing, "
            "pharmacy details, prescriber information, rejection analysis, "
            "settlement codes, reversals, coordination of benefits, "
            "reimbursement, prescription details, and generic alternatives."
        ),
        "api_endpoint": "/myclaims/claims/v1/claim/byclaimnumber",
        "intents": [
            "claim_status",         # General claim summary / adjudication outcome
            "multi_claim_summary",  # Summary of ALL claims for a member
            "pharmacy_info",        # Dispensing pharmacy details
            "prescriber_info",      # Prescribing physician details
            "pricing_info",         # Copay, ingredient cost, patient pay
            "reimbursement_info",   # Amount paid TO the pharmacy
            "rejection_reasons",    # Why the claim was denied + resolution steps
            "settlement_info",      # Pharmacy response/feedback codes
            "rx_details",           # RX number, quantity, days supply
            "reversal_info",        # Reversals, R&R, manual adjustments
            "cob_info",             # Coordination of Benefits / dual coverage
            "generic_availability", # Generic alternatives / therapeutic equivalents
        ],
    },
    "claim_history_search": {
        "name": "Claim History Search",
        "description": (
            "Claim history and drug search API: searching claims by date range, "
            "drug information lookups, fill date queries, drug interaction "
            "reviews (DUR), and compound medication details."
        ),
        "api_endpoint": "/myclaims/claims/v1/claim/history",
        "intents": [
            "claim_status",         # Claim status within search context
            "date_range_claims",    # Claims in a date range / accumulation history
            "drug_info",            # Drug name, NDC, GPI, formulary status
            "fill_date_info",       # Prescription fill/dispense date
            "drug_interaction_info",# DUR edits, drug interaction alerts
            "compound_info",        # Compound medication / MIC breakdown
        ],
    },
    "benefits_api": {
        "name": "Benefits API",
        "description": (
            "Member benefits API: benefit phase and coverage tier, "
            "audit trail and change history, approval details including "
            "transition fill (TF), plan overrides, and BPG configuration."
        ),
        "api_endpoint": "/myclaims/benefits/v1/member",
        "intents": [
            "beneficiary_info",     # Member benefit phase, coverage, accumulations
            "audit_info",           # Audit trail, change history, modification log
            "approval_info",        # Approvals, TF, BPG, plan overrides
        ],
    },
    "general": {
        "name": "General",
        "description": (
            "Non-API intents: greetings, help requests, and queries "
            "unrelated to pharmacy claims."
        ),
        "api_endpoint": None,
        "intents": [
            "greeting",             # Hello, hi, welcome
            "help",                 # How to use the system
            "out_of_scope",         # Unrelated queries
        ],
    },
}
```

### 3.2 Cross-Domain Intent: `claim_status`

`claim_status` is the only intent that appears in **two domains** (Cap-API and Claim History Search). The domain router disambiguates based on the query context:

| Query Pattern | Routed Domain | Reasoning |
|---------------|---------------|-----------|
| "Show claim status for CLM12345" | **Cap-API** | Single claim number → direct lookup |
| "What's the adjudication outcome?" | **Cap-API** | Status of a specific claim |
| "Show me all claims from January" | **Claim History Search** | Date range → search API |
| "List claims for this drug" | **Claim History Search** | Drug-based search |
| "What happened on claim sequence 3?" | **Cap-API** | Specific sequence → direct lookup |

### 3.3 Intent Count Summary

| Domain | Intents | Unique Intents | Shared |
|--------|---------|----------------|--------|
| Cap-API | 12 | 12 | `claim_status` (shared) |
| Claim History Search | 6 | 5 + 1 shared | `claim_status` (shared) |
| Benefits API | 3 | 3 | — |
| General | 3 | 3 | — |
| **Total** | **24** | **23 unique + 1 shared** | |

---

## 4. Proposed Architecture: The Cascading Router

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          USER QUERY ARRIVES                                     │
│                    "Show me the pricing summary for this claim"                 │
└─────────────────┬───────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TIER 0: GENERAL GROUP FAST-PATH                                ⏱ <0.1ms       │
│                                                                                 │
│  • Pattern match for greeting/help/out_of_scope                                 │
│  • If matched → handle directly, skip all tiers                                 │
│  • "hi", "hello", "help me" → resolved instantly                               │
│                                                                                 │
│  Result: Not a general query → continue to Tier 1                               │
└─────────────────┬───────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: DOMAIN ROUTING + INTENT SHORTLISTING                   ⏱ <1ms         │
│                                                                                 │
│  Step 1: Domain Selection                                                       │
│  • Cosine similarity against 4 domain centroids                                 │
│  • Result: Cap-API (0.86) ← winner                                              │
│                                                                                 │
│  Step 2: Intent Shortlisting (within winning domain)                            │
│  • Cosine similarity against 12 Cap-API intent centroids                        │
│  • Return Top-5: [pricing_info: 0.84, claim_status: 0.79,                      │
│                   settlement_info: 0.76, reimbursement_info: 0.73,             │
│                   cob_info: 0.68]                                               │
│                                                                                 │
│  🔑 KEY: Domain routing reduces search space from 24 intents → 3-12 intents    │
└─────────────────┬───────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TIER 2: CROSS-ENCODER RERANKING                                ⏱ 5–15ms       │
│                                                                                 │
│  • Take user query + Top-5 candidate descriptions (within selected domain)      │
│  • Cross-encoder scores each (query, description) pair                          │
│  • Word-level attention resolves "pricing summary" vs "claim summary"           │
│                                                                                 │
│  Result: pricing_info: 0.94 ✅  (HIGH CONFIDENCE → route immediately)           │
│          claim_status: 0.41                                                      │
│          settlement_info: 0.28                                                   │
│          reimbursement_info: 0.22                                                │
│          cob_info: 0.09                                                          │
│                                                                                 │
│  Decision: Score 0.94 > 0.85 threshold                                          │
│  → ROUTE TO Cap-API / pricing_info                                              │
└─────────────────┬───────────────────────────────────────────────────────────────┘
                  │
                  │  (Only if score < 0.85)
                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TIER 3: GEMINI 2.5 FLASH TOOL-CALLING                         ⏱ 200–800ms    │
│                                                                                 │
│  • Receive query + Top-5 intent schemas FROM THE SELECTED DOMAIN                │
│  • Gemini selects the best tool OR asks a clarifying question                   │
│  • Parameter validation: "Is claim_number present? Is sequence needed?"          │
│                                                                                 │
│  Context is TINY: only 3-12 tools from one domain (not all 24)                  │
│                                                                                 │
│  Example Gemini Response:                                                       │
│  {                                                                              │
│    "tool": "pricing_info",                                                      │
│    "domain": "cap_api",                                                         │
│    "confidence": 0.97,                                                          │
│    "reasoning": "User explicitly asked for pricing breakdown"                   │
│  }                                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Detailed Tier Design

### 5.1 Tier 0: General Group Fast-Path

Before any embedding computation, a fast pattern check handles trivial intents:

```python
import re

GENERAL_PATTERNS = {
    "greeting": re.compile(
        r"^(hi|hello|hey|howdy|good\s*(morning|afternoon|evening|day)|"
        r"greetings|welcome|sup|yo)\b",
        re.IGNORECASE
    ),
    "help": re.compile(
        r"\b(help|how\s+do\s+i|guide|instructions|steps\s+to|"
        r"how\s+to\s+(submit|file|avoid))\b",
        re.IGNORECASE
    ),
}

def check_general_fast_path(query: str) -> str | None:
    """
    Returns intent name if query matches a general pattern, else None.
    ~0.01ms — pure regex, no embeddings.
    """
    query_stripped = query.strip()
    
    # Very short queries (≤3 words) that match greeting patterns
    if len(query_stripped.split()) <= 3:
        for intent, pattern in GENERAL_PATTERNS.items():
            if pattern.search(query_stripped):
                return intent
    
    return None  # Not a general query → proceed to Tier 1
```

### 5.2 Tier 1: Domain Routing + Intent Shortlisting

Tier 1 operates in **two steps**: first select the domain, then shortlist intents within it.

#### What Is a Centroid?

A **centroid** is the mathematical average of multiple embedding vectors. We maintain **two levels of centroids**:

1. **Domain centroids** (4 vectors) — the mean of ALL intent centroids within a domain
2. **Intent centroids** (24 vectors) — the mean of 50 synthetic prompts per intent

```python
# Pseudocode: Two-Level Centroid Generation
import numpy as np

# Step 1: Generate intent centroids (50 prompts each)
pricing_centroid = np.mean([get_embedding(p) for p in pricing_prompts], axis=0)
claim_status_centroid = np.mean([get_embedding(p) for p in claim_status_prompts], axis=0)
# ... for all 24 intents

# Step 2: Generate domain centroids (average of their intent centroids)
cap_api_centroid = np.mean([
    pricing_centroid, claim_status_centroid, pharmacy_centroid,
    prescriber_centroid, reimbursement_centroid, rejection_centroid,
    settlement_centroid, rx_details_centroid, reversal_centroid,
    cob_centroid, generic_availability_centroid, multi_claim_centroid
], axis=0)

history_centroid = np.mean([
    claim_status_centroid, date_range_centroid, drug_info_centroid,
    fill_date_centroid, drug_interaction_centroid, compound_centroid
], axis=0)

benefits_centroid = np.mean([
    beneficiary_centroid, audit_centroid, approval_centroid
], axis=0)

general_centroid = np.mean([
    greeting_centroid, help_centroid, out_of_scope_centroid
], axis=0)
```

#### Why Domain-First Routing?

| Aspect | Flat (Current) | Domain-First (Proposed) |
|--------|---------------|------------------------|
| **Centroid comparisons per query** | 24 intent centroids | 4 domain + 3–12 intent = 7–16 |
| **Search space for reranker** | All 24 intents | 3–12 within domain |
| **Cross-domain confusion** | Common (pricing vs audit) | Eliminated at Tier 1 |
| **Gemini context (Tier 3)** | 24 tool schemas | 3–12 tool schemas |
| **Memory footprint** | ~24 × 768 = 72KB | ~28 × 768 = 84KB (negligible diff) |

#### Implementation

```python
import numpy as np
from typing import List, Tuple, Dict, Optional

class DomainAwareCentroidRouter:
    """Tier 1: Two-level centroid routing — domain first, then intent."""
    
    def __init__(
        self,
        domain_centroids: Dict[str, np.ndarray],      # 4 domain centroids
        intent_centroids: Dict[str, np.ndarray],       # 24 intent centroids
        domain_intent_map: Dict[str, List[str]],       # domain → [intents]
    ):
        # Domain-level routing
        self.domain_names = list(domain_centroids.keys())
        self.domain_matrix = np.stack([domain_centroids[d] for d in self.domain_names])
        
        # Intent-level routing (organized by domain)
        self.domain_intent_map = domain_intent_map
        self.intent_centroids = intent_centroids
        
        # Pre-build per-domain intent matrices for fast lookup
        self.domain_intent_matrices = {}
        self.domain_intent_names = {}
        for domain, intents in domain_intent_map.items():
            names = [i for i in intents if i in intent_centroids]
            self.domain_intent_names[domain] = names
            self.domain_intent_matrices[domain] = np.stack(
                [intent_centroids[i] for i in names]
            )
    
    def route(
        self,
        query_embedding: np.ndarray,
        top_k_intents: int = 5
    ) -> Dict:
        """
        Two-level routing: domain → intent shortlist.
        
        Returns:
            {
                "domain": "cap_api",
                "domain_score": 0.86,
                "domain_scores": {"cap_api": 0.86, "benefits_api": 0.72, ...},
                "intent_shortlist": [("pricing_info", 0.84), ("claim_status", 0.79), ...],
            }
        """
        # Step 1: Domain routing (4 cosine ops)
        domain_scores = self.domain_matrix @ query_embedding
        best_domain_idx = int(np.argmax(domain_scores))
        best_domain = self.domain_names[best_domain_idx]
        
        # Step 2: Intent shortlisting within winning domain
        intent_matrix = self.domain_intent_matrices[best_domain]
        intent_names = self.domain_intent_names[best_domain]
        
        intent_scores = intent_matrix @ query_embedding
        k = min(top_k_intents, len(intent_names))
        top_indices = np.argpartition(intent_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(intent_scores[top_indices])[::-1]]
        
        shortlist = [(intent_names[i], float(intent_scores[i])) for i in top_indices]
        
        return {
            "domain": best_domain,
            "domain_score": float(domain_scores[best_domain_idx]),
            "domain_scores": {
                self.domain_names[i]: float(domain_scores[i])
                for i in range(len(self.domain_names))
            },
            "intent_shortlist": shortlist,
        }
```

**Performance**: Domain routing = `(4, 768) @ (768,)` + intent routing = `(12, 768) @ (768,)` → total **<0.1ms**.

---

### 5.3 Tier 2: Cross-Encoder Reranking

#### Why Cross-Encoders?

The current system uses **bi-encoders** (embed query and examples separately, then compare). This misses **word-level interactions**. A cross-encoder processes the query and candidate description **together**, allowing it to understand that:

- "pricing **summary**" → `pricing_info` (the word "pricing" modifies "summary")
- "claim **summary**" → `claim_status` (the word "claim" modifies "summary")

A bi-encoder embeds "summary" the same way in both cases.

#### Cross-Encoder Architecture

```
Input: "[CLS] Show me the pricing summary [SEP] Cap-API > Pricing: copay, 
        ingredient cost, dispensing fee, patient pay breakdown [SEP]"
                    ↓
           Cross-Encoder (BGE-Reranker-v2-m3 or ms-marco-MiniLM-L-6-v2)
                    ↓
           Relevance Score: 0.94
```

> **Domain context in descriptions**: Cross-encoder descriptions are prefixed with the domain name, giving the model additional context about which API the intent belongs to.

#### Implementation

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Tuple, Dict

class CrossEncoderReranker:
    """Tier 2: Precision reranking with word-level attention (domain-aware)."""
    
    # Recommended models (sorted by accuracy vs speed tradeoff):
    # 1. BAAI/bge-reranker-v2-m3         — Best accuracy, ~15ms/pair
    # 2. cross-encoder/ms-marco-MiniLM-L-6-v2 — Good balance, ~5ms/pair
    # 3. BAAI/bge-reranker-base           — Fastest, ~3ms/pair
    
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        
        # Domain-aware intent descriptions for cross-encoder comparison
        self.intent_descriptions = INTENT_DESCRIPTIONS  # See below
    
    def rerank(
        self, 
        query: str, 
        domain: str,
        candidates: List[Tuple[str, float]],  # From Tier 1
        threshold: float = 0.85
    ) -> Tuple[str, str, float, bool]:
        """
        Rerank Top-K candidates using cross-encoder.
        
        Args:
            query: User's natural language query
            domain: Selected domain from Tier 1 (e.g., "cap_api")
            candidates: List of (intent_name, tier1_score) from centroid retrieval
            threshold: Confidence threshold for immediate routing
            
        Returns:
            (best_intent, domain, score, needs_escalation)
        """
        pairs = []
        intent_names = []
        
        for intent_name, _ in candidates:
            description = self.intent_descriptions.get(intent_name, intent_name)
            pairs.append((query, description))
            intent_names.append(intent_name)
        
        # Batch encode all pairs
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(scores).numpy()
        
        ranked = sorted(zip(intent_names, scores), key=lambda x: x[1], reverse=True)
        best_intent, best_score = ranked[0]
        
        # Check confidence gap between #1 and #2
        gap = (best_score - ranked[1][1]) if len(ranked) > 1 else best_score
        
        needs_escalation = best_score < threshold or gap < 0.15
        
        return best_intent, domain, float(best_score), needs_escalation
```

#### Intent Descriptions for Cross-Encoder (Domain-Organized)

The cross-encoder compares the query against **rich natural language descriptions**, organized by domain:

```python
INTENT_DESCRIPTIONS = {

    # ========================================================================
    # CAP-API DOMAIN (12 intents)
    # Endpoint: /myclaims/claims/v1/claim/byclaimnumber
    # ========================================================================
    
    "claim_status": (
        "General claim status, adjudication outcome, processing result, "
        "and overall claim summary including whether the claim was paid, "
        "rejected, or is pending."
    ),
    "multi_claim_summary": (
        "Summary of ALL claims for a member, complete claim list, "
        "multiple claim records, and full claim history overview."
    ),
    "pharmacy_info": (
        "Dispensing pharmacy name, location, address, NCPDP number, "
        "and store details where the prescription was filled."
    ),
    "prescriber_info": (
        "Prescribing physician name, NPI number, credentials, "
        "contact information, and ordering provider details."
    ),
    "pricing_info": (
        "Pricing and cost information for a pharmacy claim including copay amount, "
        "ingredient cost, dispensing fee, patient pay, manufacturer rebate, "
        "and total out-of-pocket cost breakdown."
    ),
    "reimbursement_info": (
        "Reimbursement payment details for paper claims including the amount "
        "paid to the pharmacy, reimbursement rationale, and payment calculation."
    ),
    "rejection_reasons": (
        "Specific rejection reasons, failed edit codes, denial explanations, "
        "and actionable steps to resolve or overturn a rejected pharmacy claim."
    ),
    "settlement_info": (
        "Settlement codes and pharmacy response information sent back to the "
        "dispensing pharmacy after claim adjudication, including feedback codes "
        "and response messages."
    ),
    "rx_details": (
        "Prescription RX number, fill number, quantity dispensed, "
        "days supply, drug strength, and refill information."
    ),
    "reversal_info": (
        "Claim reversal details, manual adjustments, reverse and resubmit "
        "(R&R) information, modification history, and resubmission status."
    ),
    "cob_info": (
        "Coordination of Benefits (COB) pricing, other insurance details, "
        "secondary payer information, dual coverage breakdown, "
        "and primary/secondary insurance adjudication."
    ),
    "generic_availability": (
        "Generic alternative medications, therapeutic equivalents, "
        "formulary-approved substitutes, and cheaper drug options "
        "available for the prescribed medication."
    ),

    # ========================================================================
    # CLAIM HISTORY SEARCH DOMAIN (6 intents)
    # Endpoint: /myclaims/claims/v1/claim/history
    # Note: claim_status is shared with Cap-API (defined above)
    # ========================================================================
    
    "date_range_claims": (
        "Claims within a specific date range, deductible-contributing claims, "
        "accumulation history, and prescription fill history over time."
    ),
    "drug_info": (
        "Drug name, NDC code, GPI, therapeutic class, formulary status, "
        "medication tier, and drug classification assigned by RxClaim."
    ),
    "fill_date_info": (
        "Date the prescription was filled, dispensing date, "
        "service date, and fill timestamp."
    ),
    "drug_interaction_info": (
        "Drug utilization review (DUR) edits, drug interaction alerts, "
        "clinical screening results, and override details for "
        "potential medication conflicts."
    ),
    "compound_info": (
        "Compound medication details, Most Ingredient Cost (MIC) breakdown, "
        "individual ingredient costs, funded versus unfunded components, "
        "and compound formulation information."
    ),

    # ========================================================================
    # BENEFITS API DOMAIN (3 intents)
    # Endpoint: /myclaims/benefits/v1/member
    # ========================================================================
    
    "beneficiary_info": (
        "Member benefit phase, coverage tier, eligibility status, "
        "accumulation rules, linked LOE information, and whether medical "
        "dollars contribute to the member's accumulations."
    ),
    "audit_info": (
        "Audit trail and change history for a claim, including "
        "modification records, edit timestamps, and who made changes."
    ),
    "approval_info": (
        "Claim approval details including plan overrides, transition fill (TF) "
        "status and type, BPG configuration, accumulation bypass, "
        "and Smart PA or Member PA applied during adjudication."
    ),

    # ========================================================================
    # GENERAL DOMAIN (3 intents)
    # No API endpoint — handled locally
    # ========================================================================
    
    "greeting": (
        "Casual greeting, hello, hi, welcome message, "
        "or conversation starter."
    ),
    "help": (
        "Help request, guidance on claim submission, instructions "
        "for using the system, and how to avoid claim rejections."
    ),
    "out_of_scope": (
        "Question unrelated to pharmacy claims, prescriptions, "
        "insurance benefits, or healthcare — such as weather, "
        "recipes, sports, travel, or random text."
    ),
}
```

---

### 5.4 Tier 3: Gemini 2.5 Flash Tool-Calling

When Tier 2 cannot decide with high confidence, we pass the **narrowed shortlist from the selected domain** (not all 24 intents) to Gemini as a **tool-calling** prompt. This keeps the context window small and the cost minimal.

#### Tool-Calling Prompt Design

```python
TIER3_SYSTEM_PROMPT = """You are an intent routing assistant for a pharmacy claims system.

The user's query has been routed to the "{domain_name}" domain ({domain_description}).
API Endpoint: {api_endpoint}

Given the user query and the shortlist of candidate tools from this domain, 
select the BEST matching tool.
If the query is ambiguous, ask the user a clarifying question instead of guessing.

Rules:
1. Select exactly ONE tool if you are confident.
2. If the user's query could match multiple tools, ask a clarifying question.
3. If required parameters (like claim_number) are missing, note them.
4. Never hallucinate tool names — only use tools from the provided list.
5. If the query doesn't belong to this domain at all, return "out_of_scope".
"""

def build_tier3_tools(domain: str, candidates: list) -> list:
    """Build Gemini tool definitions from top-k candidates within a domain."""
    domain_config = DOMAIN_REGISTRY[domain]
    tools = []
    for intent_name, score in candidates:
        api_config = INTENT_API_ROUTING.get(intent_name, {})
        tools.append({
            "type": "function",
            "function": {
                "name": intent_name,
                "description": INTENT_DESCRIPTIONS[intent_name],
                "parameters": {
                    "type": "object",
                    "properties": {
                        entity: {"type": "string", "description": f"Required: {entity}"}
                        for entity in api_config.get("required_entities", [])
                    },
                    "required": api_config.get("required_entities", [])
                }
            }
        })
    return tools
```

#### Example Gemini Interaction (Cap-API Domain)

```
System: "You are routing within the Cap-API domain 
         (Endpoint: /myclaims/claims/v1/claim/byclaimnumber)"
Tools:  [pricing_info, claim_status, settlement_info, reimbursement_info, cob_info]

User: "How much did the pharmacy get paid for this claim?"

Gemini Response:
{
  "tool_calls": [{
    "function": {
      "name": "reimbursement_info",
      "arguments": {}
    }
  }],
  "reasoning": "The user asks about what the pharmacy was paid, which is 
                reimbursement (amount paid TO the pharmacy), not pricing 
                (amount paid BY the patient). settlement_info is about 
                response codes, not payment amounts. All candidates are 
                correctly within Cap-API domain."
}
```

#### Example: Cross-Domain Ambiguity

When the domain router is uncertain between two domains, Tier 3 receives candidates from **both domains**:

```
System: "The domain router is uncertain between Cap-API and Claim History Search."
Tools:  [claim_status (Cap-API), rx_details (Cap-API), 
         claim_status (History), fill_date_info (History), drug_info (History)]

User: "What happened with my prescription last month?"

Gemini Response:
{
  "tool_calls": [{
    "function": {
      "name": "date_range_claims",
      "arguments": {"date_range": "last month"}
    }
  }],
  "domain": "claim_history_search",
  "reasoning": "'last month' implies a date-based search, which belongs to 
                Claim History Search, not a single-claim Cap-API lookup."
}
```

This is exactly the kind of nuance that embeddings miss but an LLM catches instantly.

---

## 5. End-to-End Workflow

### 5.1 Workflow Diagram

```
User Query: "What's the TF status for this claim?"
    │
    ▼
[1] EMBED QUERY ─────────────────────────── ⏱ ~20ms (API call to embedding service)
    │  Vector: [0.023, -0.118, 0.445, ...]
    │
    ▼
[2] CENTROID RETRIEVAL ──────────────────── ⏱ <0.1ms (matrix multiply)
    │  Top 5:
    │    1. approval_info:     0.87   ← TF = Transition Fill
    │    2. prior_auth_info:   0.79   ← "status" overlap
    │    3. claim_status:      0.76   ← "status" overlap
    │    4. beneficiary_info:  0.71   ← benefit phase overlap
    │    5. drug_info:         0.65
    │
    ▼
[3] CROSS-ENCODER RERANK ───────────────── ⏱ ~10ms (5 pairs)
    │  Reranked:
    │    1. approval_info:     0.93 ✅  "TF" matches "transition fill" in description
    │    2. prior_auth_info:   0.31     "TF" ≠ "prior authorization"
    │    3. claim_status:      0.24
    │    4. beneficiary_info:  0.18
    │    5. drug_info:         0.08
    │
    │  Score 0.93 > 0.85 threshold
    │  Gap: 0.93 - 0.31 = 0.62 > 0.15 min gap
    │
    ▼
[4] ROUTE TO approval_info ─────────────── Total: ~30ms (no LLM call!)
```

### 5.2 Ambiguous Query — Tier 3 Escalation

```
User Query: "Show me the payment details"
    │
    ▼
[1] EMBED QUERY ─────────────────────────── ⏱ ~20ms
    │
    ▼
[2] CENTROID RETRIEVAL ──────────────────── ⏱ <0.1ms
    │  Top 5:
    │    1. pricing_info:       0.82
    │    2. reimbursement_info: 0.81
    │    3. settlement_info:    0.78
    │    4. cob_info:           0.72
    │    5. claim_status:       0.68
    │
    ▼
[3] CROSS-ENCODER RERANK ───────────────── ⏱ ~10ms
    │  Reranked:
    │    1. pricing_info:       0.72   ⚠️ Below threshold
    │    2. reimbursement_info: 0.68
    │    3. settlement_info:    0.44
    │    ...
    │
    │  Score 0.72 < 0.85 threshold — ESCALATE
    │  Gap: 0.72 - 0.68 = 0.04 < 0.15 min gap — CONFIRM ESCALATION
    │
    ▼
[4] GEMINI 2.5 FLASH ───────────────────── ⏱ ~400ms
    │
    │  Gemini sees: "payment details" query + 5 tool schemas
    │  Gemini recognizes: "payment details" from patient perspective = pricing_info
    │                     "payment details" from pharmacy perspective = reimbursement_info
    │
    │  Gemini Response:
    │  {
    │    "clarification": "Could you clarify what payment details you need?\n
    │                      1. **Your cost** (copay, out-of-pocket) → pricing breakdown\n
    │                      2. **Pharmacy reimbursement** (what the pharmacy was paid)"
    │  }
    │
    ▼
[5] RETURN CLARIFICATION TO USER ────────── Total: ~430ms
```

---

## 6. Concrete Examples

### 6.1 Example: High-Confidence Fast Path (Tier 1+2 Only)

| Query | Tier 1 Top-1 | Tier 2 Score | Routed To | LLM Called? | Total Time |
|-------|-------------|-------------|-----------|-------------|------------|
| "What's the copay for this claim?" | `pricing_info` (0.89) | 0.96 | `pricing_info` | No | ~30ms |
| "Show me the DUR edits" | `drug_interaction_info` (0.91) | 0.97 | `drug_interaction_info` | No | ~30ms |
| "Who prescribed this medication?" | `prescriber_info` (0.93) | 0.98 | `prescriber_info` | No | ~30ms |
| "What's the DAW code?" | `daw_info` (0.88) | 0.95 | `daw_info` | No | ~30ms |
| "hi there" | `greeting` (0.85) | 0.99 | `greeting` | No | ~30ms |

### 6.2 Example: Ambiguous Queries Resolved at Tier 2

| Query | Tier 1 Top-2 | Tier 2 Winner | Score/Gap | Resolution |
|-------|-------------|---------------|-----------|------------|
| "Show claim summary" | `claim_status` (0.84), `pricing_info` (0.81) | `claim_status` (0.91) vs `pricing_info` (0.38) | Gap=0.53 | Tier 2 resolves via "claim summary" ≠ "pricing summary" |
| "Transition fill info" | `approval_info` (0.86), `beneficiary_info` (0.74) | `approval_info` (0.94) vs `beneficiary_info` (0.21) | Gap=0.73 | "TF" in approval description |
| "Generic alternatives" | `generic_availability` (0.88), `daw_info` (0.79) | `generic_availability` (0.93) vs `daw_info` (0.34) | Gap=0.59 | Cross-encoder distinguishes "alternative" from "substitution allowed" |

### 6.3 Example: Queries Requiring Tier 3 (Gemini)

| Query | Tier 2 Top-2 | Tier 2 Gap | Gemini Decision |
|-------|-------------|-----------|-----------------|
| "Show me the payment details" | `pricing_info` (0.72), `reimbursement_info` (0.68) | 0.04 | **Clarification**: "Patient cost or pharmacy reimbursement?" |
| "What happened with my prescription?" | `claim_status` (0.69), `rx_details` (0.65) | 0.04 | `claim_status` — user wants overall outcome |
| "Show me all the drug information and pricing" | `drug_info` (0.71), `pricing_info` (0.70) | 0.01 | **Multi-intent**: routes to `claim_status` (comprehensive summary) |

### 6.4 Example: Current System Failure → New System Success

**Query**: "What can be done to overcome the rejection?"

| System | Intent | Confidence | Correct? |
|--------|--------|------------|----------|
| **Current (embedded)** | `rejection_reasons` | 0.82 | Partially — it's about resolution, not just "why" |
| **Current (keyword)** | `rejection_reasons` | 0.75 | Same issue |
| **New (Cascading)** | Tier 2 → `rejection_reasons` (0.92) | 0.92 | ✅ Cross-encoder matches "overcome rejection" to "actionable steps to resolve" in description |

---

## 7. Data Preparation & Centroid Generation

### 7.1 Synthetic Prompt Generation with Gemini

For each intent, generate **50 diverse user utterances** using Gemini:

```python
GENERATION_PROMPT = """Generate 50 diverse user queries that a pharmacy call center 
agent would use to request {intent_description}.

Requirements:
1. Vary the phrasing: questions, commands, casual requests
2. Include domain-specific jargon: NDC, GPI, DAW, TF, PDE, DUR, COB, etc.
3. Include typos and abbreviations that real users might type
4. Include queries of varying length (3 words to 20 words)
5. Do NOT include queries that could belong to a different intent
6. Focus on DISCRIMINATIVE language — words that uniquely identify this intent

Output as a JSON array of strings.

Intent: {intent_name}
Description: {intent_description}

Related intents to AVOID overlapping with:
{related_intents_with_descriptions}
"""
```

### 7.2 Anti-Overlap Generation Strategy

The key innovation is telling Gemini **which intents NOT to overlap with**:

```python
# When generating prompts for "pricing_info", include this context:
related_intents = """
- settlement_info: Settlement codes sent TO the pharmacy (not cost to patient)
- reimbursement_info: Amount paid TO the pharmacy for paper claims
- cob_info: Coordination of benefits between two insurance plans
- beneficiary_info: Member's benefit phase and accumulation status
"""
```

This forces the synthetic prompts to use **discriminative language** that distinguishes each intent.

### 7.3 Centroid Generation Script

```python
#!/usr/bin/env python3
"""
scripts/generate_centroids.py
Generate centroid vectors for all intents from synthetic prompts.
"""

import json
import numpy as np
from pathlib import Path
from services.google_embeddings import get_embedding, get_google_embeddings

def generate_centroids():
    """Generate and save centroid vectors for all intents."""
    
    embeddings_service = get_google_embeddings()
    centroids = {}
    
    # Load synthetic prompts (generated by Gemini, reviewed by team)
    prompts_dir = Path("classifiers/centroids/prompts")
    
    for prompt_file in prompts_dir.glob("*.json"):
        intent_name = prompt_file.stem  # e.g., "pricing_info"
        
        with open(prompt_file) as f:
            prompts = json.load(f)  # List of 50 strings
        
        print(f"Generating centroid for {intent_name} ({len(prompts)} prompts)...")
        
        # Batch embed all 50 prompts
        embeddings = embeddings_service.embed(prompts)
        embedding_matrix = np.array(embeddings)  # Shape: (50, 768)
        
        # Compute centroid (mean vector)
        centroid = np.mean(embedding_matrix, axis=0)  # Shape: (768,)
        
        # L2 normalize for cosine similarity
        centroid = centroid / np.linalg.norm(centroid)
        
        centroids[intent_name] = centroid
        
        # Also compute std dev to detect outlier prompts
        distances = np.linalg.norm(embedding_matrix - centroid, axis=1)
        outliers = np.where(distances > np.mean(distances) + 2 * np.std(distances))[0]
        if len(outliers) > 0:
            print(f"  ⚠️  {len(outliers)} outlier prompts detected — review these:")
            for idx in outliers:
                print(f"     [{idx}] {prompts[idx]}")
    
    # Save centroids
    output_path = Path("classifiers/centroids/centroids.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **centroids)
    
    print(f"\n✅ Saved {len(centroids)} centroids to {output_path}")
    print(f"   Total size: {output_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    generate_centroids()
```

---

## 8. Implementation Plan

### Phase 1: Data Preparation (Week 1)

| # | Task | Output |
|---|------|--------|
| 1.1 | Write `INTENT_DESCRIPTIONS` dict (all 30 intents) | `classifiers/cascading/intent_descriptions.py` |
| 1.2 | Generate 50 synthetic prompts per intent using Gemini | `classifiers/centroids/prompts/*.json` (30 files) |
| 1.3 | Review & de-duplicate prompts, remove cross-intent overlaps | Cleaned prompt files |
| 1.4 | Generate centroid vectors | `classifiers/centroids/centroids.npz` |
| 1.5 | Write accuracy benchmark (use existing test queries) | `tests/test_cascading_router.py` |

### Phase 2: Tier 1 + Tier 2 Implementation (Week 2)

| # | Task | Output |
|---|------|--------|
| 2.1 | Implement `CentroidRouter` class (Tier 1) | `classifiers/cascading/centroid_router.py` |
| 2.2 | Implement `CrossEncoderReranker` class (Tier 2) | `classifiers/cascading/cross_encoder_reranker.py` |
| 2.3 | Implement `CascadingClassifier` (orchestrates Tier 1→2→3) | `classifiers/cascading/cascading_classifier.py` |
| 2.4 | Add `cross-encoder` to `requirements.txt` | Updated requirements |
| 2.5 | Run accuracy benchmark vs current embedded classifier | Benchmark results |

### Phase 3: Tier 3 + LangGraph Integration (Week 3)

| # | Task | Output |
|---|------|--------|
| 3.1 | Implement Gemini tool-calling for Tier 3 | `classifiers/cascading/llm_disambiguator.py` |
| 3.2 | Create new LangGraph node: `cascading_router_node` | `nodes/cascading_router.py` |
| 3.3 | Wire into `langgraph_agent.py` (behind feature flag) | Updated graph |
| 3.4 | Add config toggle: `use_cascading_router` | `config/config.py` |
| 3.5 | A/B test: cascading vs embedded classifier | Test results |

### Phase 4: Production Rollout (Week 4)

| # | Task | Output |
|---|------|--------|
| 4.1 | Enable for 10% of traffic (canary) | Feature flag |
| 4.2 | Monitor accuracy, latency, LLM cost | Dashboard |
| 4.3 | Tune thresholds based on production data | Updated thresholds |
| 4.4 | Full rollout | 100% traffic |
| 4.5 | Deprecate `embedded_classifier.py` | Cleanup |

---

## 9. Directory Structure

```
classifiers/
├── cascading/                          # NEW: Cascading Router
│   ├── __init__.py
│   ├── cascading_classifier.py         # Main orchestrator (Tier 1 → 2 → 3)
│   ├── centroid_router.py              # Tier 1: Fast vector retrieval
│   ├── cross_encoder_reranker.py       # Tier 2: Precision reranking
│   ├── llm_disambiguator.py            # Tier 3: Gemini tool-calling fallback
│   └── intent_descriptions.py          # Rich natural language intent descriptions
│
├── centroids/                          # NEW: Centroid data
│   ├── prompts/                        # 50 synthetic prompts per intent
│   │   ├── pricing_info.json
│   │   ├── claim_status.json
│   │   ├── rejection_reasons.json
│   │   └── ... (30 files)
│   ├── centroids.npz                   # Pre-computed centroid vectors
│   └── metadata.json                   # Provider, dimension, generation date
│
├── embedded_classifier.py              # EXISTING (to be deprecated)
├── keyword_classifier.py               # EXISTING (already deprecated)
├── intent_classifier_wrapper.py        # EXISTING (will add cascading option)
└── intent_classifier.py                # EXISTING (EDGAR legacy)

nodes/
├── cascading_router.py                 # NEW: LangGraph node for cascading router

scripts/
├── generate_centroids.py               # NEW: Centroid generation script
├── generate_synthetic_prompts.py       # NEW: Gemini prompt generation
└── benchmark_classifiers.py            # NEW: A/B accuracy benchmark
```

---

## 10. LangGraph Integration

### 10.1 New Node: `cascading_router_node`

The cascading router replaces the current `intent_agent_node` in the graph:

```python
# nodes/cascading_router.py

from classifiers.cascading.cascading_classifier import CascadingClassifier
from state.agent_state import AgentState

# Singleton
_classifier = None

def get_classifier() -> CascadingClassifier:
    global _classifier
    if _classifier is None:
        _classifier = CascadingClassifier()
    return _classifier

async def cascading_router_node(state: AgentState) -> dict:
    """
    LangGraph node: Classify intent using cascading router.
    Drop-in replacement for intent_agent_node.
    """
    classifier = get_classifier()
    query = state.get("normalized_text") or state.get("user_input", "")
    
    result = await classifier.classify_async(query)
    
    return {
        "intent": result["intent"],
        "confidence": result["confidence"],
        "all_scores": result["all_scores"],
        "is_complex": result["is_complex"],
        "needs_clarification": result["needs_clarification"],
        "classification_tier": result["tier"],          # NEW: which tier resolved it
        "tier3_reasoning": result.get("reasoning", ""), # NEW: Gemini's reasoning (if used)
    }
```

### 10.2 Updated Graph Wiring

```python
# langgraph_agent.py (modified section)

from config.config import settings

if settings.use_cascading_router:
    from nodes.cascading_router import cascading_router_node
    graph.add_node("intent_agent", cascading_router_node)
else:
    from agents.extended_intent_agent_node import intent_agent_node
    graph.add_node("intent_agent", intent_agent_node)

# Rest of the graph remains IDENTICAL — same edges, same routers
# The cascading router returns the same state keys as the current intent_agent
```

### 10.3 LangGraph Flow Comparison

```
CURRENT FLOW:
  orchestrator → safety → cache → intent_agent → confidence_checker
       → [low confidence] → llm_judge → confidence_checker (loop)
       → [high confidence] → build_context → call_api → response

NEW FLOW (with Cascading Router):
  orchestrator → safety → cache → cascading_router → confidence_checker
       → [Tier 1+2 resolved] → build_context → call_api → response
       → [Tier 3 clarification] → clarification → response (ask user)
       
  Key difference: The llm_judge node is NO LONGER NEEDED for intent routing.
  Tier 3 handles disambiguation internally. 
  The llm_judge path can be removed or repurposed for other validation.
```

---

## 11. Adding a New Domain (Onboarding Playbook)

### Example: Adding a New `"formulary_lookup"` Intent

**Total time: ~15 minutes. Zero code changes.**

#### Step 1: Write the Description (2 min)

Add to `classifiers/cascading/intent_descriptions.py`:

```python
INTENT_DESCRIPTIONS["formulary_lookup"] = (
    "Formulary tier lookup for a specific drug, including whether "
    "the medication is on the plan's formulary, its coverage tier, "
    "step therapy requirements, and quantity limits."
)
```

#### Step 2: Generate Synthetic Prompts (5 min)

Run the prompt generator:

```bash
python scripts/generate_synthetic_prompts.py \
    --intent formulary_lookup \
    --description "Formulary tier lookup including coverage tier, step therapy, quantity limits" \
    --avoid "drug_info,generic_availability,prior_auth_info" \
    --count 50
```

This creates `classifiers/centroids/prompts/formulary_lookup.json`:

```json
[
    "Is this drug on the formulary?",
    "Check the formulary tier for Lipitor",
    "What tier is my medication on?",
    "Does the plan cover this drug?",
    "Show formulary coverage for NDC 12345678901",
    "Is there a step therapy requirement for this med?",
    "What are the quantity limits for this prescription?",
    "Check if atorvastatin is a preferred drug",
    "Formulary status for GPI 39400010",
    ...
]
```

#### Step 3: Regenerate Centroids (3 min)

```bash
python scripts/generate_centroids.py
# Output: ✅ Saved 31 centroids to classifiers/centroids/centroids.npz
```

#### Step 4: Add API Routing Config (3 min)

Add to `config/api_routing_config.py`:

```python
"formulary_lookup": {
    "api_endpoint": ENDPOINTS["formulary_search"],
    "method": "POST",
    "required_entities": ["drug_name_or_ndc"],
    "optional_entities": ["plan_id"],
    "description": "Formulary tier and coverage lookup",
}
```

#### Step 5: Verify (2 min)

```bash
python scripts/benchmark_classifiers.py --intent formulary_lookup
# Tests: 50/50 prompts correctly classified ✅
# No regression on other intents ✅
```

**That's it.** No classifier retraining, no weight tuning, no code changes to the LangGraph graph.

---

## 12. Performance & Cost Analysis

### 12.1 Latency Breakdown

| Component | Current System | Cascading Router |
|-----------|---------------|------------------|
| Query embedding | ~20ms | ~20ms (same) |
| Intent matching | ~2ms (600 cosine ops) | <0.1ms (30 centroid ops) |
| Cross-encoder rerank | N/A | ~10ms (5 pairs) |
| **Subtotal (Tier 1+2)** | **~22ms** | **~30ms** |
| LLM judge (when triggered) | ~800ms (full re-classification) | ~400ms (5-tool context only) |
| **LLM trigger rate** | **~30-40% of queries** | **~10-20% of queries** |
| **Average end-to-end** | **~250ms** | **~70ms** |

### 12.2 Cost Comparison (per 10,000 queries)

| Cost Component | Current System | Cascading Router |
|----------------|---------------|------------------|
| Embedding API calls | 10,000 × $0.00002 = $0.20 | 10,000 × $0.00002 = $0.20 |
| LLM judge calls | 3,500 × $0.002 = $7.00 | 1,500 × $0.001 = $1.50 |
| Cross-encoder (local) | $0.00 | $0.00 (runs on CPU) |
| **Total** | **$7.20** | **$1.70** |
| **Savings** | — | **76% cost reduction** |

> **Note**: LLM cost per call is lower in the cascading system because Tier 3 only sends 5 tool schemas (~200 tokens) instead of the full re-classification prompt (~800 tokens).

### 12.3 Accuracy Projections

| Metric | Current Embedded | Cascading Router (Projected) |
|--------|-----------------|------------------------------|
| Top-1 accuracy (clear queries) | ~92% | ~97% |
| Top-1 accuracy (ambiguous queries) | ~68% | ~89% |
| False positive rate (`out_of_scope` misses) | ~8% | ~3% |
| Clarification rate (instead of wrong answer) | ~5% | ~12% (intentional — better to ask than guess) |

---

## 13. Migration Strategy

### 13.1 Feature Flag Approach

```python
# config/config.py
class Settings:
    # Classifier selection
    use_cascading_router: bool = False       # NEW: Enable cascading router
    use_embedding_classifier: bool = True    # EXISTING: Embedding classifier
    
    # Cascading router sub-options
    cascading_tier3_enabled: bool = True     # Allow Gemini fallback
    cascading_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cascading_tier2_threshold: float = 0.85  # Confidence for immediate routing
    cascading_tier2_min_gap: float = 0.15    # Min gap between #1 and #2
```

### 13.2 Migration Path

```
Phase 1: Shadow Mode
  ├── Both classifiers run in parallel
  ├── Current system routes traffic
  ├── Cascading router results logged for comparison
  └── No user-facing impact

Phase 2: Canary (10%)
  ├── 10% of traffic uses cascading router
  ├── Monitor accuracy, latency, cost
  └── Automated regression alerts

Phase 3: Gradual Rollout (10% → 50% → 100%)
  ├── Increase traffic percentage weekly
  ├── Tune thresholds based on production data
  └── Document any edge cases

Phase 4: Deprecation
  ├── Remove embedded_classifier.py
  ├── Remove keyword_classifier.py
  ├── Remove llm_judge node (absorbed by Tier 3)
  └── Clean up config flags
```

### 13.3 Rollback Plan

If the cascading router underperforms in production:

1. Set `use_cascading_router = False` in config
2. The existing `embedded_classifier.py` immediately takes over
3. No code deployment needed — config change only

---

## Appendix

### A. Cross-Encoder Model Comparison

| Model | Size | Accuracy (NDCG@10) | Speed (ms/pair) | Recommended For |
|-------|------|---------------------|-----------------|-----------------|
| `BAAI/bge-reranker-v2-m3` | 568M | 0.740 | ~15ms | Highest accuracy |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | 0.691 | ~5ms | **Best balance** ✅ |
| `BAAI/bge-reranker-base` | 278M | 0.712 | ~8ms | Good middle ground |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | 4.4M | 0.631 | ~2ms | Fastest |

**Recommendation**: Start with `ms-marco-MiniLM-L-6-v2` (22MB, 5ms/pair). Upgrade to `bge-reranker-v2-m3` if accuracy demands it.

### B. Embedding Dimensions by Provider

| Provider | Model | Dimension | Centroid File Size (30 intents) |
|----------|-------|-----------|--------------------------------|
| Google Vertex AI | text-embedding-005 | 768 | ~90 KB |
| Azure OpenAI | text-embedding-ada-002 | 1536 | ~180 KB |
| Azure OpenAI | text-embedding-3-small | 1536 | ~180 KB |

### C. Confidence Threshold Tuning Guide

| Tier 2 Threshold | Effect |
|-----------------|--------|
| 0.95 | Very conservative — most queries hit Tier 3 (expensive but very accurate) |
| **0.85** | **Recommended** — balances accuracy and cost |
| 0.75 | Aggressive — fewer Tier 3 calls, slightly more misroutes |
| 0.65 | Too aggressive — similar to current flat classifier behavior |

| Min Gap Threshold | Effect |
|-------------------|--------|
| 0.25 | Conservative — escalates when top-2 are within 0.25 |
| **0.15** | **Recommended** — catches genuine ambiguity |
| 0.05 | Aggressive — only escalates near-ties |

### D. Glossary

| Term | Definition |
|------|------------|
| **Centroid** | The mean (average) vector of all embeddings for an intent's synthetic prompts |
| **Cross-Encoder** | A transformer model that processes two texts jointly (versus independently) for more accurate similarity |
| **Bi-Encoder** | The current approach — embeds query and examples independently, then compares vectors |
| **Tool-Calling** | LLM feature where the model selects from predefined function signatures |
| **Shortlist** | The Top-K candidates from Tier 1, passed to Tier 2 for reranking |
| **Reranking** | Re-scoring a shortlist using a more accurate (but slower) model |
| **Escalation** | Passing a query to the next tier when the current tier cannot decide |
| **DAW** | Dispense As Written — brand vs generic indicator |
| **TF** | Transition Fill — temporary coverage during plan changes |
| **DUR** | Drug Utilization Review — clinical safety screening |
| **COB** | Coordination of Benefits — dual insurance adjudication |
| **PDE** | Prescription Drug Event — Medicare Part D reporting record |
| **MIC** | Most Ingredient Cost — compound drug pricing |
| **BPG** | Benefit Plan Group — plan configuration identifier |

---

### E. References

- [BGE Reranker v2](https://huggingface.co/BAAI/bge-reranker-v2-m3) — Cross-encoder model
- [ms-marco-MiniLM](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) — Lightweight cross-encoder
- [Gemini 2.5 Flash](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini) — LLM for Tier 3
- [Matryoshka Embeddings](https://arxiv.org/abs/2205.13147) — Efficient embedding compression
- [ColBERT v2](https://arxiv.org/abs/2112.01488) — Late-interaction reranking (future consideration)
