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
    - [F. Domain Imbalance & Cluster Overlap Problem (v2 Fix)](#f-domain-imbalance--cluster-overlap-problem-v2-fix)

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

## 6. End-to-End Workflow

### 6.1 Workflow Diagram — Cap-API Fast Path

```
User Query: "What's the TF status for this claim?"
    │
    ▼
[0] GENERAL CHECK ───────────────────────── ⏱ <0.01ms (pattern match)
    │  Not greeting/help/out_of_scope → continue
    │
    ▼
[1] EMBED QUERY ─────────────────────────── ⏱ ~20ms (API call to embedding service)
    │  Vector: [0.023, -0.118, 0.445, ...]
    │
    ▼
[2] DOMAIN ROUTING ──────────────────────── ⏱ <0.05ms (4 cosine ops)
    │  Domain scores:
    │    benefits_api:          0.84 ← winner ("TF" = approval/transition fill)
    │    cap_api:               0.71
    │    claim_history_search:  0.58
    │    general:               0.12
    │
    │  → Selected domain: Benefits API (3 intents)
    │
    ▼
[3] INTENT SHORTLISTING ────────────────── ⏱ <0.05ms (3 cosine ops)
    │  Within Benefits API:
    │    1. approval_info:    0.87   ← TF = Transition Fill
    │    2. beneficiary_info: 0.71   ← benefit phase overlap
    │    3. audit_info:       0.52
    │
    ▼
[4] CROSS-ENCODER RERANK ───────────────── ⏱ ~8ms (3 pairs — small domain!)
    │  Reranked:
    │    1. approval_info:    0.93 ✅  "TF" matches "transition fill" in description
    │    2. beneficiary_info: 0.18
    │    3. audit_info:       0.08
    │
    │  Score 0.93 > 0.85 threshold
    │  Gap: 0.93 - 0.18 = 0.75 > 0.15 min gap
    │
    ▼
[5] ROUTE TO Benefits API / approval_info   Total: ~28ms (no LLM call!)
    │  Endpoint: /myclaims/benefits/v1/member
```

### 6.2 Workflow Diagram — Claim History Search Path

```
User Query: "Show me claims from January to March for this drug"
    │
    ▼
[0] GENERAL CHECK → not general
    │
    ▼
[1] EMBED QUERY ─────────────────────────── ⏱ ~20ms
    │
    ▼
[2] DOMAIN ROUTING ──────────────────────── ⏱ <0.05ms
    │  claim_history_search:  0.89 ← winner (date range + drug = search)
    │  cap_api:               0.68
    │  benefits_api:          0.42
    │  general:               0.08
    │
    │  → Selected domain: Claim History Search (6 intents)
    │
    ▼
[3] INTENT SHORTLISTING ────────────────── ⏱ <0.05ms (6 cosine ops)
    │  Within Claim History Search:
    │    1. date_range_claims:    0.91  ← "January to March"
    │    2. drug_info:            0.74  ← "for this drug"
    │    3. fill_date_info:       0.68
    │    4. claim_status:         0.55
    │    5. drug_interaction_info: 0.41
    │
    ▼
[4] CROSS-ENCODER RERANK ───────────────── ⏱ ~10ms (5 pairs)
    │    1. date_range_claims:  0.96 ✅  "January to March" = date range
    │    2. drug_info:          0.32
    │    ...
    │
    ▼
[5] ROUTE TO Claim History / date_range_claims  Total: ~30ms
    │  Endpoint: /myclaims/claims/v1/claim/history
```

### 6.3 Ambiguous Query — Cap-API Tier 3 Escalation

```
User Query: "Show me the payment details"
    │
    ▼
[0] GENERAL CHECK → not general
    │
    ▼
[1] EMBED QUERY ─────────────────────────── ⏱ ~20ms
    │
    ▼
[2] DOMAIN ROUTING ──────────────────────── ⏱ <0.05ms
    │  cap_api:               0.85 ← winner (payment = pricing/reimbursement)
    │  claim_history_search:  0.52
    │  benefits_api:          0.38
    │  general:               0.06
    │
    │  → Selected domain: Cap-API (12 intents)
    │
    ▼
[3] INTENT SHORTLISTING ────────────────── ⏱ <0.05ms
    │  Within Cap-API:
    │    1. pricing_info:       0.82
    │    2. reimbursement_info: 0.81
    │    3. settlement_info:    0.78
    │    4. cob_info:           0.72
    │    5. claim_status:       0.68
    │
    ▼
[4] CROSS-ENCODER RERANK ───────────────── ⏱ ~10ms (5 pairs)
    │    1. pricing_info:       0.72   ⚠️ Below threshold
    │    2. reimbursement_info: 0.68
    │    3. settlement_info:    0.44
    │    ...
    │
    │  Score 0.72 < 0.85 threshold — ESCALATE
    │  Gap: 0.72 - 0.68 = 0.04 < 0.15 min gap — CONFIRM ESCALATION
    │
    ▼
[5] GEMINI 2.5 FLASH ───────────────────── ⏱ ~400ms
    │  Domain context: "Cap-API — /myclaims/claims/v1/claim/byclaimnumber"
    │  Tools: only 5 Cap-API intents (not all 24!)
    │
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
[6] RETURN CLARIFICATION TO USER ────────── Total: ~430ms
```

### 6.4 Cross-Domain Routing for Shared Intent

```
User Query: "Show me the claim status"
    │
    ▼
[2] DOMAIN ROUTING ──────────────────────── 
    │  cap_api:               0.81  ← "claim status" as single-claim lookup
    │  claim_history_search:  0.76  ← "claim status" as search
    │  benefits_api:          0.45
    │
    │  Gap: 0.81 - 0.76 = 0.05 → domain is AMBIGUOUS
    │
    ▼
[3] MULTI-DOMAIN SHORTLIST ─────────────── Pull intents from top-2 domains
    │  From Cap-API:            claim_status (0.84), rx_details (0.61)
    │  From Claim History:      claim_status (0.79), date_range_claims (0.58)
    │
    ▼
[4] CROSS-ENCODER RERANK ───────────────── 
    │  Cap-API / claim_status:     0.88 ✅  (single claim lookup)
    │  History / claim_status:     0.71     (search context)
    │  ...
    │
    │  Score 0.88 > 0.85 → ROUTE TO Cap-API / claim_status
    │  Endpoint: /myclaims/claims/v1/claim/byclaimnumber
```

---

## 7. Concrete Examples

### 7.1 Example: High-Confidence Fast Path by Domain (Tier 0/1/2 Only)

**Cap-API Domain:**

| Query | Domain Score | Intent | Tier 2 Score | LLM? | Time |
|-------|-------------|--------|-------------|------|------|
| "What's the copay for this claim?" | cap_api: 0.88 | `pricing_info` | 0.96 | No | ~30ms |
| "Who prescribed this medication?" | cap_api: 0.91 | `prescriber_info` | 0.98 | No | ~30ms |
| "Why was this claim rejected?" | cap_api: 0.87 | `rejection_reasons` | 0.95 | No | ~30ms |
| "Show me the COB details" | cap_api: 0.85 | `cob_info` | 0.93 | No | ~30ms |
| "Generic alternatives available?" | cap_api: 0.84 | `generic_availability` | 0.91 | No | ~30ms |

**Claim History Search Domain:**

| Query | Domain Score | Intent | Tier 2 Score | LLM? | Time |
|-------|-------------|--------|-------------|------|------|
| "Show me the DUR edits" | history: 0.90 | `drug_interaction_info` | 0.97 | No | ~28ms |
| "When was this prescription filled?" | history: 0.87 | `fill_date_info` | 0.94 | No | ~28ms |
| "Display claims from the past 6 months" | history: 0.92 | `date_range_claims` | 0.96 | No | ~28ms |
| "Show compound ingredient breakdown" | history: 0.86 | `compound_info` | 0.93 | No | ~28ms |
| "Drug name and NDC for this claim" | history: 0.88 | `drug_info` | 0.95 | No | ~28ms |

**Benefits API Domain:**

| Query | Domain Score | Intent | Tier 2 Score | LLM? | Time |
|-------|-------------|--------|-------------|------|------|
| "What benefit phase is this member in?" | benefits: 0.93 | `beneficiary_info` | 0.97 | No | ~25ms |
| "Show the audit trail for this claim" | benefits: 0.89 | `audit_info` | 0.96 | No | ~25ms |
| "Transition fill status for this claim" | benefits: 0.87 | `approval_info` | 0.94 | No | ~25ms |

**General Domain (Tier 0 — instant):**

| Query | Resolution | LLM? | Time |
|-------|-----------|------|------|
| "hi there" | `greeting` (pattern match) | No | <0.1ms |
| "help me submit a claim" | `help` (pattern match) | No | <0.1ms |
| "What's the weather?" | `out_of_scope` (Tier 1 centroid) | No | ~20ms |

### 7.2 Example: Ambiguous Queries Resolved at Tier 2 (Within Domain)

| Query | Domain | Tier 1 Top-2 (in domain) | Tier 2 Winner | Gap | Resolution |
|-------|--------|-------------------------|---------------|-----|------------|
| "Show claim summary" | Cap-API | `claim_status` (0.84), `pricing_info` (0.81) | `claim_status` (0.91) | 0.53 | "claim summary" ≠ "pricing summary" |
| "Transition fill info" | Benefits | `approval_info` (0.86), `beneficiary_info` (0.74) | `approval_info` (0.94) | 0.73 | "TF" in approval description |
| "Generic alternatives" | Cap-API | `generic_availability` (0.88), `rx_details` (0.67) | `generic_availability` (0.93) | 0.59 | "alternative" = generic, not rx |
| "Show DUR override" | History | `drug_interaction_info` (0.85), `drug_info` (0.72) | `drug_interaction_info` (0.92) | 0.55 | "DUR override" = interaction review |

### 7.3 Example: Queries Requiring Tier 3 (Gemini) — Intra-Domain Ambiguity

| Query | Domain | Tier 2 Top-2 | Gap | Gemini Decision |
|-------|--------|-------------|-----|-----------------|
| "Show me the payment details" | Cap-API | `pricing_info` (0.72), `reimbursement_info` (0.68) | 0.04 | **Clarification**: "Patient cost or pharmacy reimbursement?" |
| "What happened with my prescription?" | Cap-API | `claim_status` (0.69), `rx_details` (0.65) | 0.04 | `claim_status` — user wants overall outcome |
| "Member changes on this claim" | Benefits | `audit_info` (0.74), `beneficiary_info` (0.71) | 0.03 | `audit_info` — "changes" = modification history |

### 7.4 Example: Cross-Domain Ambiguity Resolved at Tier 3

| Query | Domain Scores | Top Domains | Gemini Decision |
|-------|--------------|------------|-----------------|
| "What happened with this claim last month?" | cap: 0.78, history: 0.76 | Both | `date_range_claims` (History) — "last month" = date search |
| "Claim status for drug interactions" | history: 0.77, cap: 0.74 | Both | `drug_interaction_info` (History) — drug interaction focus |
| "Show all claim details and benefits" | cap: 0.73, benefits: 0.71 | Both | **Clarification**: "Claim details (Cap-API) or member benefits (Benefits API)?" |

### 7.5 Example: Current System Failure → New System Success

**Query**: "Show the audit trail for this claim's approval"

| System | Intent | Domain | Correct? |
|--------|--------|--------|----------|
| **Current (flat embedded)** | `audit_info` (0.83) vs `approval_info` (0.81) | No domain concept | ⚠️ Randomly picks one — both in Benefits but for different purposes |
| **New (Cascading)** | Domain: Benefits (0.91) → Tier 2: `approval_info` (0.89) vs `audit_info` (0.72) | Benefits API | ✅ Cross-encoder sees "approval" modifies the query, not "audit" |

---

## 8. Data Preparation & Centroid Generation

### 8.1 Synthetic Prompt Generation with Gemini

For each intent, generate **50 diverse user utterances** using Gemini, with **domain context** to avoid cross-domain overlap:

```python
GENERATION_PROMPT = """Generate 50 diverse user queries that a pharmacy call center 
agent would use to request {intent_description}.

DOMAIN CONTEXT:
This intent belongs to the "{domain_name}" domain.
Domain API: {api_endpoint}
Other intents in this domain: {sibling_intents}

Requirements:
1. Vary the phrasing: questions, commands, casual requests
2. Include domain-specific jargon: NDC, GPI, DAW, TF, PDE, DUR, COB, etc.
3. Include typos and abbreviations that real users might type
4. Include queries of varying length (3 words to 20 words)
5. Do NOT include queries that could belong to a different intent
6. Do NOT include queries that could belong to a different DOMAIN
7. Focus on DISCRIMINATIVE language — words that uniquely identify this intent

Output as a JSON array of strings.

Intent: {intent_name}
Description: {intent_description}

INTENTS IN SAME DOMAIN to AVOID overlapping with:
{sibling_intents_with_descriptions}

INTENTS IN OTHER DOMAINS to AVOID overlapping with:
{other_domain_intents_with_descriptions}
"""
```

### 8.2 Anti-Overlap Generation Strategy (Domain-Aware)

The key innovation is telling Gemini both **intra-domain** and **cross-domain** overlaps to avoid:

```python
# When generating prompts for "pricing_info" (Cap-API), include this context:

same_domain_intents = """
SAME DOMAIN (Cap-API) — avoid overlap with:
- settlement_info: Settlement codes sent TO the pharmacy (not cost to patient)
- reimbursement_info: Amount paid TO the pharmacy for paper claims
- cob_info: Coordination of benefits between two insurance plans
- claim_status: General claim status, NOT pricing details
- rx_details: Prescription number and quantity, NOT cost
"""

other_domain_intents = """
OTHER DOMAINS — avoid overlap with:
- beneficiary_info (Benefits API): Member's benefit phase and accumulation status
- approval_info (Benefits API): Approval/TF details, NOT pricing
- fill_date_info (Claim History): When the prescription was filled, NOT how much it cost
"""
```

This forces the synthetic prompts to use **discriminative language** that distinguishes each intent both within and across domains.

### 8.3 Centroid Generation Script (Domain-Aware)

```python
#!/usr/bin/env python3
"""
scripts/generate_centroids.py
Generate two-level centroid vectors: domain centroids + intent centroids.
"""

import json
import numpy as np
from pathlib import Path
from services.google_embeddings import get_google_embeddings

DOMAIN_REGISTRY = {
    "cap_api": [
        "claim_status", "multi_claim_summary", "pharmacy_info", "prescriber_info",
        "pricing_info", "reimbursement_info", "rejection_reasons", "settlement_info",
        "rx_details", "reversal_info", "cob_info", "generic_availability",
    ],
    "claim_history_search": [
        "claim_status", "date_range_claims", "drug_info", "fill_date_info",
        "drug_interaction_info", "compound_info",
    ],
    "benefits_api": [
        "beneficiary_info", "audit_info", "approval_info",
    ],
    "general": [
        "greeting", "help", "out_of_scope",
    ],
}

def generate_centroids():
    """Generate and save two-level centroid vectors."""
    
    embeddings_service = get_google_embeddings()
    intent_centroids = {}
    domain_centroids = {}
    
    prompts_dir = Path("classifiers/centroids/prompts")
    
    # ---- Step 1: Generate intent centroids ----
    for prompt_file in prompts_dir.glob("*.json"):
        intent_name = prompt_file.stem
        
        with open(prompt_file) as f:
            prompts = json.load(f)
        
        print(f"Generating centroid for {intent_name} ({len(prompts)} prompts)...")
        
        embeddings = embeddings_service.embed(prompts)
        embedding_matrix = np.array(embeddings)
        
        centroid = np.mean(embedding_matrix, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        intent_centroids[intent_name] = centroid
        
        # Detect outlier prompts
        distances = np.linalg.norm(embedding_matrix - centroid, axis=1)
        outliers = np.where(distances > np.mean(distances) + 2 * np.std(distances))[0]
        if len(outliers) > 0:
            print(f"  ⚠️  {len(outliers)} outlier prompts detected — review these:")
            for idx in outliers:
                print(f"     [{idx}] {prompts[idx]}")
    
    # ---- Step 2: Generate domain centroids (average of intent centroids) ----
    for domain, intents in DOMAIN_REGISTRY.items():
        domain_intent_vectors = [
            intent_centroids[i] for i in intents if i in intent_centroids
        ]
        if domain_intent_vectors:
            domain_centroid = np.mean(domain_intent_vectors, axis=0)
            domain_centroid = domain_centroid / np.linalg.norm(domain_centroid)
            domain_centroids[domain] = domain_centroid
            print(f"Domain centroid: {domain} (from {len(domain_intent_vectors)} intents)")
    
    # ---- Step 3: Save ----
    output_dir = Path("classifiers/centroids")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(output_dir / "intent_centroids.npz", **intent_centroids)
    np.savez_compressed(output_dir / "domain_centroids.npz", **domain_centroids)
    
    # Save metadata
    metadata = {
        "provider": "Google Cloud Vertex AI",
        "model": "text-embedding-005",
        "dimension": len(next(iter(intent_centroids.values()))),
        "num_intent_centroids": len(intent_centroids),
        "num_domain_centroids": len(domain_centroids),
        "domains": {d: intents for d, intents in DOMAIN_REGISTRY.items()},
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Saved {len(intent_centroids)} intent centroids + {len(domain_centroids)} domain centroids")

if __name__ == "__main__":
    generate_centroids()
```

---

## 9. Implementation Plan

### Phase 1: Data Preparation (Week 1)

| # | Task | Output |
|---|------|--------|
| 1.1 | Define `DOMAIN_REGISTRY` with 4 domains and 24 intents | `classifiers/cascading/domain_registry.py` |
| 1.2 | Write `INTENT_DESCRIPTIONS` dict (24 intents, domain-organized) | `classifiers/cascading/intent_descriptions.py` |
| 1.3 | Generate 50 synthetic prompts per intent using Gemini (domain-aware) | `classifiers/centroids/prompts/*.json` (24 files) |
| 1.4 | Review & de-duplicate prompts, remove cross-domain overlaps | Cleaned prompt files |
| 1.5 | Generate two-level centroids (4 domain + 24 intent) | `classifiers/centroids/domain_centroids.npz` + `intent_centroids.npz` |
| 1.6 | Write accuracy benchmark (use existing test queries) | `tests/test_cascading_router.py` |

### Phase 2: Tier 0 + Tier 1 + Tier 2 Implementation (Week 2)

| # | Task | Output |
|---|------|--------|
| 2.1 | Implement `GeneralFastPath` (Tier 0 — pattern match) | `classifiers/cascading/general_fast_path.py` |
| 2.2 | Implement `DomainAwareCentroidRouter` (Tier 1 — two-level) | `classifiers/cascading/centroid_router.py` |
| 2.3 | Implement `CrossEncoderReranker` (Tier 2 — domain-scoped) | `classifiers/cascading/cross_encoder_reranker.py` |
| 2.4 | Implement `CascadingClassifier` (orchestrates Tier 0→1→2→3) | `classifiers/cascading/cascading_classifier.py` |
| 2.5 | Add `cross-encoder` to `requirements.txt` | Updated requirements |
| 2.6 | Run accuracy benchmark vs current embedded classifier | Benchmark results |

### Phase 3: Tier 3 + LangGraph Integration (Week 3)

| # | Task | Output |
|---|------|--------|
| 3.1 | Implement Gemini tool-calling for Tier 3 (domain-scoped) | `classifiers/cascading/llm_disambiguator.py` |
| 3.2 | Create new LangGraph node: `cascading_router_node` | `nodes/cascading_router.py` |
| 3.3 | Wire into `langgraph_agent.py` (behind feature flag) | Updated graph |
| 3.4 | Add config toggle: `use_cascading_router` | `config/config.py` |
| 3.5 | A/B test: cascading vs embedded classifier (per domain) | Test results |

### Phase 4: Production Rollout (Week 4)

| # | Task | Output |
|---|------|--------|
| 4.1 | Enable for 10% of traffic (canary) | Feature flag |
| 4.2 | Monitor accuracy per domain, latency, LLM cost | Dashboard |
| 4.3 | Tune thresholds based on production data | Updated thresholds |
| 4.4 | Full rollout | 100% traffic |
| 4.5 | Deprecate `embedded_classifier.py` | Cleanup |

---

## 10. Directory Structure

```
classifiers/
├── cascading/                          # NEW: Domain-Aware Cascading Router
│   ├── __init__.py
│   ├── cascading_classifier.py         # Main orchestrator (Tier 0 → 1 → 2 → 3)
│   ├── domain_registry.py             # Domain definitions + intent-to-domain mapping
│   ├── general_fast_path.py           # Tier 0: Pattern match for greeting/help/oos
│   ├── centroid_router.py              # Tier 1: Two-level domain + intent retrieval
│   ├── cross_encoder_reranker.py       # Tier 2: Precision reranking (domain-scoped)
│   ├── llm_disambiguator.py            # Tier 3: Gemini tool-calling (domain-scoped)
│   └── intent_descriptions.py          # Rich descriptions organized by domain
│
├── centroids/                          # NEW: Two-level centroid data
│   ├── prompts/                        # 50 synthetic prompts per intent
│   │   ├── # --- Cap-API (12 intents) ---
│   │   ├── claim_status.json
│   │   ├── multi_claim_summary.json
│   │   ├── pharmacy_info.json
│   │   ├── prescriber_info.json
│   │   ├── pricing_info.json
│   │   ├── reimbursement_info.json
│   │   ├── rejection_reasons.json
│   │   ├── settlement_info.json
│   │   ├── rx_details.json
│   │   ├── reversal_info.json
│   │   ├── cob_info.json
│   │   ├── generic_availability.json
│   │   ├── # --- Claim History Search (6 intents) ---
│   │   ├── date_range_claims.json
│   │   ├── drug_info.json
│   │   ├── fill_date_info.json
│   │   ├── drug_interaction_info.json
│   │   ├── compound_info.json
│   │   ├── # --- Benefits API (3 intents) ---
│   │   ├── beneficiary_info.json
│   │   ├── audit_info.json
│   │   ├── approval_info.json
│   │   ├── # --- General (3 intents) ---
│   │   ├── greeting.json
│   │   ├── help.json
│   │   └── out_of_scope.json
│   ├── domain_centroids.npz            # 4 domain centroid vectors
│   ├── intent_centroids.npz            # 24 intent centroid vectors
│   └── metadata.json                   # Provider, dimension, domain→intent mapping
│
├── embedded_classifier.py              # EXISTING (to be deprecated)
├── keyword_classifier.py               # EXISTING (already deprecated)
├── intent_classifier_wrapper.py        # EXISTING (will add cascading option)
└── intent_classifier.py                # EXISTING (EDGAR legacy)

nodes/
├── cascading_router.py                 # NEW: LangGraph node for cascading router

scripts/
├── generate_centroids.py               # NEW: Two-level centroid generation
├── generate_synthetic_prompts.py       # NEW: Domain-aware Gemini prompt generation
└── benchmark_classifiers.py            # NEW: Per-domain A/B accuracy benchmark
```

---

## 11. LangGraph Integration

### 11.1 New Node: `cascading_router_node`

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
    LangGraph node: Classify intent using domain-aware cascading router.
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
        "domain": result["domain"],                     # NEW: which API domain
        "api_endpoint": result["api_endpoint"],         # NEW: resolved API endpoint
        "classification_tier": result["tier"],           # NEW: which tier resolved it
        "tier3_reasoning": result.get("reasoning", ""),  # NEW: Gemini's reasoning (if used)
    }
```

### 11.2 Updated Graph Wiring

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
# PLUS domain/api_endpoint for downstream nodes to use directly
```

### 11.3 LangGraph Flow Comparison

```
CURRENT FLOW:
  orchestrator → safety → cache → intent_agent → confidence_checker
       → [low confidence] → llm_judge → confidence_checker (loop)
       → [high confidence] → build_context → call_api → response

NEW FLOW (with Domain-Aware Cascading Router):
  orchestrator → safety → cache → cascading_router → confidence_checker
       → [Tier 0: general] → response_agent → END (no API call)
       → [Tier 1+2 resolved] → build_context → call_api → response
       → [Tier 3 clarification] → clarification → response (ask user)
       
  Key differences:
  1. The llm_judge node is NO LONGER NEEDED for intent routing.
     Tier 3 handles disambiguation internally.
  2. Domain + API endpoint are determined at classification time,
     so build_context and call_api know which API to call.
  3. General intents (greeting/help/oos) short-circuit at Tier 0
     without any embedding or API calls.
```

### 11.4 Domain-Aware API Routing

The cascading router output includes the domain's API endpoint, so `call_claims_tool_node` can route directly:

```python
# tools/claims_api.py (modified section)

async def call_claims_tool_node(state: AgentState) -> dict:
    domain = state.get("domain")
    api_endpoint = state.get("api_endpoint")
    
    if domain == "general":
        # No API call needed — handled by response agent
        return {"api_response": None}
    
    # Use the domain's endpoint directly (no more intent → endpoint lookup)
    endpoint = api_endpoint or DOMAIN_REGISTRY[domain]["api_endpoint"]
    
    # Call the API...
```

---

## 12. Adding a New Domain (Onboarding Playbook)

### Example A: Adding a New Intent to an Existing Domain

**Scenario**: Add `"prior_auth_info"` to the **Cap-API** domain.

**Total time: ~10 minutes. Zero code changes.**

#### Step 1: Add Intent Description (1 min)

```python
# classifiers/cascading/intent_descriptions.py
INTENT_DESCRIPTIONS["prior_auth_info"] = (
    "Prior authorization (PA) status, PA number, Smart PA configuration, "
    "Member PA details, approval or denial status, and authorization "
    "requirements for the prescribed medication."
)
```

#### Step 2: Register in Domain (1 min)

```python
# classifiers/cascading/domain_registry.py
DOMAIN_REGISTRY["cap_api"]["intents"].append("prior_auth_info")
```

#### Step 3: Generate Synthetic Prompts (5 min)

```bash
python scripts/generate_synthetic_prompts.py \
    --intent prior_auth_info \
    --domain cap_api \
    --avoid-same-domain "approval_info,rejection_reasons" \
    --avoid-other-domains "approval_info:benefits_api" \
    --count 50
```

#### Step 4: Regenerate Centroids (2 min)

```bash
python scripts/generate_centroids.py
# Output: ✅ Saved 25 intent centroids + 4 domain centroids
```

#### Step 5: Verify (1 min)

```bash
python scripts/benchmark_classifiers.py --intent prior_auth_info --domain cap_api
# ✅ 50/50 correctly classified to cap_api/prior_auth_info
# ✅ No regression on other Cap-API intents
# ✅ No regression on other domains
```

---

### Example B: Adding an Entirely New Domain

**Scenario**: Add a new **"Formulary API"** domain with 3 intents.

**Total time: ~20 minutes. Minimal code changes (just the registry).**

#### Step 1: Define the Domain (2 min)

```python
# classifiers/cascading/domain_registry.py
DOMAIN_REGISTRY["formulary_api"] = {
    "name": "Formulary API",
    "description": (
        "Drug formulary lookup API: formulary tier, coverage status, "
        "step therapy requirements, quantity limits, and prior auth requirements."
    ),
    "api_endpoint": "/myclaims/formulary/v1/drug",
    "intents": [
        "formulary_lookup",      # Is this drug on the formulary?
        "step_therapy_info",     # Step therapy requirements
        "quantity_limit_info",   # Quantity and day supply limits
    ],
}
```

#### Step 2: Write Intent Descriptions (3 min)

```python
INTENT_DESCRIPTIONS["formulary_lookup"] = (
    "Formulary tier lookup for a specific drug, including whether "
    "the medication is on the plan's formulary, its coverage tier, "
    "and preferred/non-preferred status."
)
INTENT_DESCRIPTIONS["step_therapy_info"] = (
    "Step therapy requirements for a medication, including which "
    "drugs must be tried first before this drug is covered."
)
INTENT_DESCRIPTIONS["quantity_limit_info"] = (
    "Quantity limits, day supply limits, and dosage restrictions "
    "for a specific medication on the plan formulary."
)
```

#### Step 3: Generate Prompts + Centroids (10 min)

```bash
# Generate 50 prompts for each new intent
python scripts/generate_synthetic_prompts.py \
    --domain formulary_api \
    --all-intents \
    --count 50

# Regenerate all centroids (includes new domain centroid)
python scripts/generate_centroids.py
# Output: ✅ Saved 27 intent centroids + 5 domain centroids
```

#### Step 4: Add API Routing Config (3 min)

```python
# config/api_routing_config.py
"formulary_lookup": {
    "api_endpoint": "/myclaims/formulary/v1/drug",
    "method": "POST",
    "required_entities": ["drug_name_or_ndc"],
    "optional_entities": ["plan_id"],
    "description": "Formulary tier and coverage lookup",
}
# ... similarly for step_therapy_info and quantity_limit_info
```

#### Step 5: Verify (2 min)

```bash
python scripts/benchmark_classifiers.py --domain formulary_api
# ✅ All 3 intents correctly classified
# ✅ No regression on existing 4 domains
# ✅ New domain centroid properly separates from Cap-API
```

**That's it.** The LangGraph graph, cascading router code, and cross-encoder all automatically include the new domain — no code changes to any routing logic.

---

## 13. Performance & Cost Analysis

### 13.1 Latency Breakdown

| Component | Current System | Domain-Aware Cascading Router |
|-----------|---------------|-------------------------------|
| General fast-path (Tier 0) | N/A | <0.01ms (pattern match) |
| Query embedding | ~20ms | ~20ms (same) |
| Domain routing | N/A | <0.05ms (4 cosine ops) |
| Intent matching | ~2ms (480 cosine ops) | <0.05ms (3–12 within domain) |
| Cross-encoder rerank | N/A | ~5–10ms (3–5 pairs in domain) |
| **Subtotal (non-LLM)** | **~22ms** | **~25–30ms** |
| LLM judge (when triggered) | ~800ms (all 24 intents) | ~300ms (3–12 tools from 1 domain) |
| **LLM trigger rate** | **~30-40% of queries** | **~10-15% of queries** |
| **Average end-to-end** | **~250ms** | **~55ms** |

> **Domain routing advantage**: Because Tier 3 only receives tools from the selected domain (3–12 instead of 24), the Gemini context window is 50–80% smaller, making it faster AND cheaper.

### 13.2 Cost Comparison (per 10,000 queries)

| Cost Component | Current System | Domain-Aware Cascading |
|----------------|---------------|------------------------|
| Embedding API calls | 10,000 × $0.00002 = $0.20 | 10,000 × $0.00002 = $0.20 |
| LLM judge calls | 3,500 × $0.002 = $7.00 | 1,250 × $0.0008 = $1.00 |
| Cross-encoder (local) | $0.00 | $0.00 (runs on CPU) |
| **Total** | **$7.20** | **$1.20** |
| **Savings** | — | **83% cost reduction** |

> **Note**: LLM cost per call is lower because Tier 3 sends only 3–12 tool schemas from one domain (~100–200 tokens) instead of a full re-classification prompt across all intents (~800 tokens).

### 13.3 Accuracy Projections (by Domain)

| Metric | Current (Flat) | Cascading (Projected) |
|--------|---------------|----------------------|
| **Domain routing accuracy** | N/A | ~98% |
| Top-1 intent (Cap-API, 12 intents) | ~88% | ~96% |
| Top-1 intent (History, 6 intents) | ~90% | ~97% |
| Top-1 intent (Benefits, 3 intents) | ~93% | ~99% |
| Top-1 intent (General, 3 intents) | ~95% | ~99% |
| **Overall Top-1 accuracy** | **~90%** | **~97%** |
| Cross-domain misrouting rate | ~12% | ~2% |
| Clarification rate (instead of wrong answer) | ~5% | ~10% (intentional) |

---

## 14. Migration Strategy

### 14.1 Feature Flag Approach

```python
# config/config.py
class Settings:
    # Classifier selection
    use_cascading_router: bool = False       # NEW: Enable domain-aware cascading
    use_embedding_classifier: bool = True    # EXISTING: Flat embedding classifier
    
    # Cascading router sub-options
    cascading_tier3_enabled: bool = True     # Allow Gemini fallback
    cascading_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cascading_tier2_threshold: float = 0.85  # Confidence for immediate routing
    cascading_tier2_min_gap: float = 0.15    # Min gap between #1 and #2
    cascading_domain_min_gap: float = 0.10   # Min gap between top-2 domains
```

### 14.2 Migration Path

```
Phase 1: Shadow Mode
  ├── Both classifiers run in parallel
  ├── Current system routes traffic
  ├── Cascading router results logged for comparison
  ├── Per-domain accuracy tracked separately
  └── No user-facing impact

Phase 2: Canary (10%)
  ├── 10% of traffic uses cascading router
  ├── Monitor accuracy per domain, latency, cost
  └── Automated regression alerts

Phase 3: Gradual Rollout (10% → 50% → 100%)
  ├── Increase traffic percentage weekly
  ├── Tune thresholds per domain based on production data
  └── Document any edge cases

Phase 4: Deprecation
  ├── Remove embedded_classifier.py
  ├── Remove keyword_classifier.py
  ├── Remove llm_judge node (absorbed by Tier 3)
  └── Clean up config flags
```

### 14.3 Rollback Plan

If the cascading router underperforms in production:

1. Set `use_cascading_router = False` in config
2. The existing `embedded_classifier.py` immediately takes over
3. No code deployment needed — config change only

---

## Appendix

### A. Domain & Intent Reference Table

| Domain | Intent | Description (short) |
|--------|--------|---------------------|
| **Cap-API** | `claim_status` | Claim status / adjudication outcome |
| | `multi_claim_summary` | Summary of ALL claims for a member |
| | `pharmacy_info` | Dispensing pharmacy details |
| | `prescriber_info` | Prescribing physician details |
| | `pricing_info` | Copay, patient pay, cost breakdown |
| | `reimbursement_info` | Amount paid TO the pharmacy |
| | `rejection_reasons` | Rejection codes + resolution steps |
| | `settlement_info` | Pharmacy response/feedback codes |
| | `rx_details` | RX number, quantity, days supply |
| | `reversal_info` | Reversals, R&R, adjustments |
| | `cob_info` | Coordination of Benefits |
| | `generic_availability` | Generic alternatives |
| **Claim History** | `claim_status` | Claim status within search context |
| | `date_range_claims` | Claims in a date range |
| | `drug_info` | Drug name, NDC, formulary status |
| | `fill_date_info` | Prescription fill date |
| | `drug_interaction_info` | DUR edits, interaction alerts |
| | `compound_info` | Compound medication / MIC |
| **Benefits API** | `beneficiary_info` | Member benefit phase, coverage |
| | `audit_info` | Audit trail, change history |
| | `approval_info` | Approvals, TF, BPG, overrides |
| **General** | `greeting` | Hello, hi, welcome |
| | `help` | How to use the system |
| | `out_of_scope` | Unrelated queries |

### B. Cross-Encoder Model Comparison

| Model | Size | Accuracy (NDCG@10) | Speed (ms/pair) | Recommended For |
|-------|------|---------------------|-----------------|-----------------|
| `BAAI/bge-reranker-v2-m3` | 568M | 0.740 | ~15ms | Highest accuracy |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | 0.691 | ~5ms | **Best balance** ✅ |
| `BAAI/bge-reranker-base` | 278M | 0.712 | ~8ms | Good middle ground |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | 4.4M | 0.631 | ~2ms | Fastest |

**Recommendation**: Start with `ms-marco-MiniLM-L-6-v2` (22MB, 5ms/pair). Upgrade to `bge-reranker-v2-m3` if accuracy demands it.

### C. Embedding Dimensions by Provider

| Provider | Model | Dimension | Centroid File Size (28 centroids) |
|----------|-------|-----------|-----------------------------------|
| Google Vertex AI | text-embedding-005 | 768 | ~84 KB |
| Azure OpenAI | text-embedding-ada-002 | 1536 | ~168 KB |
| Azure OpenAI | text-embedding-3-small | 1536 | ~168 KB |

### D. Confidence Threshold Tuning Guide

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

| Domain Gap Threshold | Effect |
|---------------------|--------|
| 0.20 | Conservative — frequently pulls intents from top-2 domains |
| **0.10** | **Recommended** — only pulls from second domain when truly close |
| 0.03 | Aggressive — almost never considers second domain |

### E. Glossary

| Term | Definition |
|------|------------|
| **Domain** | A group of related intents backed by a single API endpoint (e.g., Cap-API, Benefits API) |
| **Centroid** | The mean (average) vector of all embeddings for an intent's (or domain's) synthetic prompts |
| **Cross-Encoder** | A transformer model that processes two texts jointly (versus independently) for more accurate similarity |
| **Bi-Encoder** | The current approach — embeds query and examples independently, then compares vectors |
| **Tool-Calling** | LLM feature where the model selects from predefined function signatures |
| **Shortlist** | The Top-K candidates from Tier 1, passed to Tier 2 for reranking |
| **Domain Routing** | First-level classification that determines which API domain a query belongs to |
| **Cross-Domain Intent** | An intent (like `claim_status`) that can be served by multiple domains |
| **Reranking** | Re-scoring a shortlist using a more accurate (but slower) model |
| **Escalation** | Passing a query to the next tier when the current tier cannot decide |
| **TF** | Transition Fill — temporary coverage during plan changes |
| **DUR** | Drug Utilization Review — clinical safety screening |
| **COB** | Coordination of Benefits — dual insurance adjudication |
| **MIC** | Most Ingredient Cost — compound drug pricing |
| **BPG** | Benefit Plan Group — plan configuration identifier |

---

### F. Domain Imbalance & Cluster Overlap Problem (v2 Fix)

#### F.1 The Problem

When domains have **vastly different numbers of intents**, naive centroid computation creates accuracy drops:

| Domain | Intents | Raw Embeddings (v1) | Effect |
|--------|---------|---------------------|--------|
| Cap-API | 12 | 240 vectors averaged | Centroid = vague "center of pharmacy" |
| Claim History | 6 | 120 vectors averaged | Centroid = moderate spread |
| Benefits API | 3 | 60 vectors averaged | Centroid = tight cluster |
| **New Overrides Domain** | **15** | **300 vectors averaged** | **Centroid overlaps Benefits API** |

**Why this fails**: The `cap_api` centroid is the average of 240 vectors spanning 12 diverse topics (pricing, pharmacy, COB, reversals...). This creates a "gravitational well" that absorbs queries from neighboring domains. A small domain like `benefits_api` (3 focused intents) gets overwhelmed when its centroid overlaps with a larger domain's spread.

**Observed failure**: With the v1 approach:
- Intent Accuracy: **72.04%**
- Domain Accuracy: **61.14%** (nearly 40% of queries routed to wrong API!)

#### F.2 Root Causes Identified

| # | Root Cause | Impact |
|---|-----------|--------|
| 1 | **Domain centroids from raw embeddings** — larger domains contribute more vectors, diluting the centroid | Domain accuracy ~61% |
| 2 | **Euclidean distance** — poor discriminator in 768-dimensional space | Intent accuracy ~72% |
| 3 | **Flat classification** — domain and intent classified independently | Cross-domain misrouting |
| 4 | **No multi-domain fallback** — ambiguous queries locked to single domain | Benefits API queries stolen by Cap-API |

#### F.3 The Fix: Three-Layer Defense

**Fix 1: Intent-Weighted Domain Centroids**

```python
# v1 (BROKEN): Domain centroid = mean(ALL raw embeddings)
#   cap_api centroid = mean(240 vectors from 12 intents)
#   benefits_api centroid = mean(60 vectors from 3 intents)
#   → Cap-API centroid absorbs everything

# v2 (FIXED): Domain centroid = mean(INTENT centroids)
#   cap_api centroid = mean(12 intent centroids)  — each intent = 1 vote
#   benefits_api centroid = mean(3 intent centroids) — each intent = 1 vote
#   → Each intent contributes equally regardless of example count
```

This ensures that a domain with 3 tightly-focused intents produces a tight, well-defined centroid that clearly separates from a domain with 15 diverse intents.

**Fix 2: Cosine Similarity Instead of Euclidean Distance**

```python
# v1 (BROKEN): Euclidean distance — magnitude-sensitive
#   "COB summary" → cap_api (Euclidean: 1.23) vs benefits_api (1.25)
#   Wrong! Margin is only 0.02 in magnitude space

# v2 (FIXED): Cosine similarity — direction-sensitive  
#   "COB summary" → cap_api (cosine: 0.87) vs benefits_api (0.72)
#   Correct! Clear 0.15 margin in angular space
```

**Fix 3: Hierarchical Classification with Multi-Domain Fallback**

```
User Query: "Show the approval overrides for this claim"
    │
    ▼
[1] DOMAIN ROUTING (cosine similarity)
    │  benefits_api:  0.83  ← winner
    │  cap_api:       0.81  ← close second
    │  gap: 0.02 < threshold (0.03) → AMBIGUOUS
    │
    ▼
[2] MULTI-DOMAIN INTENT SEARCH (consider BOTH domains)
    │  From benefits_api:
    │    approval_info:    0.91 ✅  ← "approval overrides" = TF/BPG
    │    audit_info:       0.68
    │    beneficiary_info: 0.54
    │  From cap_api:
    │    rejection_reasons: 0.72
    │    claim_status:     0.65
    │    ...
    │
    ▼
[3] BEST INTENT DETERMINES DOMAIN
    │  approval_info (0.91) > rejection_reasons (0.72)
    │  → Route to benefits_api / approval_info ✅
```

The key insight: **even when domain routing is ambiguous, intent-level comparison resolves the ambiguity** because intent centroids are more discriminative than domain centroids.

#### F.4 Handling the "New Large Domain" Scenario

When a new `overrides` domain with 15+ intents is added and overlaps with `benefits_api`:

| Strategy | How It Helps |
|----------|-------------|
| **Intent-weighted centroids** | The new domain's centroid is the mean of 15 intent centroids, not 300 raw vectors. Each intent gets 1 vote, preventing the centroid from becoming a vague blob. |
| **Multi-domain fallback** | When `overrides` and `benefits_api` centroids are close (gap < threshold), the system searches intents in BOTH domains. The winning intent determines the final domain. |
| **Cluster tightness matters** | `benefits_api` has 3 tightly clustered intents (avg intra-domain similarity ~0.85). `overrides` with 15 spread intents has lower tightness (~0.72). For queries that truly belong to `benefits_api`, its intent centroids produce higher similarity scores. |
| **Discriminative descriptions** | Each domain's intent descriptions are crafted to exclude vocabulary from overlapping domains (see Section 8.2). |
| **Pre-deployment overlap check** | `analyze_domain_overlap()` measures domain↔domain similarity before deployment. If overlap > 0.90, the system warns and suggests improving intent descriptions. |

#### F.5 Implementation: `intent_detection_v2.py`

The v2 implementation in `intent_detection_v2.py` includes:

```python
# 1. Intent-weighted domain centroids
domain_centroid = mean([intent_centroids[i] for i in domain_intents])

# 2. Cosine similarity
similarity = dot(query, centroid) / (norm(query) * norm(centroid))

# 3. Hierarchical classification with multi-domain fallback
if domain_gap < threshold:
    search_intents_in_top_k_domains()
    best_intent_determines_final_domain()

# 4. Diagnostic: Domain overlap analysis
analyze_domain_overlap()  # Detects problematic overlap before deployment
```

Run the v2 evaluation:
```bash
cd Intent_detection_system
python intent_detection_v2.py
```

This produces:
- Per-domain accuracy breakdown
- Domain confusion matrix
- Domain cluster overlap analysis
- Full ablation: v1 baseline vs v2 improvements

---

### G. References

- [BGE Reranker v2](https://huggingface.co/BAAI/bge-reranker-v2-m3) — Cross-encoder model
- [ms-marco-MiniLM](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) — Lightweight cross-encoder
- [Gemini 2.5 Flash](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini) — LLM for Tier 3
- [Matryoshka Embeddings](https://arxiv.org/abs/2205.13147) — Efficient embedding compression
- [ColBERT v2](https://arxiv.org/abs/2112.01488) — Late-interaction reranking (future consideration)
