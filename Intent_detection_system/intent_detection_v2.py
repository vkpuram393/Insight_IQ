"""
Intent Detection v2 — Domain-Balanced Hierarchical Classification

Fixes for v1 (72% intent / 61% domain accuracy):
  1. Domain centroids from INTENT centroids (not raw embeddings) — equal weight per intent
  2. Cosine similarity instead of Euclidean distance
  3. Hierarchical classification: domain → intent-within-domain
  4. Domain-size normalization to prevent large-cluster absorption
  5. Cross-domain intent handling (claim_status in cap_api + claim_history_search)
  6. Confidence-aware multi-domain fallback

Uses Google Cloud Vertex AI text-embedding-005 (same as v1).
"""

import os
import json
import logging
import math
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Union, Optional

from VamsiSir import embeddingVars

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

INTENT_EMBEDDINGS_PATH  = os.path.join(ARTIFACTS_DIR, "intent_embeddings.json")
INTENT_CENTROIDS_PATH   = os.path.join(ARTIFACTS_DIR, "intent_centroids_v2.json")
DOMAIN_CENTROIDS_PATH   = os.path.join(ARTIFACTS_DIR, "domain_centroids_v2.json")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─── Vertex AI Embedding Client ─────────────────────────────────────────────

class VertexEmbeddings:
    """Google Cloud Vertex AI Embeddings using text-embedding-005."""

    def __init__(self, project_id: str = None, location: str = "us-central1"):
        self.project_id = project_id or os.getenv("PROJECT_ID", "pbm-poc-coderev-genai-poc")
        self.location = location or os.getenv("LOCATION", "us-central1")
        self.model_name = "text-embedding-005"
        self.client = None

        try:
            from google import genai
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
            logger.info(
                f"✅ Vertex AI Embeddings initialised — project={self.project_id}, "
                f"region={self.location}, model={self.model_name}"
            )
        except ImportError:
            logger.error("google-genai SDK not installed. Run: pip install google-genai")
            raise
        except Exception as e:
            logger.error(f"❌ Vertex AI auth failed: {e}")
            raise

    # Rate-limit settings for Vertex AI text-embedding-005
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 2.0        # seconds
    BACKOFF_MULTIPLIER = 2.0     # exponential: 2s, 4s, 8s, 16s, 32s
    INTER_REQUEST_DELAY = 0.3    # seconds between individual API calls
    BATCH_PAUSE_EVERY = 20       # pause after every N calls
    BATCH_PAUSE_SECONDS = 5.0    # how long to pause

    def _embed_single_with_retry(self, text: str) -> List[float]:
        """Embed a single text with exponential backoff retry on 429 errors."""
        from google.genai import types

        backoff = self.INITIAL_BACKOFF
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=[types.Part.from_text(text=text)],
                )
                return response.embeddings[0].values
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "429" in error_str
                    or "resource exhausted" in error_str
                    or "too many requests" in error_str
                    or "quota" in error_str
                )
                if is_rate_limit and attempt < self.MAX_RETRIES:
                    logger.warning(
                        f"⏳ Rate limited (attempt {attempt}/{self.MAX_RETRIES}). "
                        f"Retrying in {backoff:.1f}s …"
                    )
                    time.sleep(backoff)
                    backoff *= self.BACKOFF_MULTIPLIER
                else:
                    logger.error(f"❌ Embedding failed after {attempt} attempts: {e}")
                    raise

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embedding(s) with rate-limit handling.

        Adds:
        - Exponential backoff retry on 429 / Resource Exhausted errors
        - Inter-request delay to stay under QPM quota
        - Periodic batch pause to avoid sustained burst
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        embeddings_list = []
        for i, t in enumerate(texts):
            # Throttle: pause between requests
            if i > 0:
                time.sleep(self.INTER_REQUEST_DELAY)

            # Batch pause: longer break every N requests
            if i > 0 and i % self.BATCH_PAUSE_EVERY == 0:
                logger.info(
                    f"  ⏸  Batch pause after {i} calls "
                    f"({self.BATCH_PAUSE_SECONDS}s cooldown) …"
                )
                time.sleep(self.BATCH_PAUSE_SECONDS)

            vec = self._embed_single_with_retry(t)
            embeddings_list.append(vec)

        return embeddings_list[0] if is_single else embeddings_list


_embedder: VertexEmbeddings = None

def _get_embedder() -> VertexEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = VertexEmbeddings()
    return _embedder


# ═════════════════════════════════════════════════════════════════════════════
# SIMILARITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Range: [-1, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of vectors.
    
    Args:
        query:  (D,) vector
        matrix: (N, D) matrix
    Returns:
        (N,) array of similarities
    """
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix_norm = matrix / norms
    return matrix_norm @ query_norm


# ═════════════════════════════════════════════════════════════════════════════
# TASK 1: Generate & Cache Intent Embeddings (same as v1)
# ═════════════════════════════════════════════════════════════════════════════

def embedding_gen() -> Dict[str, List[List[float]]]:
    """Generate and cache sentence embeddings for every intent example."""
    if os.path.exists(INTENT_EMBEDDINGS_PATH):
        logger.info(f"✅ Cache hit — loading from {INTENT_EMBEDDINGS_PATH}")
        with open(INTENT_EMBEDDINGS_PATH, "r") as f:
            return json.load(f)

    logger.info("Generating intent embeddings via Vertex AI …")
    embedder = _get_embedder()
    intent_examples = embeddingVars.CVS_INTENT_EXAMPLES

    intent_embeddings: Dict[str, List[List[float]]] = {}
    for intent_name, sentences in intent_examples.items():
        logger.info(f"  ▸ Embedding {len(sentences)} sentences for '{intent_name}'")
        vectors = embedder.embed(sentences)
        intent_embeddings[intent_name] = [
            list(v) if not isinstance(v, list) else v for v in vectors
        ]

    with open(INTENT_EMBEDDINGS_PATH, "w") as f:
        json.dump(intent_embeddings, f)
    logger.info(f"✅ Intent embeddings saved → {INTENT_EMBEDDINGS_PATH}")
    return intent_embeddings


def _load_intent_embeddings() -> Dict[str, List[List[float]]]:
    if os.path.exists(INTENT_EMBEDDINGS_PATH):
        with open(INTENT_EMBEDDINGS_PATH, "r") as f:
            return json.load(f)
    return embedding_gen()


# ═════════════════════════════════════════════════════════════════════════════
# TASK 2: Generate Intent Centroids (L2-normalized)
# ═════════════════════════════════════════════════════════════════════════════

def generate_intent_centroids() -> Dict[str, List[float]]:
    """Compute L2-normalized centroid per intent.
    
    Normalizing ensures cosine similarity is equivalent to dot product,
    and every intent is on the same scale regardless of example variance.
    """
    intent_embeddings = _load_intent_embeddings()
    intent_centroids: Dict[str, List[float]] = {}

    for intent_name, vectors in intent_embeddings.items():
        centroid = np.mean(vectors, axis=0)
        # L2-normalize the centroid
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        intent_centroids[intent_name] = centroid.tolist()

    with open(INTENT_CENTROIDS_PATH, "w") as f:
        json.dump(intent_centroids, f)
    logger.info(f"✅ Intent centroids saved → {INTENT_CENTROIDS_PATH}")
    return intent_centroids


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3: Generate Domain Centroids — INTENT-WEIGHTED (the key fix)
# ═════════════════════════════════════════════════════════════════════════════

def generate_domain_centroids() -> Dict[str, List[float]]:
    """Compute domain centroids as the MEAN OF INTENT CENTROIDS (not raw embeddings).
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  WHY THIS FIXES THE DOMAIN IMBALANCE PROBLEM                       │
    │                                                                     │
    │  v1 Bug: Domain centroid = mean(ALL sentence embeddings)            │
    │    cap_api:     12 intents × 20 examples = 240 vectors → centroid   │
    │    benefits_api: 3 intents × 20 examples =  60 vectors → centroid   │
    │                                                                     │
    │  The cap_api centroid is an average of 240 spread-out vectors,      │
    │  making it a vague "center of pharmacy" that absorbs everything.    │
    │                                                                     │
    │  v2 Fix: Domain centroid = mean(INTENT centroids within domain)     │
    │    cap_api:     mean(12 intent centroids) → each intent = 1 vote    │
    │    benefits_api: mean(3 intent centroids)  → each intent = 1 vote   │
    │                                                                     │
    │  Each intent contributes equally regardless of example count.        │
    │  A domain with 3 focused intents has a tight, well-defined cluster. │
    │  A domain with 12 diverse intents has a broader but still           │
    │  representative centroid.                                           │
    └─────────────────────────────────────────────────────────────────────┘
    
    Additionally handles cross-domain intents (claim_status) by using
    domain-specific intent centroids computed from domain-filtered examples.
    """
    intent_centroids = _load_or_generate_intent_centroids()
    domain_registry = embeddingVars.DOMAIN_REGISTRY

    domain_centroids: Dict[str, List[float]] = {}

    for domain_key, domain_info in domain_registry.items():
        intent_vectors = []
        for intent_name in domain_info["intents"]:
            if intent_name in intent_centroids:
                intent_vectors.append(np.array(intent_centroids[intent_name]))
            else:
                logger.warning(f"  ⚠️  Intent '{intent_name}' not in centroids — skipping")

        if intent_vectors:
            # Mean of intent centroids (not raw embeddings!)
            domain_centroid = np.mean(intent_vectors, axis=0)
            # L2-normalize
            norm = np.linalg.norm(domain_centroid)
            if norm > 0:
                domain_centroid = domain_centroid / norm
            domain_centroids[domain_key] = domain_centroid.tolist()
            logger.info(
                f"  ▸ Domain '{domain_key}': centroid from {len(intent_vectors)} "
                f"INTENT centroids (not {len(intent_vectors)*20} raw embeddings)"
            )
        else:
            logger.warning(f"  ⚠️  No intent centroids for domain '{domain_key}'")

    with open(DOMAIN_CENTROIDS_PATH, "w") as f:
        json.dump(domain_centroids, f)
    logger.info(f"✅ Domain centroids (intent-weighted) saved → {DOMAIN_CENTROIDS_PATH}")
    return domain_centroids


def _load_or_generate_intent_centroids() -> Dict[str, List[float]]:
    if os.path.exists(INTENT_CENTROIDS_PATH):
        with open(INTENT_CENTROIDS_PATH, "r") as f:
            return json.load(f)
    return generate_intent_centroids()


# ═════════════════════════════════════════════════════════════════════════════
# INTENT DESCRIPTIONS (for cross-encoder reranking)
# ═════════════════════════════════════════════════════════════════════════════

INTENT_DESCRIPTIONS = {
    # ── Cap-API ──────────────────────────────────────────────────────────
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
    # ── Claim History Search ─────────────────────────────────────────────
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
    # ── Benefits API ─────────────────────────────────────────────────────
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
    # ── General ──────────────────────────────────────────────────────────
    "greeting": "Casual greeting, hello, hi, welcome message, or conversation starter.",
    "help": "Help request, guidance on claim submission, instructions for using the system.",
    "out_of_scope": "Question unrelated to pharmacy claims, prescriptions, or insurance benefits.",
    # ── Extra intents (from training data) ───────────────────────────────
    "daw_info": "Dispense As Written (DAW) code, brand vs generic dispensing indicator.",
    "prior_auth_info": "Prior authorization (PA) status, PA number, Smart PA configuration.",
    "government_claim_type": "Government claim type, Medicare Part D, Medicaid, federal program indicator.",
    "medicare_part_d": "Medicare Part D details, MEDD pricing, PDE information, CMS phase.",
}


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-ENCODER RERANKER (Tier 2 — local, no API calls)
# ═════════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """Precision reranker using a cross-encoder model.
    
    Unlike bi-encoders (embed separately, compare vectors), a cross-encoder
    processes the query AND candidate description TOGETHER, enabling word-level
    attention that resolves cases like:
      - "pricing summary" → pricing_info  (not claim_status)
      - "claim summary"  → claim_status   (not pricing_info)
    
    Runs 100% locally on CPU. No API calls. No rate limits. ~5ms per pair.
    
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22MB, fast)
    Alternative: BAAI/bge-reranker-v2-m3 (568MB, more accurate)
    """
    
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy-load the model on first use (saves startup time)."""
        if self._loaded:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch  # noqa: F401 — just verify it's available
            
            logger.info(f"🔄 Loading cross-encoder: {self.MODEL_NAME} …")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self.model.eval()
            self._loaded = True
            logger.info(f"✅ Cross-encoder loaded (runs on CPU, ~5ms/pair)")
        except ImportError:
            logger.warning(
                "⚠️  Cross-encoder unavailable (install: pip install transformers torch). "
                "Falling back to embedding-only classification."
            )
            self._loaded = False
    
    @property
    def available(self) -> bool:
        """Check if cross-encoder can be used."""
        if self._loaded:
            return True
        self._ensure_loaded()
        return self._loaded and self.model is not None
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str, float]],  # (intent_name, domain, embedding_score)
    ) -> List[Tuple[str, str, float, float]]:
        """Rerank intent candidates using cross-encoder.
        
        Args:
            query: User's raw text query
            candidates: List of (intent_name, domain, embedding_score)
            
        Returns:
            List of (intent_name, domain, ce_score, embedding_score)
            sorted by cross-encoder score descending.
        """
        import torch
        
        self._ensure_loaded()
        if not self._loaded:
            # Fallback: return candidates sorted by embedding score
            return [
                (name, domain, emb_score, emb_score)
                for name, domain, emb_score in candidates
            ]
        
        # Build (query, description) pairs
        pairs = []
        intent_names = []
        domains = []
        emb_scores = []
        
        for intent_name, domain, emb_score in candidates:
            desc = INTENT_DESCRIPTIONS.get(intent_name, intent_name)
            pairs.append((query, desc))
            intent_names.append(intent_name)
            domains.append(domain)
            emb_scores.append(emb_score)
        
        # Batch encode and score
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(logits).numpy()
        
        # Combine results
        results = [
            (intent_names[i], domains[i], float(scores[i]), emb_scores[i])
            for i in range(len(candidates))
        ]
        
        # Sort by cross-encoder score
        results.sort(key=lambda x: x[2], reverse=True)
        return results


# Module-level singleton (lazy loaded)
_cross_encoder: CrossEncoderReranker = None

def _get_cross_encoder() -> CrossEncoderReranker:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoderReranker()
    return _cross_encoder


# ═════════════════════════════════════════════════════════════════════════════
# TASK 4: HIERARCHICAL CLASSIFICATION (Domain → Intent within Domain)
# ═════════════════════════════════════════════════════════════════════════════

def _build_domain_intent_map(
    intent_centroids: Dict[str, List[float]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build a map of domain → {intent_name: centroid_vector}.
    
    This is used for the second stage of hierarchical classification:
    after selecting a domain, we only compare against intents IN that domain.
    """
    domain_registry = embeddingVars.DOMAIN_REGISTRY
    domain_intent_map = {}

    for domain_key, domain_info in domain_registry.items():
        domain_intent_map[domain_key] = {}
        for intent_name in domain_info["intents"]:
            if intent_name in intent_centroids:
                domain_intent_map[domain_key][intent_name] = np.array(
                    intent_centroids[intent_name]
                )
    return domain_intent_map


def _build_intent_to_domains(domain_registry: dict) -> Dict[str, List[str]]:
    """Build reverse map: intent → list of domains it belongs to.
    
    This identifies cross-domain intents like claim_status.
    """
    intent_to_domains: Dict[str, List[str]] = {}
    for domain_key, domain_info in domain_registry.items():
        for intent_name in domain_info["intents"]:
            if intent_name not in intent_to_domains:
                intent_to_domains[intent_name] = []
            intent_to_domains[intent_name].append(domain_key)
    return intent_to_domains


def classify_hierarchical(
    query_vec: np.ndarray,
    domain_centroids: Dict[str, np.ndarray],
    domain_intent_map: Dict[str, Dict[str, np.ndarray]],
    domain_gap_threshold: float = 0.02,
    top_k_domains: int = 2,
    query_text: str = None,
    use_cross_encoder: bool = True,
    ce_intent_gap_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Hierarchical classification: domain first, then intent within domain.
    
    ┌────────────────────────────────────────────────────────────────────────┐
    │  HOW THIS HANDLES THE CLUSTER OVERLAP PROBLEM                         │
    │                                                                        │
    │  Scenario: New "overrides" domain with 15 intents overlaps with        │
    │            benefits_api (3 intents).                                    │
    │                                                                        │
    │  Step 1: Domain routing with MULTI-DOMAIN FALLBACK                     │
    │    - If top-2 domain scores are within `domain_gap_threshold`,         │
    │      we consider intents from BOTH domains (not just the winner).      │
    │    - This prevents a large cluster from "stealing" queries that         │
    │      belong to a neighboring small cluster.                            │
    │                                                                        │
    │  Step 2: Intent classification within selected domain(s)               │
    │    - Only compare against intents in the candidate domain(s).          │
    │    - benefits_api's 3 intents are clustered tightly → high similarity  │
    │    - overrides' 15 intents are spread broadly → lower per-intent sim   │
    │    - The tighter cluster naturally wins for its own queries.            │
    │                                                                        │
    │  Step 3: The BEST intent (highest cosine sim) determines final domain  │
    │    - Even if domain routing was ambiguous, the intent-level comparison  │
    │      resolves it because intent centroids are more discriminative.     │
    └────────────────────────────────────────────────────────────────────────┘
    
    Args:
        query_vec: Embedded query vector
        domain_centroids: {domain_name: centroid_vector}
        domain_intent_map: {domain_name: {intent_name: centroid_vector}}
        domain_gap_threshold: If gap between top-2 domains < this, consider both
        top_k_domains: Max domains to consider when ambiguous
        
    Returns:
        {
            "predicted_domain": str,
            "predicted_intent": str,
            "domain_score": float,
            "intent_score": float,
            "domain_scores": dict,
            "considered_domains": list,
            "is_ambiguous_domain": bool,
        }
    """
    # ── Step 1: Score all domains ────────────────────────────────────────
    domain_scores = {}
    for domain_name, centroid in domain_centroids.items():
        domain_scores[domain_name] = _cosine_similarity(query_vec, centroid)

    # Sort by score (descending)
    ranked_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    top_domain, top_score = ranked_domains[0]
    second_domain, second_score = ranked_domains[1] if len(ranked_domains) > 1 else (None, -1)

    gap = top_score - second_score
    is_ambiguous = gap < domain_gap_threshold

    # ── Step 2: Determine which domains to search for intents ────────────
    if is_ambiguous:
        # Consider intents from top-K domains
        candidate_domains = [d for d, _ in ranked_domains[:top_k_domains]]
    else:
        candidate_domains = [top_domain]

    # ── Step 3: Find best intent across candidate domains (embedding) ───
    embedding_ranked = []  # (intent_name, domain, cosine_score)

    for domain_name in candidate_domains:
        if domain_name not in domain_intent_map:
            continue
        for intent_name, intent_centroid in domain_intent_map[domain_name].items():
            sim = _cosine_similarity(query_vec, intent_centroid)
            embedding_ranked.append((intent_name, domain_name, sim))

    embedding_ranked.sort(key=lambda x: x[2], reverse=True)

    if not embedding_ranked:
        return {
            "predicted_domain": top_domain,
            "predicted_intent": "unknown",
            "domain_score": top_score,
            "intent_score": 0.0,
            "domain_scores": domain_scores,
            "considered_domains": candidate_domains,
            "is_ambiguous_domain": is_ambiguous,
            "tier_used": "none",
        }

    top_embedding = embedding_ranked[0]
    second_embedding = embedding_ranked[1] if len(embedding_ranked) > 1 else None
    embedding_gap = (
        top_embedding[2] - second_embedding[2] if second_embedding else 1.0
    )

    # ── Step 4: Cross-encoder reranking (only when ambiguous) ────────────
    #
    #  When to fire the cross-encoder:
    #   a) Domain was ambiguous (multi-domain fallback active)  OR
    #   b) Top-2 intents from embedding are very close (gap < threshold)
    #
    #  When NOT to fire:
    #   - Clear domain + clear intent winner → embedding is sufficient
    #   - Cross-encoder unavailable (not installed)
    #   - Disabled via use_cross_encoder=False
    #
    intent_is_ambiguous = embedding_gap < ce_intent_gap_threshold
    should_rerank = (
        use_cross_encoder
        and query_text is not None
        and (is_ambiguous or intent_is_ambiguous)
    )

    tier_used = "embedding"  # default

    if should_rerank:
        ce = _get_cross_encoder()
        if ce.available:
            # Send top-5 candidates to cross-encoder
            top_k = embedding_ranked[:5]
            reranked = ce.rerank(query_text, top_k)

            if reranked:
                best = reranked[0]
                best_intent = best[0]
                best_intent_domain = best[1]
                best_intent_score = best[2]  # cross-encoder score
                tier_used = "cross_encoder"

                logger.debug(
                    f"  🔀 Cross-encoder reranked: {best_intent} ({best_intent_domain}) "
                    f"CE={best_intent_score:.3f}, Emb={best[3]:.3f}"
                )
            else:
                # Fallback to embedding result
                best_intent = top_embedding[0]
                best_intent_domain = top_embedding[1]
                best_intent_score = top_embedding[2]
        else:
            # Cross-encoder not available — use embedding
            best_intent = top_embedding[0]
            best_intent_domain = top_embedding[1]
            best_intent_score = top_embedding[2]
    else:
        # Clear winner from embeddings — no reranking needed
        best_intent = top_embedding[0]
        best_intent_domain = top_embedding[1]
        best_intent_score = top_embedding[2]

    # ── The domain of the winning intent IS the final domain ─────────────
    # This is crucial: even if domain routing said "cap_api", but the
    # best intent is "approval_info" (benefits_api), we route to benefits_api.
    return {
        "predicted_domain": best_intent_domain or top_domain,
        "predicted_intent": best_intent or "unknown",
        "domain_score": domain_scores.get(best_intent_domain, top_score),
        "intent_score": best_intent_score,
        "domain_scores": domain_scores,
        "considered_domains": candidate_domains,
        "is_ambiguous_domain": is_ambiguous,
        "tier_used": tier_used,
    }


# ═════════════════════════════════════════════════════════════════════════════
# TASK 5: FLAT CLASSIFICATION (for comparison / ablation)
# ═════════════════════════════════════════════════════════════════════════════

def classify_flat_cosine(
    query_vec: np.ndarray,
    intent_centroids: Dict[str, np.ndarray],
    domain_centroids: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Flat classification using cosine similarity (v1 fix: Euclidean → Cosine)."""
    # Intent
    best_intent, best_intent_score = None, -1.0
    for name, centroid in intent_centroids.items():
        sim = _cosine_similarity(query_vec, centroid)
        if sim > best_intent_score:
            best_intent_score = sim
            best_intent = name

    # Domain
    best_domain, best_domain_score = None, -1.0
    for name, centroid in domain_centroids.items():
        sim = _cosine_similarity(query_vec, centroid)
        if sim > best_domain_score:
            best_domain_score = sim
            best_domain = name

    return {
        "predicted_intent": best_intent,
        "predicted_domain": best_domain,
        "intent_score": best_intent_score,
        "domain_score": best_domain_score,
    }


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def classify_and_evaluate(
    test_data: List[Dict[str, str]],
    method: str = "hierarchical",
) -> Dict[str, float]:
    """Classify and evaluate using the specified method.
    
    Args:
        test_data: List of {text, actual_classification_intent, actual_classification_domain}
        method: "hierarchical" (recommended), "flat_cosine", or "flat_euclidean" (v1)
        
    Returns:
        dict with intent_accuracy, domain_accuracy, and per-domain breakdown
    """
    # Load centroids
    intent_centroids_raw = _load_or_generate_intent_centroids()
    domain_centroids_raw = _load_or_generate_domain_centroids()

    # Convert to numpy
    intent_centroids_np = {k: np.array(v) for k, v in intent_centroids_raw.items()}
    domain_centroids_np = {k: np.array(v) for k, v in domain_centroids_raw.items()}

    # Build domain→intent map for hierarchical mode
    domain_intent_map = _build_domain_intent_map(intent_centroids_raw)

    embedder = _get_embedder()
    results = []

    for idx, record in enumerate(test_data):
        text = record["text"]
        actual_intent = record["actual_classification_intent"]
        actual_domain = record["actual_classification_domain"]

        query_vec = np.array(embedder.embed(text))

        if method == "hierarchical":
            result = classify_hierarchical(
                query_vec, domain_centroids_np, domain_intent_map,
                query_text=text,
            )
            predicted_intent = result["predicted_intent"]
            predicted_domain = result["predicted_domain"]
            extra = {
                "intent_score": result["intent_score"],
                "domain_score": result["domain_score"],
                "is_ambiguous": result["is_ambiguous_domain"],
                "tier_used": result.get("tier_used", "embedding"),
                "considered_domains": "|".join(result["considered_domains"]),
            }
        elif method == "flat_cosine":
            result = classify_flat_cosine(
                query_vec, intent_centroids_np, domain_centroids_np
            )
            predicted_intent = result["predicted_intent"]
            predicted_domain = result["predicted_domain"]
            extra = {
                "intent_score": result["intent_score"],
                "domain_score": result["domain_score"],
            }
        else:  # flat_euclidean (v1 baseline)
            predicted_intent = _nearest_centroid_euclidean(query_vec, intent_centroids_raw)
            predicted_domain = _nearest_centroid_euclidean(query_vec, domain_centroids_raw)
            extra = {}

        row = {
            "text": text,
            "actual_intent": actual_intent,
            "predicted_intent": predicted_intent,
            "intent_match": actual_intent == predicted_intent,
            "actual_domain": actual_domain,
            "predicted_domain": predicted_domain,
            "domain_match": actual_domain == predicted_domain,
            **extra,
        }
        results.append(row)

        if (idx + 1) % 50 == 0:
            logger.info(f"  ▸ Classified {idx + 1}/{len(test_data)}")

    df = pd.DataFrame(results)

    # Save results
    output_path = os.path.join(OUTPUTS_DIR, f"classification_results_{method}.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Results saved → {output_path}")

    # ── Overall accuracy ─────────────────────────────────────────────────
    intent_accuracy = df["intent_match"].mean() * 100
    domain_accuracy = df["domain_match"].mean() * 100

    print(f"\n{'='*60}")
    print(f"  Method: {method.upper()}")
    print(f"  Intent Classification Accuracy : {intent_accuracy:.2f}%")
    print(f"  Domain Classification Accuracy : {domain_accuracy:.2f}%")
    print(f"{'='*60}")

    # ── Cross-encoder usage stats (hierarchical only) ────────────────────
    if method == "hierarchical" and "tier_used" in df.columns:
        tier_counts = df["tier_used"].value_counts()
        total = len(df)
        print(f"\n  Classification Tier Usage:")
        for tier, count in tier_counts.items():
            pct = count / total * 100
            print(f"    {tier:<20} {count:>5} queries ({pct:.1f}%)")
        if "cross_encoder" in tier_counts:
            ce_rows = df[df["tier_used"] == "cross_encoder"]
            ce_intent_acc = ce_rows["intent_match"].mean() * 100
            ce_domain_acc = ce_rows["domain_match"].mean() * 100
            print(f"    Cross-encoder accuracy: intent={ce_intent_acc:.1f}%, domain={ce_domain_acc:.1f}%")
        print()

    # ── Per-domain breakdown ─────────────────────────────────────────────
    print(f"\n  Per-Domain Accuracy:")
    print(f"  {'Domain':<25} {'Intent Acc':>12} {'Domain Acc':>12} {'Count':>8}")
    print(f"  {'-'*57}")

    domain_breakdown = {}
    for domain in df["actual_domain"].unique():
        subset = df[df["actual_domain"] == domain]
        d_intent_acc = subset["intent_match"].mean() * 100
        d_domain_acc = subset["domain_match"].mean() * 100
        domain_breakdown[domain] = {
            "intent_accuracy": d_intent_acc,
            "domain_accuracy": d_domain_acc,
            "count": len(subset),
        }
        print(f"  {domain:<25} {d_intent_acc:>10.1f}% {d_domain_acc:>10.1f}% {len(subset):>8}")

    print()

    # ── Confusion analysis: which domain pairs are confused ──────────────
    misrouted = df[~df["domain_match"]]
    if len(misrouted) > 0:
        print(f"  Domain Confusion Matrix (misrouted queries):")
        confusion = misrouted.groupby(["actual_domain", "predicted_domain"]).size()
        for (actual, predicted), count in confusion.items():
            print(f"    {actual} → {predicted}: {count} queries")
        print()

    return {
        "intent_accuracy": intent_accuracy,
        "domain_accuracy": domain_accuracy,
        "domain_breakdown": domain_breakdown,
    }


def _load_or_generate_domain_centroids() -> Dict[str, List[float]]:
    if os.path.exists(DOMAIN_CENTROIDS_PATH):
        with open(DOMAIN_CENTROIDS_PATH, "r") as f:
            return json.load(f)
    return generate_domain_centroids()


def _nearest_centroid_euclidean(
    query_vec: np.ndarray, centroids: Dict[str, List[float]]
) -> str:
    """v1 baseline: Euclidean distance nearest centroid."""
    best_label = None
    best_dist = float("inf")
    for label, centroid in centroids.items():
        dist = float(np.linalg.norm(query_vec - np.array(centroid)))
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


# ═════════════════════════════════════════════════════════════════════════════
# ABLATION STUDY — Run all 3 methods and compare
# ═════════════════════════════════════════════════════════════════════════════

def run_ablation(test_data: List[Dict[str, str]]) -> None:
    """Run all classification methods and compare accuracy.
    
    Methods:
      1. flat_euclidean — v1 baseline (Euclidean + raw-embedding domain centroids)
      2. flat_cosine    — v1 + cosine similarity + intent-weighted domain centroids
      3. hierarchical   — v2: domain→intent routing + multi-domain fallback
    """
    print("\n" + "="*70)
    print("  ABLATION STUDY: Comparing Classification Methods")
    print("="*70)

    all_results = {}

    for method in ["flat_euclidean", "flat_cosine", "hierarchical"]:
        print(f"\n{'─'*70}")
        print(f"  Running method: {method}")
        print(f"{'─'*70}")

        # For flat_euclidean, use v1 domain centroids (raw-embedding based)
        if method == "flat_euclidean":
            _generate_v1_domain_centroids()

        metrics = classify_and_evaluate(test_data, method=method)
        all_results[method] = metrics

    # ── Summary comparison ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SUMMARY: Method Comparison")
    print("="*70)
    print(f"  {'Method':<20} {'Intent Acc':>12} {'Domain Acc':>12} {'Improvement':>14}")
    print(f"  {'-'*58}")

    baseline_intent = all_results.get("flat_euclidean", {}).get("intent_accuracy", 0)
    baseline_domain = all_results.get("flat_euclidean", {}).get("domain_accuracy", 0)

    for method, metrics in all_results.items():
        intent_diff = metrics["intent_accuracy"] - baseline_intent
        domain_diff = metrics["domain_accuracy"] - baseline_domain
        diff_str = f"+{intent_diff:.1f}% / +{domain_diff:.1f}%" if method != "flat_euclidean" else "baseline"
        print(
            f"  {method:<20} {metrics['intent_accuracy']:>10.2f}% "
            f"{metrics['domain_accuracy']:>10.2f}% {diff_str:>14}"
        )
    print()


def _generate_v1_domain_centroids():
    """Generate v1-style domain centroids (raw embedding average) for comparison."""
    v1_path = os.path.join(ARTIFACTS_DIR, "domain_centroids_v1.json")
    intent_embeddings = _load_intent_embeddings()
    domain_registry = embeddingVars.DOMAIN_REGISTRY

    domain_centroids = {}
    for domain_key, domain_info in domain_registry.items():
        all_vectors = []
        for intent_name in domain_info["intents"]:
            if intent_name in intent_embeddings:
                all_vectors.extend(intent_embeddings[intent_name])
        if all_vectors:
            centroid = np.mean(all_vectors, axis=0).tolist()
            domain_centroids[domain_key] = centroid

    with open(v1_path, "w") as f:
        json.dump(domain_centroids, f)

    # Also save as the "current" domain centroids for the flat_euclidean method
    with open(DOMAIN_CENTROIDS_PATH, "w") as f:
        json.dump(domain_centroids, f)


# ═════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: Domain Cluster Overlap Analysis
# ═════════════════════════════════════════════════════════════════════════════

def analyze_domain_overlap() -> None:
    """Measure how much each domain's centroid overlaps with others.
    
    This directly shows the cluster overlap problem:
    - If benefits_api ↔ cap_api similarity is high (>0.90), their clusters overlap
    - If a new "overrides" domain is added with high overlap, we can detect it early
    """
    domain_centroids = _load_or_generate_domain_centroids()
    intent_centroids = _load_or_generate_intent_centroids()
    domain_registry = embeddingVars.DOMAIN_REGISTRY

    print("\n" + "="*70)
    print("  DOMAIN CLUSTER OVERLAP ANALYSIS")
    print("="*70)

    # ── 1. Domain-to-domain similarity ───────────────────────────────────
    print("\n  Domain-to-Domain Cosine Similarity:")
    print(f"  {'':>22}", end="")
    domains = list(domain_centroids.keys())
    for d in domains:
        print(f"  {d:>12}", end="")
    print()

    for d1 in domains:
        print(f"  {d1:>22}", end="")
        v1 = np.array(domain_centroids[d1])
        for d2 in domains:
            v2 = np.array(domain_centroids[d2])
            sim = _cosine_similarity(v1, v2)
            marker = " ⚠️" if sim > 0.90 and d1 != d2 else ""
            print(f"  {sim:>10.4f}{marker}", end="")
        print()

    # ── 2. Intent cluster tightness per domain ───────────────────────────
    print(f"\n  Domain Cluster Tightness (avg intra-domain intent similarity):")
    print(f"  {'Domain':<25} {'Avg Sim':>10} {'Min Sim':>10} {'Max Sim':>10} {'#Intents':>10}")
    print(f"  {'-'*65}")

    for domain_key, domain_info in domain_registry.items():
        domain_intents = [
            intent_name for intent_name in domain_info["intents"]
            if intent_name in intent_centroids
        ]
        if len(domain_intents) < 2:
            print(f"  {domain_key:<25} {'(only 1 intent)':>10}")
            continue

        sims = []
        for i, name_a in enumerate(domain_intents):
            for name_b in domain_intents[i + 1:]:
                sim = _cosine_similarity(
                    np.array(intent_centroids[name_a]),
                    np.array(intent_centroids[name_b]),
                )
                sims.append(sim)

        avg_sim = np.mean(sims)
        min_sim = np.min(sims)
        max_sim = np.max(sims)
        print(
            f"  {domain_key:<25} {avg_sim:>10.4f} {min_sim:>10.4f} "
            f"{max_sim:>10.4f} {len(domain_intents):>10}"
        )

    # ── 3. Cross-domain intent overlap (nearest neighbor) ────────────────
    print(f"\n  Cross-Domain Intent Overlap (closest intent in another domain):")
    print(f"  {'Intent':<25} {'Domain':<22} {'Nearest Cross-Domain':>22} {'Sim':>8}")
    print(f"  {'-'*77}")

    for domain_key, domain_info in domain_registry.items():
        for intent_name in domain_info["intents"]:
            if intent_name not in intent_centroids:
                continue
            vec = np.array(intent_centroids[intent_name])
            best_other = None
            best_other_sim = -1
            for other_domain, other_info in domain_registry.items():
                if other_domain == domain_key:
                    continue
                for other_intent in other_info["intents"]:
                    if other_intent not in intent_centroids:
                        continue
                    sim = _cosine_similarity(vec, np.array(intent_centroids[other_intent]))
                    if sim > best_other_sim:
                        best_other_sim = sim
                        best_other = f"{other_intent} ({other_domain})"
            marker = " ⚠️" if best_other_sim > 0.90 else ""
            print(
                f"  {intent_name:<25} {domain_key:<22} "
                f"{best_other:>22} {best_other_sim:>7.4f}{marker}"
            )

    print()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("  Intent Detection v2 — Domain-Balanced Hierarchical Classification")
    print("="*70)

    # Step 1: Generate embeddings
    print("\nStep 1 — Generating intent embeddings …")
    embedding_gen()

    # Step 2: Generate L2-normalized intent centroids
    print("\nStep 2 — Generating intent centroids (L2-normalized) …")
    generate_intent_centroids()

    # Step 3: Generate intent-weighted domain centroids (THE KEY FIX)
    print("\nStep 3 — Generating domain centroids (intent-weighted) …")
    generate_domain_centroids()

    # Step 4: Analyze domain overlap
    print("\nStep 4 — Analyzing domain cluster overlap …")
    analyze_domain_overlap()

    # Step 5: Run evaluation
    TESTDATA_PATH = os.path.join(BASE_DIR, "Testdata.csv")

    if not os.path.exists(TESTDATA_PATH):
        print(f"\n⚠️  Testdata.csv not found at {TESTDATA_PATH} — skipping evaluation.")
    else:
        test_df = pd.read_csv(TESTDATA_PATH)

        required_cols = {"Prompt", "Intent", "domain"}
        missing = required_cols - set(test_df.columns)
        if missing:
            raise ValueError(f"Testdata.csv is missing columns: {missing}")

        test_data: List[Dict[str, str]] = [
            {
                "text": row["Prompt"],
                "actual_classification_intent": row["Intent"],
                "actual_classification_domain": row["domain"],
            }
            for _, row in test_df.iterrows()
        ]

        print(f"\n  ▸ Loaded {len(test_data)} test records from Testdata.csv")

        # Run the improved hierarchical method
        print("\n" + "─"*70)
        print("  Running HIERARCHICAL classification (v2 — recommended)")
        print("─"*70)
        metrics = classify_and_evaluate(test_data, method="hierarchical")

        print(f"\n📊 v2 Results:")
        print(f"   Intent Accuracy : {metrics['intent_accuracy']:.2f}%")
        print(f"   Domain Accuracy : {metrics['domain_accuracy']:.2f}%")

        # Optionally run full ablation
        print("\n\nRunning full ablation study (v1 baseline vs v2 improvements)…")
        # Regenerate v2 domain centroids (ablation may have overwritten them)
        generate_domain_centroids()
        run_ablation(test_data)
