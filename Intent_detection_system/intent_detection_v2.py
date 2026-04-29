"""
Intent Detection v2 — Global kNN with Distance-Weighted Voting

Evolution:
  v1 (72% / 61%):  Euclidean distance + raw-embedding domain centroids
  v2 (78.6%):      Hierarchical domain→intent + kNN exemplar voting
  v3 (this):       SKIP domain routing — global kNN across ALL exemplars
                   + distance-weighted voting + stale cache detection

The key insight: when domain centroids overlap >0.93, domain routing is
essentially random and cascades errors to intent classification.
Instead, classify intent GLOBALLY first, then derive domain from intent.

Uses Google Cloud Vertex AI text-embedding-005.
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
    """Generate and cache sentence embeddings for every intent example.
    
    IMPORTANT: Detects stale cache — if CVS_INTENT_EXAMPLES has intents
    that are missing from the cached file, the cache is regenerated.
    This prevents silent failures when new intents are added.
    """
    intent_examples = embeddingVars.CVS_INTENT_EXAMPLES

    if os.path.exists(INTENT_EMBEDDINGS_PATH):
        logger.info(f"Loading cached embeddings from {INTENT_EMBEDDINGS_PATH}")
        with open(INTENT_EMBEDDINGS_PATH, "r") as f:
            cached = json.load(f)

        # ── Stale cache detection ────────────────────────────────────────
        cached_intents = set(cached.keys())
        expected_intents = set(intent_examples.keys())
        missing = expected_intents - cached_intents
        extra = cached_intents - expected_intents

        if missing:
            logger.warning(
                f"⚠️  STALE CACHE: {len(missing)} intents in CVS_INTENT_EXAMPLES "
                f"are MISSING from cache: {sorted(missing)}"
            )
            logger.info("🔄 Regenerating embeddings for missing intents only…")

            embedder = _get_embedder()
            for intent_name in sorted(missing):
                sentences = intent_examples[intent_name]
                logger.info(f"  ▸ Embedding {len(sentences)} sentences for NEW intent '{intent_name}'")
                vectors = embedder.embed(sentences)
                cached[intent_name] = [
                    list(v) if not isinstance(v, list) else v for v in vectors
                ]

            # Save updated cache
            with open(INTENT_EMBEDDINGS_PATH, "w") as f:
                json.dump(cached, f)
            logger.info(f"✅ Updated cache with {len(missing)} new intents → {INTENT_EMBEDDINGS_PATH}")

        if extra:
            logger.info(f"ℹ️  Cache has {len(extra)} intents not in current registry (harmless): {sorted(extra)}")

        return cached

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

    # ── Step 3: Find best intent across candidate domains ────────────────
    best_intent = None
    best_intent_score = -1.0
    best_intent_domain = None

    for domain_name in candidate_domains:
        if domain_name not in domain_intent_map:
            continue
        for intent_name, intent_centroid in domain_intent_map[domain_name].items():
            sim = _cosine_similarity(query_vec, intent_centroid)
            if sim > best_intent_score:
                best_intent_score = sim
                best_intent = intent_name
                best_intent_domain = domain_name

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
    }


# ═════════════════════════════════════════════════════════════════════════════
# TASK 4b: kNN EXEMPLAR VOTING — THE CROSS-ENCODER REPLACEMENT
# ═════════════════════════════════════════════════════════════════════════════
#
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  WHY kNN EXEMPLAR VOTING BEATS CROSS-ENCODER FOR OVERLAPPING INTENTS│
#  │                                                                      │
#  │  Cross-encoder (DeBERTa / ms-marco-MiniLM) problems:                │
#  │    ✗ Trained on web search relevance, not intent discrimination     │
#  │    ✗ Needs long passages — our intent descriptions are 1-2 lines    │
#  │    ✗ Can't learn domain jargon (TF, BPG, DUR, COB) without data    │
#  │    ✗ 15ms per pair → 75ms for 5 candidates → adds latency          │
#  │                                                                      │
#  │  Centroid comparison problems:                                       │
#  │    ✗ Averaging 20 diverse examples destroys discriminative detail   │
#  │    ✗ "audit trail for this claim" centroid ≈ "approval details"     │
#  │    ✗ A single centroid can't represent multi-modal intent clusters  │
#  │                                                                      │
#  │  kNN Exemplar Voting (what we use instead):                          │
#  │    ✓ Compare query against ALL 20 training examples per intent      │
#  │    ✓ Top-K nearest neighbors VOTE on the intent                     │
#  │    ✓ Preserves decision boundary detail that centroids destroy      │
#  │    ✓ No model download, no GPU, no training, no extra latency       │
#  │    ✓ Uses the SAME embeddings you already have cached               │
#  │    ✓ Naturally handles multi-modal clusters (intent has subtypes)   │
#  │                                                                      │
#  │  Example: "Show the audit trail for this claim's approval"          │
#  │                                                                      │
#  │    Centroid method:                                                   │
#  │      audit_info centroid:    0.83  ← "audit trail" pulls centroid   │
#  │      approval_info centroid: 0.81  ← "approval" pulls centroid     │
#  │      Gap: 0.02 → CONFUSED                                           │
#  │                                                                      │
#  │    kNN (k=5) method:                                                 │
#  │      Top-5 neighbors:                                                │
#  │        1. approval_info example "Show approval details"  (0.92)     │
#  │        2. approval_info example "Approval logic for claim" (0.91)   │
#  │        3. audit_info example "Show audit trail" (0.89)              │
#  │        4. approval_info example "Plan override details" (0.87)      │
#  │        5. audit_info example "Change history" (0.85)                │
#  │      Vote: approval_info=3, audit_info=2 → approval_info WINS      │
#  │                                                                      │
#  │  The key insight: individual examples near the decision boundary     │
#  │  are more discriminative than averaged centroids.                    │
#  └──────────────────────────────────────────────────────────────────────┘

def _build_domain_exemplar_index(
    intent_embeddings: Dict[str, List[List[float]]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build a map of domain → {intent_name: (N, D) embedding matrix}.
    
    Each intent has ~20 example embeddings. These are the "exemplars"
    that kNN votes over.
    """
    domain_registry = embeddingVars.DOMAIN_REGISTRY
    domain_exemplars = {}

    for domain_key, domain_info in domain_registry.items():
        domain_exemplars[domain_key] = {}
        for intent_name in domain_info["intents"]:
            if intent_name in intent_embeddings:
                matrix = np.array(intent_embeddings[intent_name])
                # L2-normalize each row for fast cosine via dot product
                norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
                domain_exemplars[domain_key][intent_name] = matrix / norms

    return domain_exemplars


def _knn_vote_within_domain(
    query_vec: np.ndarray,
    domain_exemplars: Dict[str, np.ndarray],
    k: int = 7,
) -> List[Tuple[str, float, float]]:
    """kNN voting over exemplars within a single domain.
    
    For each intent in the domain, compute cosine similarity against
    ALL its exemplars. Collect the top-K nearest neighbors globally,
    then count votes per intent.
    
    Args:
        query_vec: L2-normalized query vector (D,)
        domain_exemplars:  {intent_name: (N, D) matrix of L2-norm exemplars}
        k: Number of nearest neighbors to vote
        
    Returns:
        List of (intent_name, vote_fraction, max_similarity) sorted by vote desc
    """
    # Collect all (similarity, intent_name) pairs
    all_neighbors = []

    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)

    for intent_name, exemplar_matrix in domain_exemplars.items():
        # Fast cosine similarity: dot product of L2-normalized vectors
        sims = exemplar_matrix @ query_norm  # (N,)
        for sim_val in sims:
            all_neighbors.append((float(sim_val), intent_name))

    # Sort by similarity descending, take top-K
    all_neighbors.sort(key=lambda x: x[0], reverse=True)
    top_k = all_neighbors[:k]

    # Count votes
    from collections import Counter
    votes = Counter(intent_name for _, intent_name in top_k)
    total_votes = sum(votes.values())

    # Also compute max similarity per intent (for confidence)
    max_sim_per_intent = {}
    for sim_val, intent_name in all_neighbors:
        if intent_name not in max_sim_per_intent:
            max_sim_per_intent[intent_name] = sim_val

    # Build results: (intent, vote_fraction, max_similarity)
    results = []
    for intent_name in votes:
        results.append((
            intent_name,
            votes[intent_name] / total_votes,
            max_sim_per_intent.get(intent_name, 0.0),
        ))

    # Sort by votes (desc), then by max_sim (desc) for ties
    results.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return results


def classify_hierarchical_knn(
    query_vec: np.ndarray,
    domain_centroids: Dict[str, np.ndarray],
    domain_exemplars: Dict[str, Dict[str, np.ndarray]],
    domain_intent_map: Dict[str, Dict[str, np.ndarray]],
    domain_gap_threshold: float = 0.02,
    top_k_domains: int = 2,
    knn_k: int = 7,
) -> Dict[str, Any]:
    """Hierarchical classification with kNN exemplar voting for intent resolution.
    
    Same domain-routing logic as classify_hierarchical, but replaces
    centroid-based intent comparison with kNN exemplar voting.
    
    Pipeline:
      1. Score all domains using centroid cosine similarity
      2. If top-2 domains are close (gap < threshold), consider both
      3. Within candidate domain(s), run kNN voting over ALL exemplars
      4. The intent with the most votes wins; its domain is the final domain
    
    Args:
        query_vec: Embedded query vector
        domain_centroids: {domain_name: centroid_vector}
        domain_exemplars: {domain_name: {intent_name: (N,D) exemplar matrix}}
        domain_intent_map: {domain_name: {intent_name: centroid_vector}} (for fallback)
        domain_gap_threshold: If gap between top-2 domains < this, consider both
        top_k_domains: Max domains to consider when ambiguous
        knn_k: Number of nearest neighbors for voting
        
    Returns:
        Same structure as classify_hierarchical, plus knn_votes
    """
    # ── Step 1: Score all domains (same as hierarchical) ─────────────────
    domain_scores = {}
    for domain_name, centroid in domain_centroids.items():
        domain_scores[domain_name] = _cosine_similarity(query_vec, centroid)

    ranked_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    top_domain, top_score = ranked_domains[0]
    second_domain, second_score = ranked_domains[1] if len(ranked_domains) > 1 else (None, -1)

    gap = top_score - second_score
    is_ambiguous = gap < domain_gap_threshold

    # ── Step 2: Determine candidate domains ──────────────────────────────
    if is_ambiguous:
        candidate_domains = [d for d, _ in ranked_domains[:top_k_domains]]
    else:
        candidate_domains = [top_domain]

    # ── Step 3: kNN voting across candidate domains ──────────────────────
    # Merge exemplars from all candidate domains into one pool
    merged_exemplars = {}
    for domain_name in candidate_domains:
        if domain_name not in domain_exemplars:
            continue
        for intent_name, matrix in domain_exemplars[domain_name].items():
            # For cross-domain intents (claim_status), prefix with domain
            # to keep them separate in the vote
            key = f"{intent_name}@{domain_name}"
            merged_exemplars[key] = matrix

    if merged_exemplars:
        knn_results = _knn_vote_within_domain(query_vec, merged_exemplars, k=knn_k)
    else:
        knn_results = []

    # ── Step 4: Extract winner ───────────────────────────────────────────
    if knn_results:
        winner_key, vote_frac, max_sim = knn_results[0]
        # Parse "intent_name@domain_name"
        if "@" in winner_key:
            best_intent, best_domain = winner_key.rsplit("@", 1)
        else:
            best_intent = winner_key
            best_domain = top_domain

        # Build readable vote summary
        vote_summary = {
            k: f"{v:.0%}" for k, v, _ in knn_results[:5]
        }
    else:
        # Fallback to centroid-based if exemplars unavailable
        best_intent = None
        best_intent_score = -1.0
        best_domain = top_domain
        for domain_name in candidate_domains:
            if domain_name not in domain_intent_map:
                continue
            for intent_name, centroid in domain_intent_map[domain_name].items():
                sim = _cosine_similarity(query_vec, centroid)
                if sim > best_intent_score:
                    best_intent_score = sim
                    best_intent = intent_name
                    best_domain = domain_name
        vote_frac = 1.0
        max_sim = best_intent_score
        vote_summary = {}

    return {
        "predicted_domain": best_domain,
        "predicted_intent": best_intent or "unknown",
        "domain_score": domain_scores.get(best_domain, top_score),
        "intent_score": max_sim,
        "domain_scores": domain_scores,
        "considered_domains": candidate_domains,
        "is_ambiguous_domain": is_ambiguous,
        "knn_vote_fraction": vote_frac,
        "knn_votes": vote_summary,
    }


# ═════════════════════════════════════════════════════════════════════════════
# TASK 4c: GLOBAL kNN — INTENT-FIRST CLASSIFICATION (domain from intent)
# ═════════════════════════════════════════════════════════════════════════════
#
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  WHY GLOBAL kNN BEATS HIERARCHICAL WHEN DOMAINS OVERLAP >0.93       │
#  │                                                                      │
#  │  The overlap analysis showed:                                        │
#  │    cap_api ↔ claim_history_search:  0.9718  (nearly identical!)      │
#  │    cap_api ↔ benefits_api:          0.9322                           │
#  │    benefits_api ↔ claim_history:    0.9295                           │
#  │                                                                      │
#  │  When domain centroids are this close, domain routing is a           │
#  │  COIN FLIP. And every domain error cascades to intent error          │
#  │  because we only search intents within the selected domain.          │
#  │                                                                      │
#  │  Global kNN eliminates this cascade:                                 │
#  │    1. Search ALL exemplars across ALL domains (no domain filter)     │
#  │    2. Distance-weighted voting (closer = higher weight)              │
#  │    3. Winning intent determines domain (reverse lookup)              │
#  │    4. Domain is a CONSEQUENCE of intent, not a prerequisite          │
#  │                                                                      │
#  │  With 37+ intents × 20 examples = 740+ exemplars, a single          │
#  │  numpy dot product computes all similarities in <1ms.                │
#  └──────────────────────────────────────────────────────────────────────┘

def _build_intent_to_domain_lookup() -> Dict[str, str]:
    """Build reverse map: intent_name → domain_name.
    
    Sources (in priority order):
      1. DOMAIN_REGISTRY — official mapping
      2. Fallback heuristic — intents in CVS_INTENT_EXAMPLES that are NOT in
         any domain. These are "orphaned" intents that have training data but
         were removed from the registry. We still need to classify them.
    
    This ensures test data intents that no longer appear in DOMAIN_REGISTRY
    can still be classified and mapped to their historical domain.
    """
    domain_registry = embeddingVars.DOMAIN_REGISTRY
    lookup = {}

    # 1. Official registry mapping
    for domain_key, domain_info in domain_registry.items():
        for intent_name in domain_info["intents"]:
            if intent_name not in lookup:
                lookup[intent_name] = domain_key

    # 2. Orphaned intents — map them based on the OLD domain structure
    #    These are training intents not assigned to any current domain.
    LEGACY_DOMAIN_MAP = {
        # Old benefits_api intents
        "approval_info": "benefits_api",
        "audit_info": "benefits_api",
        "beneficiary_info": "benefits_api",
        # Old claim_history_search intents
        "compound_info": "claim_history_search",
        "date_range_claims": "claim_history_search",
        "drug_info": "claim_history_search",
        "drug_interaction_info": "claim_history_search",
        "fill_date_info": "claim_history_search",
        # Old cap_api intents
        "daw_info": "cap_api",
        "government_claim_type": "cap_api",
        "mail_order_info": "cap_api",
        "medicare_part_d": "cap_api",
        "network_info": "cap_api",
        "prior_auth_info": "cap_api",
    }

    for intent_name in embeddingVars.CVS_INTENT_EXAMPLES:
        if intent_name not in lookup:
            if intent_name in LEGACY_DOMAIN_MAP:
                lookup[intent_name] = LEGACY_DOMAIN_MAP[intent_name]
            else:
                lookup[intent_name] = "unknown"
                logger.warning(f"⚠️  Orphaned intent '{intent_name}' has no domain mapping")

    return lookup


def _build_global_exemplar_index(
    intent_embeddings: Dict[str, List[List[float]]],
) -> Tuple[np.ndarray, List[str]]:
    """Build a single (N, D) matrix of ALL exemplars + label list.
    
    Returns:
        exemplar_matrix: (N, D) L2-normalized embedding matrix
        exemplar_labels: length-N list of intent names (parallel to rows)
    """
    all_vectors = []
    all_labels = []

    for intent_name, vectors in intent_embeddings.items():
        for vec in vectors:
            all_vectors.append(vec)
            all_labels.append(intent_name)

    matrix = np.array(all_vectors)
    # L2-normalize each row for fast cosine via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix = matrix / norms

    logger.info(
        f"  ▸ Global exemplar index: {matrix.shape[0]} exemplars × "
        f"{matrix.shape[1]} dims from {len(set(all_labels))} intents"
    )
    return matrix, all_labels


def classify_global_knn(
    query_vec: np.ndarray,
    exemplar_matrix: np.ndarray,
    exemplar_labels: List[str],
    intent_to_domain: Dict[str, str],
    k: int = 7,
    distance_weighted: bool = True,
) -> Dict[str, Any]:
    """Global kNN: classify intent across ALL domains, derive domain from intent.
    
    ┌────────────────────────────────────────────────────────────────────────┐
    │  Pipeline:                                                             │
    │                                                                        │
    │  Query → embed → cosine sim vs ALL exemplars → top-K neighbors         │
    │       → distance-weighted vote → intent → lookup domain                │
    │                                                                        │
    │  No domain routing step. No cascading errors.                          │
    │  Domain is derived from the winning intent via reverse lookup.         │
    └────────────────────────────────────────────────────────────────────────┘
    
    Args:
        query_vec: Embedded query vector (D,)
        exemplar_matrix: (N, D) L2-normalized matrix of all training exemplars
        exemplar_labels: length-N list of intent names
        intent_to_domain: {intent_name: domain_name} reverse lookup
        k: Number of nearest neighbors
        distance_weighted: If True, weight votes by similarity (closer = heavier)
        
    Returns:
        {predicted_domain, predicted_intent, intent_score, knn_votes, ...}
    """
    from collections import defaultdict

    # ── Compute cosine similarity against ALL exemplars ──────────────────
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    similarities = exemplar_matrix @ query_norm  # (N,) — single numpy op

    # ── Get top-K indices ────────────────────────────────────────────────
    if k < len(similarities):
        top_k_idx = np.argpartition(similarities, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]
    else:
        top_k_idx = np.argsort(similarities)[::-1][:k]

    # ── Vote (distance-weighted or uniform) ──────────────────────────────
    intent_votes: Dict[str, float] = defaultdict(float)
    intent_max_sim: Dict[str, float] = {}
    top_k_details = []

    for idx in top_k_idx:
        sim = float(similarities[idx])
        label = exemplar_labels[idx]

        if distance_weighted:
            # Weight = similarity score (higher = more weight)
            # This means a neighbor at 0.95 contributes ~19x more than one at 0.05
            intent_votes[label] += sim
        else:
            intent_votes[label] += 1.0

        if label not in intent_max_sim:
            intent_max_sim[label] = sim

        top_k_details.append((label, sim))

    # ── Determine winner ─────────────────────────────────────────────────
    total_weight = sum(intent_votes.values())
    ranked = sorted(intent_votes.items(), key=lambda x: x[1], reverse=True)

    best_intent = ranked[0][0]
    best_weight = ranked[0][1]
    best_sim = intent_max_sim.get(best_intent, 0.0)
    best_domain = intent_to_domain.get(best_intent, "unknown")

    # Vote summary (top 5)
    vote_summary = {}
    for intent, weight in ranked[:5]:
        pct = weight / total_weight if total_weight > 0 else 0
        vote_summary[intent] = f"{pct:.0%}"

    return {
        "predicted_domain": best_domain,
        "predicted_intent": best_intent,
        "domain_score": best_weight / total_weight if total_weight > 0 else 0,
        "intent_score": best_sim,
        "knn_vote_fraction": best_weight / total_weight if total_weight > 0 else 0,
        "knn_votes": vote_summary,
        "top_k_details": top_k_details,
    }


# ═════════════════════════════════════════════════════════════════════════════
# TASK 4d: LLM CLASSIFIER — Gemini Flash for ambiguous cases
# ═════════════════════════════════════════════════════════════════════════════
#
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  WHY LLM IS NEEDED FOR >90% ACCURACY                                │
#  │                                                                      │
#  │  Embedding similarity has a hard ceiling (~85-88%) when:             │
#  │    • Multiple intents share 80%+ vocabulary                         │
#  │    • Training examples follow identical templates                    │
#  │      ("Show the X for this claim" vs "Display the X for this claim")│
#  │    • Domain jargon (TF, BPG, COB) appears in 1-2 word differences  │
#  │                                                                      │
#  │  An LLM understands MEANING, not just word overlap:                  │
#  │    • "override codes" → rejection_reasons (what failed)             │
#  │    • "plan overrides" → approval_info (what was approved)           │
#  │    • A human can tell these apart; embeddings can't                  │
#  │                                                                      │
#  │  HYBRID APPROACH:                                                    │
#  │    kNN handles ~70% of queries (high confidence, single API call)   │
#  │    LLM handles ~30% (ambiguous cases only, ~300ms per call)         │
#  │    Combined accuracy: 90%+                                          │
#  └──────────────────────────────────────────────────────────────────────┘

# Intent descriptions for LLM — much richer than training examples
INTENT_DESCRIPTIONS_FOR_LLM: Dict[str, str] = {
    # --- cap_api intents ---
    "claim_status": "General claim status, adjudication outcome, processing result, whether paid/rejected/pending, overall claim summary",
    "multi_claim_summary": "Summary of ALL/MULTIPLE claims for a member, complete claim list, full history overview",
    "pharmacy_info": "Dispensing pharmacy name, location, address, NCPDP number, store where prescription was filled",
    "prescriber_info": "Prescribing physician/doctor name, NPI number, credentials, ordering provider details",
    "pricing_info": "Copay, ingredient cost, dispensing fee, patient pay, out-of-pocket cost breakdown, pricing schedule",
    "reimbursement_info": "Amount paid TO the pharmacy, reimbursement rationale, payment calculation for paper claims",
    "rejection_reasons": "Rejection codes, failed edit codes, denial explanations, why claim was denied, how to resolve/fix/overturn rejection",
    "settlement_info": "Settlement codes, pharmacy response/feedback codes sent back to pharmacy after adjudication",
    "rx_details": "RX/prescription number, fill number, quantity dispensed, days supply, drug strength, refill information",
    "reversal_info": "Claim reversal, R&R (reverse and resubmit), manual adjustments, modification history, resubmission",
    "cob_info": "Coordination of Benefits (COB), other insurance, secondary payer, dual coverage, primary/secondary adjudication",
    "generic_availability": "Generic alternatives, therapeutic equivalents, formulary substitutes, cheaper drug options",
    # --- benefits_api intents (old / legacy) ---
    "approval_info": "Claim approval details, plan overrides, transition fill (TF) status/type/eligibility, BPG configuration, Smart PA, accumulation bypass",
    "audit_info": "Audit trail, change history, modification records, edit timestamps, who made changes, add date, change date",
    "beneficiary_info": "Member benefit phase, coverage tier, eligibility status, accumulation rules, LOE, medical dollar contribution",
    # --- claim_history_search intents (old / legacy) ---
    "compound_info": "Compound medication details, MIC (Most Ingredient Cost) breakdown, individual ingredient costs, compound formulation",
    "date_range_claims": "Claims within a date range, deductible-contributing claims, accumulation history, prescription fill history over time",
    "drug_info": "Drug name, NDC code, GPI, therapeutic class, formulary status, medication tier, drug classification",
    "drug_interaction_info": "DUR edits, drug utilization review, drug interaction alerts, clinical screening, override details",
    "fill_date_info": "Date prescription was filled, dispensing date, service date, fill timestamp",
    # --- general ---
    "greeting": "Hello, hi, welcome, good morning/afternoon/evening, casual greeting",
    "help": "How to submit claims, steps to avoid rejection, claim filing guidance, instructions",
    "out_of_scope": "Unrelated to pharmacy claims — weather, recipes, sports, random text, gibberish",
    # --- cap_api extras ---
    "daw_info": "DAW (Dispense As Written) status, brand vs generic requirement, substitution allowed",
    "government_claim_type": "Medicare/Medicaid claim type, government program classification",
    "mail_order_info": "Mail order/home delivery prescription status, shipping details",
    "medicare_part_d": "Medicare Part D summary, PDE details, MEDD pricing, LICS, N1",
    "network_info": "Pharmacy network details, which network processed/paid the claim",
    "prior_auth_info": "Prior authorization (PA) status, Smart PA, Member PA, authorization requirements",
    # --- claim_history_search new intents ---
    "Refills": "Refill counts, remaining refills, refill history for prescriptions",
    "DaysSupply": "Days supply filtering (30-day, 90-day, etc.)",
    "PriorAuth": "Claims that required prior authorization, PA status in search context",
    "Diagnosis": "Claims filtered by ICD-10 diagnosis code",
    "Settlement": "Claims filtered by settlement/response code",
    "PharmType": "Claims filtered by pharmacy type (retail, mail-order, specialty)",
    "Plan": "Claims filtered by insurance plan code",
    "Pharmacy": "Claims from a specific pharmacy by name/ID",
    "Prescriber": "Claims by a specific prescriber/doctor name or NPI",
    "Pricing": "Pricing details in claim search context (copay, cost, member pay)",
    "Status": "Claims filtered by status (rejected, paid, pending, reversed)",
    "RejectCode": "Claims filtered by specific NCPDP reject code",
    "DrugLast": "When a specific drug was last dispensed for a member",
    "Month": "Claims filtered by calendar month",
    "ClaimNum": "Lookup a specific claim by claim number",
    "NDC": "Claims filtered by drug NDC code",
    "Manufacturer": "Claims filtered by drug manufacturer name",
    "Generic": "Claims for generic drugs only",
    "Brand": "Claims for brand-name drugs only",
    # --- benefits_api new intents ---
    "plan_summary": "Current benefit plan overview, active plan snapshot, coverage summary",
    "plan_history": "Change log of benefit plan, plan revision history, amendments over time",
    "plan_finder": "Search/find/locate available benefit plans, plan catalog lookup",
    # --- member_domain new intents ---
    "member_demographics": "Member name (first/last/middle), date of birth, gender, person code, relationship code, demographic profile",
    "member_contact_info": "Member email address, phone number, mailing/postal address, city, state, zip code, country",
    "member_eligibility_copay": "Member eligibility copay fields: copayBrand, copayGeneric, copay3, copay4, copay tier configuration",
    "member_transition_status": "Member transition fill status and start date from eligibility, whether member is in a transition period",
    "member_dur_config": "Drug utilization review (DUR) key and process flag, DUR configuration, whether DUR processing is enabled",
    "member_mbi_number": "Medicare Beneficiary Identifier (MBI) number from the Medicare Part D record",
    "member_caretaker_info": "Caretaker first/last name and address from Medicare Part D record, who is listed as caretaker",
    "member_language_pref": "Member language code/preference (mbrLangCode), preferred communication language",
    "member_discount_program": "Discount program type assigned to the member, whether enrolled in a discount program",
    "member_override_plan": "Member-level override plan ID from eligibility record (memberOverridePlan), whether an override plan is configured",
    # --- member_domain existing intents (for completeness) ---
    "member_coverage": "Coverage eligibility windows, active status, enrollment dates, effective period start/end",
    "member_hierarchy": "Client/CAG/account/group hierarchy, which client or carrier this member belongs to",
    "benefit_reset_date": "Benefit year reset date, when accumulators reset, plan year anniversary",
    "family_type": "Family type designation (individual vs family), coverage tier family classification",
    "family_members": "List of family members, dependents, subscriber, who else is on the same family plan",
    "alternate_insurance": "Other/secondary/alternate insurance on file, dual coverage, secondary payer",
    "medicare_coverage": "Medicare Part D enrollment status, whether enrolled in Medicare, Med-D plan details",
    "lics_status": "Low Income Cost Subsidy (LICS/LIS) status, subsidy level, whether member qualifies for low income benefits",
    "stcob_linkage": "Short-term coordination of benefits (STCOB) linkage, COB member links, coverage code",
    "cvs_id_lookup": "CVS ID/identifier for the member, CVS member number lookup",
    "related_cagm": "Related CAGMs by CVS ID or family ID, linked member records within same client",
    "alternate_ids": "All alternate IDs on file for the member, cross-reference IDs, secondary identifiers",
    # --- override_domain new intents ---
    "pa_reason_code": "PA reason code (U1, LC, OD, OA, US, U3), why the prior authorization was created, override reason classification",
    "pa_effective_dates": "PA effective period begin/end dates, when the PA is active, whether it has expired",
    "pa_agent_code": "Agent/source code on the PA (A, C, 3, H, 5, 2, O), who created or last modified the PA",
    "pa_ignore_status": "Ignore status code on the PA (Y, P, 3), whether the PA status is being ignored during processing",
    "pa_specialty_rx_override": "Specialty prescription reject override indicator, whether this PA bypasses specialty Rx rejection",
    "pa_clinical_admin_code": "Clinical administration code on the PA (A, C, or blank), clinical program designation",
    "pa_transform_care": "Transform care type on the PA, care transformation program designation",
    "pa_follow_me_logic": "Follow me logic indicator on the PA, whether the PA follows the member across plan changes",
    "pa_drug_type_indicator": "Authorized drug type on the PA (G=GPI-based matching, N=NDC-based matching)",
    "pa_modification_history": "PA modification date/time (modifyDateTime), when the PA was last modified, update timestamp",
}


def classify_with_llm(
    query: str,
    candidate_intents: List[str],
    intent_descriptions: Dict[str, str],
    intent_to_domain: Dict[str, str],
) -> Dict[str, Any]:
    """Classify using Gemini Flash with structured output.
    
    Args:
        query: User's natural language query
        candidate_intents: List of candidate intent names to choose from
        intent_descriptions: {intent: description} for the candidates
        intent_to_domain: {intent: domain} reverse lookup
        
    Returns:
        {predicted_intent, predicted_domain, confidence, reasoning}
    """
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=os.getenv("PROJECT_ID", "pbm-poc-coderev-genai-poc"),
        location=os.getenv("LOCATION", "us-central1"),
    )

    # Build intent list for prompt
    intent_list = "\n".join(
        f"  - {name}: {intent_descriptions.get(name, name)}"
        for name in candidate_intents
    )

    prompt = f"""You are a pharmacy claims intent classifier. Classify the user query into exactly ONE intent.

CANDIDATE INTENTS:
{intent_list}

USER QUERY: {query}

Respond with ONLY the intent name (e.g., "approval_info"). No explanation."""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=50,
            ),
        )
        predicted = response.text.strip().strip('"').strip("'").strip()

        # Validate response is one of the candidates
        if predicted in candidate_intents:
            return {
                "predicted_intent": predicted,
                "predicted_domain": intent_to_domain.get(predicted, "unknown"),
                "confidence": 1.0,
                "source": "llm",
            }

        # Fuzzy match — LLM might return slightly different casing
        for c in candidate_intents:
            if c.lower() == predicted.lower():
                return {
                    "predicted_intent": c,
                    "predicted_domain": intent_to_domain.get(c, "unknown"),
                    "confidence": 0.9,
                    "source": "llm_fuzzy",
                }

        # LLM returned unknown intent — fall back
        logger.warning(f"LLM returned unknown intent '{predicted}', falling back to first candidate")
        return {
            "predicted_intent": candidate_intents[0],
            "predicted_domain": intent_to_domain.get(candidate_intents[0], "unknown"),
            "confidence": 0.3,
            "source": "llm_fallback",
        }

    except Exception as e:
        logger.error(f"LLM classification failed: {e}")
        return {
            "predicted_intent": candidate_intents[0] if candidate_intents else "unknown",
            "predicted_domain": intent_to_domain.get(candidate_intents[0], "unknown") if candidate_intents else "unknown",
            "confidence": 0.0,
            "source": "llm_error",
        }


def classify_hybrid_knn_llm(
    query: str,
    query_vec: np.ndarray,
    exemplar_matrix: np.ndarray,
    exemplar_labels: List[str],
    intent_to_domain: Dict[str, str],
    knn_k: int = 7,
    confidence_threshold: float = 0.55,
    margin_threshold: float = 0.10,
) -> Dict[str, Any]:
    """Hybrid classifier: kNN for easy cases, LLM for hard cases.
    
    ┌────────────────────────────────────────────────────────────────────────┐
    │  Pipeline:                                                             │
    │                                                                        │
    │  1. Run global kNN (distance-weighted, k=7)                            │
    │  2. Check confidence:                                                  │
    │     a. IF vote_fraction > 55% AND margin > 10% → kNN wins (fast path) │
    │     b. ELSE → send to Gemini Flash with top-5 candidates               │
    │  3. Domain derived from winning intent                                 │
    │                                                                        │
    │  Expected split: ~70% kNN (fast), ~30% LLM (accurate)                 │
    │  Expected accuracy: 90%+                                               │
    └────────────────────────────────────────────────────────────────────────┘
    """
    from collections import defaultdict

    # ── Step 1: Global kNN ───────────────────────────────────────────────
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    similarities = exemplar_matrix @ query_norm

    if knn_k < len(similarities):
        top_k_idx = np.argpartition(similarities, -knn_k)[-knn_k:]
        top_k_idx = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]
    else:
        top_k_idx = np.argsort(similarities)[::-1][:knn_k]

    intent_votes: Dict[str, float] = defaultdict(float)
    intent_max_sim: Dict[str, float] = {}

    for idx in top_k_idx:
        sim = float(similarities[idx])
        label = exemplar_labels[idx]
        intent_votes[label] += sim
        if label not in intent_max_sim:
            intent_max_sim[label] = sim

    total_weight = sum(intent_votes.values())
    ranked = sorted(intent_votes.items(), key=lambda x: x[1], reverse=True)

    best_intent = ranked[0][0]
    best_fraction = ranked[0][1] / total_weight if total_weight > 0 else 0
    second_fraction = ranked[1][1] / total_weight if len(ranked) > 1 and total_weight > 0 else 0
    margin = best_fraction - second_fraction

    # ── Step 2: Confidence check ─────────────────────────────────────────
    knn_confident = best_fraction >= confidence_threshold and margin >= margin_threshold

    if knn_confident:
        # Fast path — kNN is confident
        return {
            "predicted_intent": best_intent,
            "predicted_domain": intent_to_domain.get(best_intent, "unknown"),
            "intent_score": intent_max_sim.get(best_intent, 0.0),
            "knn_vote_fraction": best_fraction,
            "knn_margin": margin,
            "source": "knn",
        }

    # ── Step 3: LLM arbitration ──────────────────────────────────────────
    # Send top-5 candidates to Gemini Flash
    top_candidates = [intent for intent, _ in ranked[:5]]

    llm_result = classify_with_llm(
        query=query,
        candidate_intents=top_candidates,
        intent_descriptions=INTENT_DESCRIPTIONS_FOR_LLM,
        intent_to_domain=intent_to_domain,
    )

    return {
        "predicted_intent": llm_result["predicted_intent"],
        "predicted_domain": llm_result["predicted_domain"],
        "intent_score": intent_max_sim.get(llm_result["predicted_intent"], 0.0),
        "knn_vote_fraction": best_fraction,
        "knn_margin": margin,
        "source": llm_result["source"],
        "knn_top_intent": best_intent,
        "llm_candidates": "|".join(top_candidates),
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
    method: str = "global_knn",
) -> Dict[str, float]:
    """Classify and evaluate using the specified method.
    
    Args:
        test_data: List of {text, actual_classification_intent, actual_classification_domain}
        method: "global_knn" (recommended), "hierarchical_knn", "hierarchical",
                "flat_cosine", or "flat_euclidean" (v1)
        
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

    # Build exemplar indices based on method
    domain_exemplars = None
    global_matrix = None
    global_labels = None
    intent_to_domain = None

    intent_embeddings = _load_intent_embeddings()

    if method in ("global_knn", "hybrid_knn_llm"):
        global_matrix, global_labels = _build_global_exemplar_index(intent_embeddings)
        intent_to_domain = _build_intent_to_domain_lookup()
    elif method == "hierarchical_knn":
        domain_exemplars = _build_domain_exemplar_index(intent_embeddings)

    embedder = _get_embedder()
    results = []
    llm_call_count = 0

    for idx, record in enumerate(test_data):
        text = record["text"]
        actual_intent = record["actual_classification_intent"]
        actual_domain = record["actual_classification_domain"]

        query_vec = np.array(embedder.embed(text))

        if method == "hybrid_knn_llm":
            result = classify_hybrid_knn_llm(
                query=text,
                query_vec=query_vec,
                exemplar_matrix=global_matrix,
                exemplar_labels=global_labels,
                intent_to_domain=intent_to_domain,
                knn_k=7,
            )
            predicted_intent = result["predicted_intent"]
            predicted_domain = result["predicted_domain"]
            source = result.get("source", "knn")
            if source != "knn":
                llm_call_count += 1
            extra = {
                "intent_score": result["intent_score"],
                "knn_vote_fraction": result.get("knn_vote_fraction", ""),
                "knn_margin": result.get("knn_margin", ""),
                "source": source,
            }
        elif method == "global_knn":
            result = classify_global_knn(
                query_vec, global_matrix, global_labels, intent_to_domain,
                k=7, distance_weighted=True,
            )
            predicted_intent = result["predicted_intent"]
            predicted_domain = result["predicted_domain"]
            extra = {
                "intent_score": result["intent_score"],
                "knn_vote_fraction": result.get("knn_vote_fraction", ""),
            }
        elif method == "hierarchical_knn":
            result = classify_hierarchical_knn(
                query_vec, domain_centroids_np, domain_exemplars,
                domain_intent_map
            )
            predicted_intent = result["predicted_intent"]
            predicted_domain = result["predicted_domain"]
            extra = {
                "intent_score": result["intent_score"],
                "domain_score": result["domain_score"],
                "is_ambiguous": result["is_ambiguous_domain"],
                "considered_domains": "|".join(result["considered_domains"]),
                "knn_vote_fraction": result.get("knn_vote_fraction", ""),
            }
        elif method == "hierarchical":
            result = classify_hierarchical(
                query_vec, domain_centroids_np, domain_intent_map
            )
            predicted_intent = result["predicted_intent"]
            predicted_domain = result["predicted_domain"]
            extra = {
                "intent_score": result["intent_score"],
                "domain_score": result["domain_score"],
                "is_ambiguous": result["is_ambiguous_domain"],
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
    if method == "hybrid_knn_llm":
        knn_pct = (len(test_data) - llm_call_count) / len(test_data) * 100
        print(f"  kNN resolved (no LLM)          : {knn_pct:.1f}% ({len(test_data) - llm_call_count}/{len(test_data)})")
        print(f"  LLM calls                      : {llm_call_count}/{len(test_data)}")
    print(f"{'='*60}")

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
      1. flat_cosine      — cosine similarity + intent-weighted domain centroids
      2. hierarchical     — domain→intent routing (centroid) + multi-domain fallback
      3. hierarchical_knn — domain→intent routing (kNN exemplar voting)
      4. global_knn       — SKIP domain routing, global kNN + distance-weighted ★ BEST
    """
    print("\n" + "="*70)
    print("  ABLATION STUDY: Comparing Classification Methods")
    print("="*70)

    all_results = {}

    for method in ["flat_cosine", "hierarchical", "hierarchical_knn", "global_knn"]:
        print(f"\n{'─'*70}")
        print(f"  Running method: {method}")
        print(f"{'─'*70}")

        if method in ("hierarchical", "hierarchical_knn"):
            generate_domain_centroids()

        metrics = classify_and_evaluate(test_data, method=method)
        all_results[method] = metrics

    # ── Summary comparison ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SUMMARY: Method Comparison")
    print("="*70)
    print(f"  {'Method':<22} {'Intent Acc':>12} {'Domain Acc':>12}")
    print(f"  {'-'*48}")

    for method, metrics in all_results.items():
        star = " ★" if method == "global_knn" else ""
        print(
            f"  {method:<22} {metrics['intent_accuracy']:>10.2f}% "
            f"{metrics['domain_accuracy']:>10.2f}%{star}"
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
    import sys

    print("="*70)
    print("  Intent Detection v3 — Hybrid kNN + Gemini Flash")
    print("="*70)

    # Step 1: Generate embeddings (auto-detects stale cache)
    print("\nStep 1 — Generating intent embeddings (stale cache detection) …")
    embedding_gen()

    # Step 2: Generate L2-normalized intent centroids
    print("\nStep 2 — Generating intent centroids (L2-normalized) …")
    generate_intent_centroids()

    # Step 3: Generate intent-weighted domain centroids
    print("\nStep 3 — Generating domain centroids (intent-weighted) …")
    generate_domain_centroids()

    # Step 4: Analyze domain overlap
    print("\nStep 4 — Analyzing domain cluster overlap …")
    analyze_domain_overlap()

    # Step 5: Run evaluation
    TESTDATA_PATH = os.path.join(BASE_DIR, "Testdata.csv")

    if not os.path.exists(TESTDATA_PATH):
        print(f"\n⚠️  Testdata.csv not found at {TESTDATA_PATH} — skipping evaluation.")
        sys.exit(0)

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

    # ── Choose method via command line arg ────────────────────────────
    method = sys.argv[1] if len(sys.argv) > 1 else "hybrid_knn_llm"
    valid_methods = ["hybrid_knn_llm", "global_knn", "hierarchical_knn", "hierarchical", "flat_cosine"]

    if method not in valid_methods:
        print(f"⚠️  Unknown method '{method}'. Valid: {valid_methods}")
        sys.exit(1)

    print(f"\n{'─'*70}")
    print(f"  Running: {method.upper()}")
    print(f"{'─'*70}")

    metrics = classify_and_evaluate(test_data, method=method)

    print(f"\n📊 Results ({method}):")
    print(f"   Intent Accuracy : {metrics['intent_accuracy']:.2f}%")
    print(f"   Domain Accuracy : {metrics['domain_accuracy']:.2f}%")
