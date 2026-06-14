"""
Intent Detection via Sentence Embeddings & Centroid Classification
Uses Google Cloud Vertex AI text-embedding-005 model (same pattern as pss-myclaims-ai-agent)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Union

from VamsiSir import embeddingVars

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# These are the paths for the various artifacts generated during the process.
INTENT_EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "intent_embeddings.json")
INTENT_CENTROIDS_PATH = os.path.join(ARTIFACTS_DIR, "intent_centroids.json")
DOMAIN_CENTROIDS_PATH = os.path.join(ARTIFACTS_DIR, "domain_centroids.json")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─── Vertex AI Embedding Client (mirrors pss-myclaims-ai-agent) ─────────────

class VertexEmbeddings:
    """Google Cloud Vertex AI Embeddings using text-embedding-005.
    
    Replicates the same SDK usage and auth pattern from
    pss-myclaims-ai-agent/services/google_embeddings.py
    """

    def __init__(self, project_id: str = None, location: str = "us-central1"):
        """Initialise the Google GenAI client for Vertex AI embeddings.

        Args:
            project_id: GCP project ID. Falls back to env var PROJECT_ID,
                        then to the default used in pss-myclaims-ai-agent.
            location:   GCP region. Falls back to env var LOCATION.
        """
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

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embedding(s) for one or more texts.

        Args:
            text: A single string or list of strings.

        Returns:
            A single embedding vector (list[float]) or list of vectors.
        """
        from google.genai import types

        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        embeddings_list = []
        for t in texts:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[types.Part.from_text(text=t)],
            )
            embeddings_list.append(response.embeddings[0].values)

        return embeddings_list[0] if is_single else embeddings_list


# Module-level singleton
_embedder: VertexEmbeddings = None


def _get_embedder() -> VertexEmbeddings:
    """Return (and lazily create) the singleton VertexEmbeddings instance."""
    global _embedder
    if _embedder is None:
        _embedder = VertexEmbeddings()
    return _embedder


# ─── Task 1: Generate & Cache Intent Embeddings ────────────────────────────

def embedding_gen() -> Dict[str, List[List[float]]]:
    """Generate and cache sentence embeddings for every intent example.

    Iterates over ``CVS_INTENT_EXAMPLES`` from VamsiSir.py, generates a
    Vertex AI embedding for each sentence, and persists the result to
    ``artifacts/intent_embeddings.json``.

    If the JSON file already exists the function returns the cached data
    without calling the embedding API.

    Returns:
        dict mapping intent names to lists of embedding vectors.
    """
    if os.path.exists(INTENT_EMBEDDINGS_PATH):
        logger.info(f"✅ Cache hit — loading intent embeddings from {INTENT_EMBEDDINGS_PATH}")
        with open(INTENT_EMBEDDINGS_PATH, "r") as f:
            return json.load(f)

    logger.info("Generating intent embeddings via Vertex AI …")
    embedder = _get_embedder()
    intent_examples = embeddingVars.CVS_INTENT_EXAMPLES

    intent_embeddings: Dict[str, List[List[float]]] = {}

    for intent_name, sentences in intent_examples.items():
        logger.info(f"  ▸ Embedding {len(sentences)} sentences for intent '{intent_name}'")
        vectors = embedder.embed(sentences)  # batch call
        # Ensure plain lists (not numpy arrays) for JSON serialisation
        intent_embeddings[intent_name] = [
            list(v) if not isinstance(v, list) else v for v in vectors
        ]

    with open(INTENT_EMBEDDINGS_PATH, "w") as f:
        json.dump(intent_embeddings, f)
    logger.info(f"✅ Intent embeddings saved → {INTENT_EMBEDDINGS_PATH}")

    return intent_embeddings


# ─── Task 2: Generate Intent Centroids ──────────────────────────────────────

def generate_intent_centroids() -> Dict[str, List[float]]:
    """Compute and cache the centroid (mean vector) per intent.

    Loads embeddings from ``artifacts/intent_embeddings.json`` (calls
    ``embedding_gen()`` first if the file is missing), computes the
    element-wise mean, and saves centroids to
    ``artifacts/intent_centroids.json``.

    Returns:
        dict mapping intent names to centroid vectors.
    """
    intent_embeddings = _load_intent_embeddings()

    intent_centroids: Dict[str, List[float]] = {}
    for intent_name, vectors in intent_embeddings.items():
        centroid = np.mean(vectors, axis=0).tolist()
        intent_centroids[intent_name] = centroid

    with open(INTENT_CENTROIDS_PATH, "w") as f:
        json.dump(intent_centroids, f)
    logger.info(f"✅ Intent centroids saved → {INTENT_CENTROIDS_PATH}")

    return intent_centroids


# ─── Task 3: Generate Domain Centroids ──────────────────────────────────────

def generate_domain_centroids() -> Dict[str, List[float]]:
    """Compute and cache the centroid (mean vector) per domain.

    Uses ``DOMAIN_REGISTRY`` to map domains → intents, then aggregates
    **all** sentence-level embeddings across those intents and computes
    the element-wise mean.

    Saves centroids to ``artifacts/domain_centroids.json``.

    Returns:
        dict mapping domain names to centroid vectors.
    """
    intent_embeddings = _load_intent_embeddings()
    domain_registry = embeddingVars.DOMAIN_REGISTRY

    domain_centroids: Dict[str, List[float]] = {}

    for domain_key, domain_info in domain_registry.items():
        all_vectors = []
        for intent_name in domain_info["intents"]:
            if intent_name in intent_embeddings:
                all_vectors.extend(intent_embeddings[intent_name])
            else:
                logger.warning(f"  ⚠️  Intent '{intent_name}' not found in embeddings — skipping")

        if all_vectors:
            centroid = np.mean(all_vectors, axis=0).tolist()
            domain_centroids[domain_key] = centroid
            logger.info(
                f"  ▸ Domain '{domain_key}': centroid from {len(all_vectors)} vectors "
                f"({len(domain_info['intents'])} intents)"
            )
        else:
            logger.warning(f"  ⚠️  No vectors for domain '{domain_key}'")

    with open(DOMAIN_CENTROIDS_PATH, "w") as f:
        json.dump(domain_centroids, f)
    logger.info(f"✅ Domain centroids saved → {DOMAIN_CENTROIDS_PATH}")

    return domain_centroids


# ─── Task 4: Classification & Accuracy Evaluation ───────────────────────────

def classify_and_evaluate(test_data: List[Dict[str, str]]) -> Dict[str, float]:
    """Classify texts by nearest centroid and measure accuracy.

    For each record the function:
    1. Generates an embedding for ``text`` via Vertex AI.
    2. Computes cosine similarity against every **intent centroid** →
       ``predicted_intent``.
    3. Computes cosine similarity against every **domain centroid** →
       ``predicted_domain``.

    Results are saved to ``outputs/classification_results.csv``.

    Args:
        test_data: list of dicts, each with keys
            ``text``, ``actual_classification_domain``,
            ``actual_classification_intent``.

    Returns:
        dict with ``intent_accuracy`` and ``domain_accuracy`` (0-100 %).
    """
    # Load centroids (generate if missing)
    if os.path.exists(INTENT_CENTROIDS_PATH):
        with open(INTENT_CENTROIDS_PATH, "r") as f:
            intent_centroids = json.load(f)
    else:
        intent_centroids = generate_intent_centroids()

    if os.path.exists(DOMAIN_CENTROIDS_PATH):
        with open(DOMAIN_CENTROIDS_PATH, "r") as f:
            domain_centroids = json.load(f)
    else:
        domain_centroids = generate_domain_centroids()

    embedder = _get_embedder()
    results = []

    for idx, record in enumerate(test_data):
        text = record["text"]
        actual_intent = record["actual_classification_intent"]
        actual_domain = record["actual_classification_domain"]

        # Embed the query text
        query_vec = np.array(embedder.embed(text))

        # Find nearest intent centroid
        predicted_intent = _nearest_centroid(query_vec, intent_centroids)

        # Find nearest domain centroid
        predicted_domain = _nearest_centroid(query_vec, domain_centroids)

        results.append({
            "text": text,
            "actual_intent": actual_intent,
            "predicted_intent": predicted_intent,
            "intent_match": actual_intent == predicted_intent,
            "actual_domain": actual_domain,
            "predicted_domain": predicted_domain,
            "domain_match": actual_domain == predicted_domain,
        })

        if (idx + 1) % 10 == 0:
            logger.info(f"  ▸ Classified {idx + 1}/{len(test_data)}")

    df = pd.DataFrame(results)

    output_path = os.path.join(OUTPUTS_DIR, "classification_results.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Results saved → {output_path}")

    intent_accuracy = df["intent_match"].mean() * 100
    domain_accuracy = df["domain_match"].mean() * 100

    print(f"\n{'='*50}")
    print(f"  Intent Classification Accuracy : {intent_accuracy:.2f}%")
    print(f"  Domain Classification Accuracy : {domain_accuracy:.2f}%")
    print(f"{'='*50}\n")

    return {"intent_accuracy": intent_accuracy, "domain_accuracy": domain_accuracy}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_intent_embeddings() -> Dict[str, List[List[float]]]:
    """Load intent embeddings from cache, generating them if needed."""
    if os.path.exists(INTENT_EMBEDDINGS_PATH):
        with open(INTENT_EMBEDDINGS_PATH, "r") as f:
            return json.load(f)
    return embedding_gen()


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def _nearest_centroid(query_vec: np.ndarray, centroids: Dict[str, List[float]]) -> str:
    """Return the centroid label with smallest Euclidean distance to *query_vec*."""
    best_label = None
    best_score = float('inf')
    for label, centroid in centroids.items():
        score = _euclidean_distance(query_vec, np.array(centroid))
        if score < best_score:
            best_score = score
            best_label = label
    return best_label


# ─── Main (convenience runner) ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Step 1 — Generating intent embeddings …")
    embedding_gen()

    print("Step 2 — Generating intent centroids …")
    generate_intent_centroids()

    print("Step 3 — Generating domain centroids …")
    generate_domain_centroids()

    print("\nAll artifacts generated successfully. ✅")
    print(f"  → {INTENT_EMBEDDINGS_PATH}")
    print(f"  → {INTENT_CENTROIDS_PATH}")
    print(f"  → {DOMAIN_CENTROIDS_PATH}")

    # ── Step 4: Run classification & evaluation on Testdata.csv ──────────────
    print("\nStep 4 — Running classify_and_evaluate on Testdata.csv …")

    TESTDATA_PATH = os.path.join(BASE_DIR, "Testdata.csv")

    if not os.path.exists(TESTDATA_PATH):
        print(f"⚠️  Testdata.csv not found at {TESTDATA_PATH} — skipping evaluation.")
    else:
        test_df = pd.read_csv(TESTDATA_PATH)

        # Validate required columns
        required_cols = {"Prompt", "Intent", "domain"}
        missing = required_cols - set(test_df.columns)
        if missing:
            raise ValueError(f"Testdata.csv is missing columns: {missing}")

        # Convert rows to the format expected by classify_and_evaluate
        test_data: List[Dict[str, str]] = [
            {
                "text": row["Prompt"],
                "actual_classification_intent": row["Intent"],
                "actual_classification_domain": row["domain"],
            }
            for _, row in test_df.iterrows()
        ]

        print(f"  ▸ Loaded {len(test_data)} test records from Testdata.csv")

        metrics = classify_and_evaluate(test_data)

        print(f"\n📊 Final Evaluation Results:")
        print(f"   Intent Accuracy : {metrics['intent_accuracy']:.2f}%")
        print(f"   Domain Accuracy : {metrics['domain_accuracy']:.2f}%")
        print(f"\n✅ Classification results saved → {os.path.join(OUTPUTS_DIR, 'classification_results.csv')}")
