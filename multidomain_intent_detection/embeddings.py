"""
Multidomain Intent Detection — Vertex AI Embedding Client
==========================================================

Singleton wrapper around Google Cloud Vertex AI text-embedding-005.
Handles:
  - Rate-limit retries with exponential backoff
  - Batch embedding with inter-request delays
  - Singleton pattern (one gRPC connection per process)
"""

import os
import time
import logging
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)


class VertexEmbeddings:
    """Google Cloud Vertex AI embeddings using text-embedding-005."""

    MAX_RETRIES = 5
    INITIAL_BACKOFF = 2.0
    BACKOFF_MULTIPLIER = 2.0
    INTER_REQUEST_DELAY = 0.3
    BATCH_PAUSE_EVERY = 20
    BATCH_PAUSE_SECONDS = 5.0

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
    ):
        self.project_id = project_id or os.getenv("PROJECT_ID", "pbm-poc-coderev-genai-poc")
        self.location = location or os.getenv("LOCATION", "us-central1")
        self.model_name = "text-embedding-005"
        self._client = None

        try:
            from google import genai
            self._client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
            logger.info(
                "Vertex AI embedder initialized "
                f"(project={self.project_id}, region={self.location})"
            )
        except ImportError:
            logger.error("google-genai SDK not installed. Run: pip install google-genai")
            raise
        except Exception as e:
            logger.error(f"Vertex AI auth failed: {e}")
            raise

    # ── Single text ──────────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        """Embed a single text string → 768-dim vector."""
        from google.genai import types

        backoff = self.INITIAL_BACKOFF
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.models.embed_content(
                    model=self.model_name,
                    contents=[types.Part.from_text(text=text)],
                )
                return response.embeddings[0].values
            except Exception as e:
                is_rate_limit = any(
                    k in str(e).lower()
                    for k in ("429", "exhausted", "quota", "too many requests")
                )
                if is_rate_limit and attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"Rate-limited, retry {attempt + 1} in {backoff:.0f}s")
                    time.sleep(backoff)
                    backoff *= self.BACKOFF_MULTIPLIER
                else:
                    raise

    # ── Batch ────────────────────────────────────────────────────────────

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts with rate-limit-aware pacing."""
        results = []
        for i, text in enumerate(texts):
            if i > 0:
                time.sleep(self.INTER_REQUEST_DELAY)
            if i > 0 and i % self.BATCH_PAUSE_EVERY == 0:
                logger.info(f"Batch pause after {i} embeddings…")
                time.sleep(self.BATCH_PAUSE_SECONDS)
            results.append(self.embed(text))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

_embedder_instance: Optional[VertexEmbeddings] = None
_embedder_lock = threading.Lock()


def get_embedder(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
) -> VertexEmbeddings:
    """Return the singleton VertexEmbeddings instance (thread-safe)."""
    global _embedder_instance
    if _embedder_instance is None:
        with _embedder_lock:
            if _embedder_instance is None:
                _embedder_instance = VertexEmbeddings(project_id=project_id, location=location)
    return _embedder_instance
