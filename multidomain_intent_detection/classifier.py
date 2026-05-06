"""
Multidomain Intent Detection — Production Classifier
======================================================

Loads the pre-trained PCA + Ensemble pipeline and provides a simple API
for classifying user prompts into intents and domains.

Usage (sync):
    from multidomain_intent_detection import get_classifier

    classifier = get_classifier()
    result = classifier.classify("What is the copay on claim 132435151040074 sequence 001?")

    print(result["intent"])       # "pricing_info"
    print(result["domain"])       # "cap_api"
    print(result["confidence"])   # 0.97
    print(result["source"])       # "ensemble" or "llm"
    print(result["entities"])     # {"claim_number": "132435151040074", ...}

Usage (async):
    result = await classifier.classify_async("Show all rejected claims")

Architecture:
    1.  Normalize query (strip claim numbers for embedding focus)
    2.  Embed with Vertex AI text-embedding-005
    3.  Predict with PCA + Ensemble (SVM-RBF / LogReg / kNN)
    4.  If confidence < threshold → LLM fallback (Gemini Flash)
    5.  Return intent, domain, confidence, entities, metadata
"""

import os
import sys
import time
import json
import pickle
import asyncio
import logging
import threading
import numpy as np
from typing import Dict, Any, List, Optional

from multidomain_intent_detection.config import (
    INTENT_TO_DOMAIN,
    DOMAIN_ENDPOINTS,
    DOMAIN_NAMES,
)
from multidomain_intent_detection.normalizer import normalize_query, extract_entities
from multidomain_intent_detection.embeddings import get_embedder
from multidomain_intent_detection.llm_fallback import llm_classify
# Import the canonical set from pipeline (auto-derived from CONFUSION_PAIRS)
from multidomain_intent_detection.pipeline import CONFUSION_PRONE_INTENTS as _CONFUSION_PRONE_INTENTS

logger = logging.getLogger(__name__)

# ── Load tuning config for gating thresholds ─────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_config.json")

def _load_gate_config() -> Dict:
    """Load gating thresholds from tuning_config.json."""
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH) as f:
                cfg = json.load(f)
            return cfg.get("gating", {})
        except (json.JSONDecodeError, IOError):
            pass
    return {}

def _load_prod_config() -> Dict:
    """Load production config from tuning_config.json."""
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH) as f:
                cfg = json.load(f)
            return cfg.get("production", {})
        except (json.JSONDecodeError, IOError):
            pass
    return {}

_GATE_CFG = _load_gate_config()
_PROD_CFG = _load_prod_config()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_model_pkl() -> str:
    """Search multiple candidate locations for v3_pipeline.pkl."""
    project_root = os.path.dirname(BASE_DIR)
    candidates = [
        # 1. artifacts/ inside this package
        os.path.join(BASE_DIR, "artifacts", "v3_pipeline.pkl"),
        # 2. Intent_detection_system/artifacts/ (sibling folder)
        os.path.join(project_root, "Intent_detection_system", "artifacts", "v3_pipeline.pkl"),
        # 3. Relative to cwd
        os.path.join(os.getcwd(), "Intent_detection_system", "artifacts", "v3_pipeline.pkl"),
        os.path.join(os.getcwd(), "artifacts", "v3_pipeline.pkl"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    # Fallback: return the most likely location (will raise FileNotFoundError later)
    return candidates[1]


MODEL_PKL = _find_model_pkl()


class MultidomainIntentClassifier:
    """
    Production intent classifier using the v3 PCA + Ensemble pipeline.

    Returns a standardized dict:
    {
        "intent":              str,    # e.g. "pricing_info"
        "domain":              str,    # e.g. "cap_api"
        "domain_name":         str,    # e.g. "Cap-API"
        "api_endpoint":        str,    # e.g. "/myclaims/claims/v1/..."
        "confidence":          float,  # 0.0–1.0
        "margin":              float,  # gap between top-1 and top-2
        "source":              str,    # "ensemble" or "llm"
        "agreement":           bool,   # all 3 sub-classifiers agree
        "top_5":               list,   # [(intent, prob), ...]
        "entities":            dict,   # extracted entities
        "needs_clarification": bool,
        "latency_ms":          float,
    }
    """

    def __init__(
        self,
        model_path: str = MODEL_PKL,
        confidence_threshold: float = None,
        margin_threshold: float = None,
        use_llm_fallback: bool = True,
    ):
        # Read defaults from config, allow overrides
        gate_cfg = _GATE_CFG
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else gate_cfg.get("confidence_threshold", 0.70)
        self.margin_threshold = margin_threshold if margin_threshold is not None else gate_cfg.get("margin_threshold", 0.05)
        self.use_llm_fallback = use_llm_fallback
        self._pipeline = None
        self._model_path = model_path
        self._load_time: Optional[float] = None
        self._load_lock = threading.Lock()

    # ── Lazy load ────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        """Lazy-load the pipeline on first classify() call (thread-safe)."""
        if self._pipeline is not None:
            return

        with self._load_lock:
            if self._pipeline is not None:
                return  # another thread loaded it while we waited

            # Re-resolve path at load time (cwd may have changed since import)
            if not os.path.exists(self._model_path):
                self._model_path = _find_model_pkl()

            if not os.path.exists(self._model_path):
                raise FileNotFoundError(
                    f"Trained pipeline not found at {self._model_path}. "
                    "Run the training script first to train and save the model.\n"
                    "  python -m multidomain_intent_detection.training"
                )

            # The pickle stores the class as '__main__.IntentPipeline' because the
            # training script was run directly.  Make it resolvable here too.
            from multidomain_intent_detection.pipeline import IntentPipeline

            import __main__
            if not hasattr(__main__, "IntentPipeline"):
                __main__.IntentPipeline = IntentPipeline

            # Also try the old module path for backward compat
            try:
                from Intent_detection_system import intent_detection_v3
            except ImportError:
                pass

            t0 = time.time()
            try:
                with open(self._model_path, "rb") as f:
                    self._pipeline = pickle.load(f)
            except (pickle.UnpicklingError, ModuleNotFoundError,
                    AttributeError, EOFError, ImportError) as e:
                raise RuntimeError(
                    f"Failed to load pipeline from {self._model_path}: {e}\n"
                    "The model file may be corrupted or built with an incompatible "
                    "sklearn version. Re-run training to rebuild it."
                ) from e
            self._load_time = time.time() - t0

            n_intents = len(self._pipeline.label_names)
            logger.info(
                f"Pipeline loaded: {n_intents} intents, "
                f"PCA-{self._pipeline.n_pca}, "
                f"loaded in {self._load_time * 1000:.0f}ms"
            )

    # ── Classify ─────────────────────────────────────────────────────────

    def classify(self, query: str) -> Dict[str, Any]:
        """Classify a user prompt into intent + domain.

        Args:
            query: Raw user prompt (may contain claim numbers, etc.)

        Returns:
            Classification result dict (see class docstring for schema).
        """
        # Input validation
        if not query or not isinstance(query, str):
            return {
                "intent": "out_of_scope", "domain": "general",
                "domain_name": "General", "api_endpoint": None,
                "confidence": 0.0, "margin": 0.0, "source": "validation",
                "agreement": True, "top_5": [], "entities": {},
                "needs_clarification": True, "latency_ms": 0.0,
            }
        query = query[:5000]  # hard limit to prevent resource exhaustion

        self._ensure_loaded()
        t0 = time.time()

        # 1. Extract entities from raw text
        entities = extract_entities(query)

        # 2. Normalize query (strip numbers) for embedding
        normalized = normalize_query(query)

        # 3. Embed
        embedder = get_embedder()
        vec = np.array(embedder.embed(normalized))

        # 4. Predict with ensemble
        pred = self._pipeline.predict_single(vec)

        # 5. Confidence gate → optional LLM fallback
        #    The pipeline's predict_single() already applies three calibration
        #    layers (temperature, disagreement penalty, confusion-pair penalty).
        #    The confidence/margin values we receive are POST-calibration, so
        #    even a query the raw ensemble scored at 0.92 may arrive here at
        #    0.55 if sub-classifiers disagreed AND it's a confusion pair.
        #
        #    Gate logic:
        #    a) Base gate: confidence >= threshold AND margin >= threshold
        #    b) Agreement gate: all 3 sub-classifiers must agree
        #    c) Confusion-pair gate: stricter thresholds for known-ambiguous intents
        confident = (
            pred["confidence"] >= self.confidence_threshold
            and pred["margin"] >= self.margin_threshold
            and pred["agreement"]  # ALL 4 sub-classifiers must agree
        )

        # For known confusion-prone intents, apply stricter thresholds
        # even when the gate above passed (catches high-conf wrong answers)
        cp_conf = _GATE_CFG.get("confusion_prone_confidence", 0.55)
        cp_margin = _GATE_CFG.get("confusion_prone_margin", 0.20)
        if confident and pred["intent"] in _CONFUSION_PRONE_INTENTS:
            confident = pred["confidence"] >= cp_conf and pred["margin"] >= cp_margin

        # If pipeline flagged this as a confusion pair, require extra clearance
        cpair_conf = _GATE_CFG.get("confusion_pair_confidence", 0.60)
        cpair_margin = _GATE_CFG.get("confusion_pair_margin", 0.25)
        if confident and pred.get("is_confusion_pair", False):
            confident = pred["confidence"] >= cpair_conf and pred["margin"] >= cpair_margin

        if confident or not self.use_llm_fallback:
            final_intent = pred["intent"]
            source = "ensemble"
        else:
            logger.info(
                f"Low confidence ({pred['confidence']:.2f}), "
                f"margin ({pred['margin']:.2f}) — calling LLM fallback"
            )
            final_intent = llm_classify(
                query,
                [name for name, _ in pred["top_5"]],
                ensemble_intent=pred["intent"],
                ensemble_confidence=pred["confidence"],
            )
            source = "llm"

        # 6. Resolve domain
        domain = INTENT_TO_DOMAIN.get(final_intent, "unknown")
        elapsed_ms = (time.time() - t0) * 1000

        result = {
            "intent": final_intent,
            "domain": domain,
            "domain_name": DOMAIN_NAMES.get(domain, domain),
            "api_endpoint": DOMAIN_ENDPOINTS.get(domain),
            "confidence": pred["confidence"],
            "margin": pred["margin"],
            "source": source,
            "agreement": pred["agreement"],
            "top_5": pred["top_5"],
            "entities": entities,
            "needs_clarification": pred["confidence"] < _PROD_CFG.get("needs_clarification_threshold", 0.4),
            "latency_ms": round(elapsed_ms, 1),
        }

        logger.info(
            f"Classified: '{query[:60]}...' → "
            f"intent={final_intent}, domain={domain}, "
            f"conf={pred['confidence']:.2f}, src={source}, "
            f"{elapsed_ms:.0f}ms"
        )
        return result

    # ── Async ────────────────────────────────────────────────────────────

    async def classify_async(self, query: str) -> Dict[str, Any]:
        """Async wrapper — runs classify() in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.classify, query)

    # ── Batch ────────────────────────────────────────────────────────────

    def classify_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple queries sequentially."""
        return [self.classify(q) for q in queries]

    # ── Model info ───────────────────────────────────────────────────────

    @property
    def model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        self._ensure_loaded()
        return {
            "n_intents": len(self._pipeline.label_names),
            "intents": sorted(self._pipeline.label_names),
            "n_domains": len(set(INTENT_TO_DOMAIN.values())),
            "domains": sorted(set(INTENT_TO_DOMAIN.values())),
            "pca_dims": self._pipeline.n_pca,
            "temperature": self._pipeline.temperature,
            "ensemble_weights": self._pipeline.weights,
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "use_llm_fallback": self.use_llm_fallback,
            "model_path": self._model_path,
            "load_time_ms": (
                round(self._load_time * 1000, 1) if self._load_time else None
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

_classifier_instance: Optional[MultidomainIntentClassifier] = None
_classifier_lock = threading.Lock()


def get_classifier(
    confidence_threshold: float = None,
    margin_threshold: float = None,
    use_llm_fallback: bool = True,
) -> MultidomainIntentClassifier:
    """Get the singleton MultidomainIntentClassifier instance (thread-safe).

    First call creates the classifier (lazy—pipeline loaded on first classify()).
    Subsequent calls return the same instance.

    Thresholds default to values from tuning_config.json if not specified.
    """
    global _classifier_instance
    if _classifier_instance is None:
        with _classifier_lock:
            if _classifier_instance is None:
                _classifier_instance = MultidomainIntentClassifier(
                    confidence_threshold=confidence_threshold,
                    margin_threshold=margin_threshold,
                    use_llm_fallback=use_llm_fallback,
                )
                logger.info("MultidomainIntentClassifier singleton created")
    return _classifier_instance
