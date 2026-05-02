"""
Multidomain Intent Detection — Training & Evaluation
======================================================

End-to-end training workflow:
  1.  Load base embeddings from VamsiSir.CVS_INTENT_EXAMPLES
  2.  Augment with real-world phrasing patterns
  3.  PCA dimension search (5-fold CV)
  4.  Train final IntentPipeline ensemble
  5.  Evaluate on held-out Testdata.csv (ensemble-only + LLM hybrid)
  6.  Save pipeline to artifacts/v3_pipeline.pkl

Run directly:
    python -m multidomain_intent_detection.training

Or import individual functions for custom workflows.
"""

import os
import sys
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from multidomain_intent_detection.config import INTENT_TO_DOMAIN, INTENT_DESCRIPTIONS
from multidomain_intent_detection.normalizer import normalize_query
from multidomain_intent_detection.embeddings import get_embedder
from multidomain_intent_detection.pipeline import IntentPipeline
from multidomain_intent_detection.llm_fallback import llm_classify
from multidomain_intent_detection.augmented_examples import AUGMENTED_EXAMPLES

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_IDS_DIR = os.path.join(os.path.dirname(BASE_DIR), "multidomain_intent_detection")
ARTIFACTS = os.path.join(_IDS_DIR, "artifacts") if os.path.isdir(os.path.join(_IDS_DIR, "artifacts")) else os.path.join(BASE_DIR, "artifacts")
OUTPUTS = os.path.join(_IDS_DIR, "outputs") if os.path.isdir(os.path.join(_IDS_DIR, "outputs")) else os.path.join(BASE_DIR, "outputs")
EMBEDDINGS_PATH = os.path.join(ARTIFACTS, "intent_embeddings.json")
MODEL_PKL = os.path.join(ARTIFACTS, "v3_pipeline.pkl")

os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load base embeddings
# ─────────────────────────────────────────────────────────────────────────────

def load_embeddings() -> Dict[str, list]:
    """Load or generate base intent embeddings from VamsiSir.CVS_INTENT_EXAMPLES."""
    # Import the intent examples registry
    sys.path.insert(0, _IDS_DIR)
    from intents_mapping import embeddingVars
    examples = embeddingVars.CVS_INTENT_EXAMPLES

    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH) as f:
            cached = json.load(f)
        needed = set(INTENT_TO_DOMAIN.keys())
        missing = needed - set(cached.keys())
        if missing:
            logger.warning(f"Cache missing {len(missing)} intents: {sorted(missing)}")
            emb = get_embedder()
            for intent in sorted(missing):
                if intent in examples:
                    cached[intent] = [list(v) for v in emb.embed_batch(examples[intent])]
            with open(EMBEDDINGS_PATH, "w") as f:
                json.dump(cached, f)
        return cached

    emb = get_embedder()
    cached = {}
    for intent, sentences in examples.items():
        cached[intent] = [list(v) for v in emb.embed_batch(sentences)]
    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(cached, f)
    return cached


def augment_embeddings(embeddings: Dict) -> Dict:
    """Add augmented training examples to cached embeddings.

    Normalizes examples (strips claim numbers) before embedding so they
    match the normalized test query space.

    Cache invalidation uses content hashing — if any example text changes
    (not just count), the affected intents are re-embedded.
    """
    import hashlib

    aug_cache_path = os.path.join(ARTIFACTS, "augmented_embeddings_v3.json")

    if os.path.exists(aug_cache_path):
        with open(aug_cache_path) as f:
            aug_cached = json.load(f)
    else:
        aug_cached = {}

    # Compute content hash per intent for invalidation
    def _content_hash(texts):
        normalized = [normalize_query(t) for t in texts]
        return hashlib.sha256("|".join(normalized).encode()).hexdigest()[:16]

    needs_generation = False
    for intent, examples in AUGMENTED_EXAMPLES.items():
        expected_hash = _content_hash(examples)
        cached_hash = aug_cached.get(f"_hash_{intent}", "")
        if (intent not in aug_cached
                or len(aug_cached[intent]) != len(examples)
                or cached_hash != expected_hash):
            needs_generation = True
            break

    if needs_generation:
        logger.info("Generating augmented training embeddings (normalized)...")
        emb = get_embedder()
        for intent, examples in AUGMENTED_EXAMPLES.items():
            expected_hash = _content_hash(examples)
            cached_hash = aug_cached.get(f"_hash_{intent}", "")
            if (intent not in aug_cached
                    or len(aug_cached[intent]) != len(examples)
                    or cached_hash != expected_hash):
                normalized = [normalize_query(ex) for ex in examples]
                logger.info(f"  Augmenting '{intent}' with {len(examples)} normalized examples")
                aug_cached[intent] = [list(v) for v in emb.embed_batch(normalized)]
                aug_cached[f"_hash_{intent}"] = expected_hash
        # Write atomically (temp file + rename)
        tmp_path = aug_cache_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(aug_cached, f)
        os.replace(tmp_path, aug_cache_path)
        logger.info(f"Augmented embeddings saved → {aug_cache_path}")

    # Merge (skip _hash_ metadata keys)
    merged = {k: list(v) for k, v in embeddings.items()}
    for intent, vecs in aug_cached.items():
        if intent.startswith("_hash_"):
            continue  # skip metadata keys
        if intent in merged:
            merged[intent] = merged[intent] + vecs
        else:
            merged[intent] = vecs
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Build training arrays
# ─────────────────────────────────────────────────────────────────────────────

def build_Xy(embeddings: Dict, filter_intents=None):
    """Convert {intent: [vectors]} → (X, y, label_names)."""
    X, y, labels, lmap = [], [], [], {}
    for name in sorted(embeddings.keys()):
        if filter_intents and name not in filter_intents:
            continue
        if name not in lmap:
            lmap[name] = len(labels)
            labels.append(name)
        for vec in embeddings[name]:
            X.append(vec)
            y.append(lmap[name])
    return np.array(X), np.array(y), labels


# ─────────────────────────────────────────────────────────────────────────────
# PCA dimension search
# ─────────────────────────────────────────────────────────────────────────────

def search_pca(X, y, labels) -> int:
    print(f"\n{'=' * 60}")
    print(f"  PCA DIMENSION SEARCH (5-fold CV)")
    print(f"{'=' * 60}")
    print(f"  {'Dims':>8} {'CV Acc':>10} {'Std':>8}")
    print(f"  {'-' * 28}")
    best_d, best_a = 50, 0.0
    for d in [20, 30, 40, 50, 75, 100, 150, 200, 250, 300]:
        if d >= X.shape[0]:
            continue
        p = IntentPipeline(n_pca=d)
        p.label_names = labels
        a, s, _ = p.cross_validate(X, y)
        star = " <-- BEST" if a > best_a else ""
        print(f"  {d:>8} {a * 100:>8.2f}% {s * 100:>6.2f}%{star}")
        if a > best_a:
            best_a = a
            best_d = d
    print(f"\n  Optimal: PCA-{best_d} ({best_a * 100:.2f}%)")
    print(f"{'=' * 60}\n")
    return best_d


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(test_data, pipeline, embedder, *, use_llm=True, conf_t=0.30, margin_t=0.05):
    """Evaluate pipeline with confusion-aware gating.  Returns {intent_accuracy, domain_accuracy, gate_stats}."""
    from multidomain_intent_detection.pipeline import CONFUSION_PRONE_INTENTS

    results, llm_n = [], 0
    gate_stats = {
        "ensemble_pass": 0, "disagreement_to_llm": 0,
        "low_conf_to_llm": 0, "confusion_pair_to_llm": 0,
    }

    for idx, rec in enumerate(test_data):
        normalized_text = normalize_query(rec["text"])
        vec = np.array(embedder.embed(normalized_text))
        pred = pipeline.predict_single(vec)

        # ── Gating (mirrors classifier.py) ───────────────────────────
        confident = (
            pred["confidence"] >= conf_t
            and pred["margin"] >= margin_t
            and pred["agreement"]
        )
        gate_reason = "pass"

        if not pred["agreement"]:
            confident = False
            gate_reason = "disagreement"
        elif confident and pred["intent"] in CONFUSION_PRONE_INTENTS:
            if not (pred["confidence"] >= 0.55 and pred["margin"] >= 0.20):
                confident = False
                gate_reason = "confusion_prone"
        if confident and pred.get("is_confusion_pair", False):
            if not (pred["confidence"] >= 0.60 and pred["margin"] >= 0.25):
                confident = False
                gate_reason = "confusion_pair"

        # Track gate stats
        if confident:
            gate_stats["ensemble_pass"] += 1
        elif gate_reason == "disagreement":
            gate_stats["disagreement_to_llm"] += 1
        elif gate_reason == "confusion_pair":
            gate_stats["confusion_pair_to_llm"] += 1
        else:
            gate_stats["low_conf_to_llm"] += 1

        # ── Classify ─────────────────────────────────────────────────
        if confident or not use_llm:
            final, src = pred["intent"], "ensemble"
        else:
            final = llm_classify(
                rec["text"],
                [n for n, _ in pred["top_5"]],
                ensemble_intent=pred["intent"],
                ensemble_confidence=pred["confidence"],
            )
            src = "llm"
            llm_n += 1

        results.append({
            "text": rec["text"],
            "actual_intent": rec["actual_intent"],
            "predicted_intent": final,
            "intent_match": rec["actual_intent"] == final,
            "actual_domain": rec["actual_domain"],
            "predicted_domain": INTENT_TO_DOMAIN.get(final, "unknown"),
            "domain_match": rec["actual_domain"] == INTENT_TO_DOMAIN.get(final, "unknown"),
            "confidence": pred["confidence"],
            "raw_confidence": pred.get("raw_confidence", pred["confidence"]),
            "margin": pred["margin"],
            "source": src,
            "agreement": pred["agreement"],
            "is_confusion_pair": pred.get("is_confusion_pair", False),
            "gate_reason": gate_reason,
        })
        if (idx + 1) % 50 == 0:
            logger.info(f"  {idx + 1}/{len(test_data)}")

    df = pd.DataFrame(results)
    mode = "hybrid" if use_llm else "ensemble_only"
    df.to_csv(os.path.join(OUTPUTS, f"results_{mode}.csv"), index=False)

    ia = df["intent_match"].mean() * 100
    da = df["domain_match"].mean() * 100
    errors = df[~df["intent_match"]]
    high_conf_errors = errors[errors["raw_confidence"] >= 0.85]

    print(f"\n{'=' * 65}")
    print(f"  {'ENSEMBLE + LLM' if use_llm else 'ENSEMBLE ONLY'}")
    print(f"  Intent Accuracy    : {ia:.2f}%")
    print(f"  Domain Accuracy    : {da:.2f}%")
    print(f"  Total Errors       : {len(errors)}")
    print(f"  High-conf Errors   : {len(high_conf_errors)} (raw conf >= 0.85)")
    if use_llm:
        ep = (len(test_data) - llm_n) / len(test_data) * 100
        print(f"  Ensemble resolved  : {ep:.1f}% ({len(test_data) - llm_n}/{len(test_data)})")
        print(f"  LLM calls          : {llm_n} ({llm_n / len(test_data) * 100:.1f}%)")
    print(f"\n  Gate Statistics:")
    for k, v in gate_stats.items():
        print(f"    {k:25s}: {v}")
    print(f"{'=' * 65}")

    print(f"\n  {'Domain':<25} {'Intent':>8} {'Domain':>8} {'Count':>6}")
    print(f"  {'-'*50}")
    for dom in sorted(df["actual_domain"].unique()):
        s = df[df["actual_domain"] == dom]
        print(f"  {dom:<25} {s['intent_match'].mean()*100:>6.1f}% {s['domain_match'].mean()*100:>6.1f}% {len(s):>6}")

    if len(errors) > 0:
        print(f"\n  Top Confusions:")
        conf = errors.groupby(["actual_intent", "predicted_intent"]).size().sort_values(ascending=False).head(10)
        for (a, p), c in conf.items():
            print(f"    {a} -> {p}: {c}")

    if use_llm and llm_n:
        lr = df[df["source"] == "llm"]
        print(f"\n  LLM accuracy: {lr['intent_match'].mean()*100:.1f}% ({int(lr['intent_match'].sum())}/{llm_n})")

    print()
    return {"intent_accuracy": ia, "domain_accuracy": da, "gate_stats": gate_stats}


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry‑point
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate():
    """Full training + evaluation pipeline."""
    print("=" * 65)
    print("  Multidomain Intent Detection — Training")
    print("  PCA + Ensemble (SVM-RBF / LogReg / kNN) + LLM Fallback")
    print("=" * 65)

    print("\nStep 1 — Loading cached embeddings...")
    all_emb = load_embeddings()

    # Test data
    for candidate in [
        os.path.join(_IDS_DIR, "Testdata_corrected.csv"),
        os.path.join(_IDS_DIR, "Testdata.csv"),
    ]:
        if os.path.exists(candidate):
            TESTDATA = candidate
            break
    else:
        print("Testdata.csv not found")
        return

    tdf = pd.read_csv(TESTDATA)
    test_data = [
        {"text": r["Prompt"], "actual_intent": r["Intent"], "actual_domain": r["domain"]}
        for _, r in tdf.iterrows()
    ]
    print(f"  Test: {len(test_data)} queries, {tdf['Intent'].nunique()} intents")

    train_intents = set(all_emb.keys()) & set(INTENT_TO_DOMAIN.keys())

    print("\nStep 2 — Augmenting training data...")
    augmented_emb = augment_embeddings(all_emb)
    X, y, labels = build_Xy(augmented_emb, train_intents)
    print(f"  {X.shape[0]} samples × {X.shape[1]} dims, {len(labels)} classes")

    print("\nStep 3 — PCA dimension search...")
    best_dim = search_pca(X, y, labels)

    print(f"\nStep 4 — Training final ensemble (PCA-{best_dim})...")
    pipe = IntentPipeline(n_pca=best_dim)
    pipe.fit(X, y, labels)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(pipe, f)
    print(f"  Saved → {MODEL_PKL}")

    print("\nStep 5 — Evaluation (ensemble only)...")
    embedder = get_embedder()
    m1 = evaluate(test_data, pipe, embedder, use_llm=False)

    if "--no-llm" not in sys.argv:
        print("\nStep 6 — Evaluation (ensemble + LLM fallback)...")
        m2 = evaluate(test_data, pipe, embedder, use_llm=True)
        print(f"\n{'=' * 60}")
        print(f"  FINAL: Ensemble {m1['intent_accuracy']:.1f}% → +LLM {m2['intent_accuracy']:.1f}%")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_and_evaluate()
