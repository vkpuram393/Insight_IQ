"""
Multidomain Intent Detection — Training & Evaluation
======================================================

End-to-end training workflow:
  1.  Load base embeddings from VamsiSir.CVS_INTENT_EXAMPLES
  2.  Augment with real-world phrasing patterns
  3.  PCA dimension search (5-fold CV)
  4.  Train final IntentPipeline ensemble
  5.  Evaluate on held-out testingFinalDataset.csv (ensemble-only + LLM hybrid)
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
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

#  These are mapping in reverse dirn i.e from intent to domain, and intent to description.
from multidomain_intent_detection.config import INTENT_TO_DOMAIN, INTENT_DESCRIPTIONS

# strips claim/sequence numbers so embeddings focus on intent semantics, not numeric IDs.
from multidomain_intent_detection.normalizer import normalize_query

# get_embedder : Return the singleton VertexEmbeddings instance (thread-safe). Basically give text-embedding005 for query
from multidomain_intent_detection.embeddings import get_embedder

from multidomain_intent_detection.pipeline import IntentPipeline

from multidomain_intent_detection.llm_fallback import llm_classify

# NEW (Task 2): Training-only LLM fallback wrapper — Gemini 2.5 Flash + thinking mode (6000-token budget).
# Scoped to this file only; production llm_classify above is untouched.
from multidomain_intent_detection.training_llm_fallback import training_llm_classify_with_thoughts

from multidomain_intent_detection.augmented_examples import AUGMENTED_EXAMPLES

logger = logging.getLogger(__name__)

# ── Temporarily disabled domains ──────────────────────────────────────────────
# Intents belonging to any domain in this set are excluded from both training
# and evaluation. Set to set() to re-enable all domains.
DISABLED_DOMAINS: set = {"benefits_api", "member_domain", "override_domain"}

# Set True to skip ensemble-only evaluation (Step 5) and go straight to LLM eval.
SKIP_ENSEMBLE_EVAL: bool = False

# Set True to skip LLM-hybrid evaluation (Step 6) entirely.
SKIP_LLM_EVAL: bool = False

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_IDS_DIR = os.path.join(os.path.dirname(BASE_DIR), "Intent_detection_system")

ARTIFACTS = os.path.join(_IDS_DIR, "artifacts") if os.path.isdir(os.path.join(_IDS_DIR, "artifacts")) else os.path.join(BASE_DIR, "artifacts")
OUTPUTS = os.path.join(_IDS_DIR, "outputs") if os.path.isdir(_IDS_DIR) else os.path.join(BASE_DIR, "outputs")

# To store intent embeddings
EMBEDDINGS_PATH = os.path.join(ARTIFACTS, "intent_embeddings.json")

# To store the final trained pipeline (PCA + ensemble)
MODEL_PKL = os.path.join(ARTIFACTS, "v3_pipeline_06082026.pkl")

os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

def _timestamp() -> str:
    """Return a filename-safe timestamp string like '2026-05-01_14-30-25'."""
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


# ─────────────────────────────────────────────────────────────────────────
# LLM fallback JSONL decision log
# ─────────────────────────────────────────────────────────────────────────
_LLM_FALLBACK_JSONL = "llm_fallback_decisions.jsonl"


def _resolve_llm_fallback_jsonl_path() -> str:
    base = Path(__file__).parent / "outputs"
    base.mkdir(parents=True, exist_ok=True)
    return str(base / _LLM_FALLBACK_JSONL)


def _log_llm_fallback_decision(
    user_prompt: str,
    actual_intent: str,
    ensemble_intent: str,
    ensemble_confidence: float,
    llm_decided_intent: str,
    llm_thinking_thoughts: str,
    llm_fallback_confidence: float,
) -> None:
    """Append one record to the LLM fallback decision JSONL."""
    try:
        path = _resolve_llm_fallback_jsonl_path()
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": user_prompt,
            "actual_intent": actual_intent or "",
            "ensemble_intent": ensemble_intent or "",
            "ensemble_confidence": round(float(ensemble_confidence), 4),
            "llm_intent": llm_decided_intent or "",
            "llm_confidence": round(float(llm_fallback_confidence), 4),
            "ensemble_correct": ensemble_intent == actual_intent,
            "llm_correct": llm_decided_intent == actual_intent,
            "thoughts": llm_thinking_thoughts or "",
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"[training] JSONL log write failed: {e}")


# ─────────────────────────────────────────────────────────────────────────
# Load base embeddings
# ─────────────────────────────────────────────────────────────────────────

def load_embeddings() -> Dict[str, list]:
    """Load or generate base intent embeddings from Intents_mapping.CVS_INTENT_EXAMPLES."""
    from multidomain_intent_detection.intents_mapping import embeddingVars
    examples = embeddingVars.CVS_INTENT_EXAMPLES

    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH) as f:
            cached = json.load(f)

        needed = set(INTENT_TO_DOMAIN.keys())
        missing = needed - set(cached.keys())

        #  If there are missing intents, log a warning and generate embeddings for them and update the cache.  This handles cases where new intents were added to INTENT_TO_DOMAIN after the cache was created.
        if missing:
            logger.warning(f"Cache missing {len(missing)} intents: {sorted(missing)}")
            emb = get_embedder()
            for intent in sorted(missing):
                if intent in examples:
                    cached[intent] = [list(v) for v in emb.embed_batch([normalize_query(s) for s in examples[intent]])]
                else:
                    logger.warning(f"No examples found for missing intent '{intent}'")

            with open(EMBEDDINGS_PATH, "w") as f:
                json.dump(cached, f)
        return cached

    emb = get_embedder()
    cached = {}
    for intent, sentences in examples.items():
        cached[intent] = [list(v) for v in emb.embed_batch([normalize_query(s) for s in sentences])]
    ts = _timestamp()
    emb_path = os.path.join(ARTIFACTS, f"intent_embeddings_{ts}.json")
    with open(emb_path, "w") as f:
        json.dump(cached, f)
    # Also write to the standard path so future runs find the cache
    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(cached, f)
    logger.info(f"Embeddings saved → {emb_path}")
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
        # Also save a timestamped copy
        ts_aug_path = os.path.join(ARTIFACTS, f"augmented_embeddings_v3_{_timestamp()}.json")
        with open(ts_aug_path, "w") as f:
            json.dump(aug_cached, f)
        logger.info(f"Augmented embeddings saved → {aug_cache_path} (copy: {ts_aug_path})")

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


# ─────────────────────────────────────────────────────────────────────────
# Build training arrays
# ─────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────
# PCA dimension search
# ─────────────────────────────────────────────────────────────────────────

def search_pca(X, y, labels) -> int:
    # Read search range from tuning_config.json
    import json
    cfg_path = os.path.join(BASE_DIR, "tuning_config.json")
    pca_range = [50, 75, 100, 150, 200, 250, 300, 400, 500]
    fixed_dims = None
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            pipe_cfg = cfg.get("pipeline", {})
            if not pipe_cfg.get("pca_search_enabled", True):
                fixed_dims = pipe_cfg.get("pca_dims")
            pca_range = pipe_cfg.get("pca_search_range", pca_range)
        except (json.JSONDecodeError, IOError):
            pass

    if fixed_dims:
        print(f"\n  PCA dims fixed at {fixed_dims} (pca_search_enabled=false)")
        return fixed_dims

    print(f"\n{'=' * 60}")
    print(f"  PCA DIMENSION SEARCH (5-fold CV)")
    print(f"{'=' * 60}")
    print(f"  {'Dims':>8} {'CV Acc':>10} {'Std':>8}")
    print(f"  {'-' * 28}")
    best_d, best_a = 50, 0.0
    for d in pca_range:
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


# ─────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────

def evaluate(test_data, pipeline, embedder, *, use_llm=True, conf_t=None, margin_t=None):
    """Evaluate pipeline with confusion-aware gating.  Returns {intent_accuracy, domain_accuracy, gate_stats}."""
    from multidomain_intent_detection.pipeline import CONFUSION_PRONE_INTENTS

    # Load gating thresholds from tuning_config.json
    _gate_cfg = {}
    _cfg_path = os.path.join(BASE_DIR, "tuning_config.json")
    if os.path.exists(_cfg_path):
        try:
            with open(_cfg_path) as _f:
                _gate_cfg = json.load(_f).get("gating", {})
        except (json.JSONDecodeError, IOError):
            pass
    cp_conf = _gate_cfg.get("confusion_prone_confidence", 0.55)
    cp_margin = _gate_cfg.get("confusion_prone_margin", 0.20)
    cpair_conf = _gate_cfg.get("confusion_pair_confidence", 0.60)
    cpair_margin = _gate_cfg.get("confusion_pair_margin", 0.25)

    # Use config defaults for base thresholds if not explicitly passed
    if conf_t is None:
        conf_t = _gate_cfg.get("confidence_threshold", 0.30)
    if margin_t is None:
        margin_t = _gate_cfg.get("margin_threshold", 0.05)

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
            if not (pred["confidence"] >= cp_conf and pred["margin"] >= cp_margin):
                confident = False
                gate_reason = "confusion_prone"
        if confident and pred.get("is_confusion_pair", False):
            if not (pred["confidence"] >= cpair_conf and pred["margin"] >= cpair_margin):
                confident = False
                gate_reason = "confusion_pair"

        # ──────────────────────────────────────────────────────────────
        # NEW (Task 2.1): FORCE LLM fallback in training.py regardless of
        # ensemble confidence. The gate/reason above are still computed
        # (for stats) but we override the routing decision here.
        # This bypass is LOCAL to training.py only.
        # ──────────────────────────────────────────────────────────────
        if use_llm:
            confident = False
            if gate_reason == "pass":
                gate_reason = "forced_llm_fallback"

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
            # NEW (Task 2): Use training-only wrapper → Gemini 2.5 Flash w/ thinking (6000-token budget).
            # Returns (intent, llm_confidence, thoughts_text). Then append the decision to CSV.
            final, llm_conf_val, llm_thoughts = training_llm_classify_with_thoughts(
                rec["text"],
                [n for n, _ in pred["top_5"]],
                ensemble_intent=pred["intent"],
                ensemble_confidence=pred["confidence"],
            )
            src = "llm"
            llm_n += 1
            try:
                _log_llm_fallback_decision(
                    user_prompt=rec["text"],
                    actual_intent=rec.get("actual_intent", ""),
                    ensemble_intent=pred["intent"],
                    ensemble_confidence=pred["confidence"],
                    llm_decided_intent=final,
                    llm_thinking_thoughts=llm_thoughts,
                    llm_fallback_confidence=llm_conf_val,
                )
            except Exception as _log_e:  # never let logging break eval
                logger.warning(f"JSONL log write failed: {_log_e}")

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
    ts = _timestamp()
    df.to_csv(os.path.join(OUTPUTS, f"results_{mode}_{ts}.csv"), index=False)

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

    # ── Per-intent accuracy table ──────────────────────────────────────────────
    print(f"\n  {'Intent':<40} {'Domain':<25} {'Acc':>7} {'Count':>6}")
    print(f"  {'-'*82}")
    intent_rows = []
    for intent, grp in df.groupby("actual_intent"):
        domain = INTENT_TO_DOMAIN.get(intent, "unknown")
        intent_rows.append((domain, intent, grp["intent_match"].mean() * 100, len(grp)))
    for domain, intent, acc, cnt in sorted(intent_rows, key=lambda r: (r[0], r[1])):
        print(f"  {intent:<40} {domain:<25} {acc:>6.1f}% {cnt:>6}")

    # ── Per-domain accuracy table (prediction-based) ───────────────────────────
    print(f"\n  {'Domain':<25} {'Domain Acc':>10} {'Count':>6}")
    print(f"  {'-'*45}")
    for dom, grp in sorted(df.groupby("actual_domain"), key=lambda kv: kv[0]):
        print(f"  {dom:<25} {grp['domain_match'].mean()*100:>9.1f}% {len(grp):>6}")

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

    # Test data — prefer combinedtestset.csv (broader coverage), fall back to testingFinalDataset.csv
    for candidate in [
        os.path.join(os.path.dirname(BASE_DIR), "combinedtestset.csv"),
    ]:
        if os.path.exists(candidate):
            TESTDATA = candidate
            break
    else:
        print("No test dataset found (combinedtestset.csv / testingFinalDataset.csv)")
        return

    tdf = pd.read_csv(TESTDATA)

    # Normalize column names — combinedtestset.csv uses (prompt, expected_intent, expected_domain)
    # while testingFinalDataset.csv uses (Prompt, Intent, domain).
    col_map = {}
    for col in tdf.columns:
        lc = col.lower()
        if lc in ("prompt",):
            col_map[col] = "Prompt"
        elif lc in ("expected_intent", "intent"):
            col_map[col] = "Intent"
        elif lc in ("expected_domain", "domain"):
            col_map[col] = "domain"
    if col_map:
        tdf.rename(columns=col_map, inplace=True)

    test_data = [
        {"text": r["Prompt"], "actual_intent": r["Intent"], "actual_domain": r["domain"]}
        for _, r in tdf.iterrows()
    ]
    print(f"  Test file: {os.path.basename(TESTDATA)}")
    print(f"  Test: {len(test_data)} queries, {tdf['Intent'].nunique()} intents")

    train_intents = set(all_emb.keys()) & set(INTENT_TO_DOMAIN.keys())

    if DISABLED_DOMAINS:
        disabled_intents = {i for i, d in INTENT_TO_DOMAIN.items() if d in DISABLED_DOMAINS}
        train_intents -= disabled_intents
        test_data = [r for r in test_data if r["actual_intent"] not in disabled_intents]
        print(f"  Disabled domains {DISABLED_DOMAINS}: excluding {sorted(disabled_intents)}")
        print(f"  Test data after domain filter: {len(test_data)} queries")

    print("\nStep 2 — Augmenting training data...")
    augmented_emb = augment_embeddings(all_emb)
    X, y, labels = build_Xy(augmented_emb, train_intents)
    print(f"  {X.shape[0]} samples × {X.shape[1]} dims, {len(labels)} classes")

    print("\nStep 3 — PCA dimension search...")
    best_dim = search_pca(X, y, labels)

    print(f"\nStep 4 — Training final ensemble (PCA-{best_dim})...")
    pipe = IntentPipeline(n_pca=best_dim)
    pipe.fit(X, y, labels)
    ts = _timestamp()
    model_path = os.path.join(ARTIFACTS, f"v3_pipeline_{ts}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    # Also save to the standard path for downstream use
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(pipe, f)
    print(f"  Saved → {model_path}")
    print(f"  Saved → {MODEL_PKL} (latest)")

    embedder = get_embedder()

    if not SKIP_ENSEMBLE_EVAL:
        print("\nStep 5 — Evaluation (ensemble only)...")
        m1 = evaluate(test_data, pipe, embedder, use_llm=False)
    else:
        print("\nStep 5 — Skipped (SKIP_ENSEMBLE_EVAL=True)")
        m1 = None

    if "--no-llm" not in sys.argv and not SKIP_LLM_EVAL:
        print("\nStep 6 — Evaluation (ensemble + LLM fallback)...")
        m2 = evaluate(test_data, pipe, embedder, use_llm=True)
        print(f"\n{'=' * 60}")
        if m1 is not None:
            print(f"  Ensemble only:  {m1['intent_accuracy']:.1f}%")
        print(f"  +LLM fallback:  {m2['intent_accuracy']:.1f}%")
        print(f"{'=' * 60}")
    elif SKIP_LLM_EVAL:
        print("\nStep 6 — Skipped (SKIP_LLM_EVAL=True)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_and_evaluate()