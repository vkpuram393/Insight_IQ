"""
Batch Intent Classification Test Runner
=========================================

Loads a labeled CSV (Prompt,Intent,domain) and evaluates the trained
model using the SAME dual-pass approach as training.py:

  Pass 1 — Ensemble only (no LLM fallback)
  Pass 2 — Hybrid (ensemble + LLM fallback for low-confidence queries)

Reports for EACH pass:
  - Intent accuracy, domain accuracy
  - High-confidence errors (raw conf >= 0.85)
  - Ensemble resolved % and LLM call count
  - Gate statistics (why queries went to LLM)
  - LLM-only accuracy
  - Per-domain breakdown
  - Per-intent breakdown (worst-first)
  - Top confusion pairs
  - Full failure details

Then prints a final delta: Ensemble X% → +LLM Y%

Usage:
    # Default: dual-pass on testingFinalDataset.csv
    python -m multidomain_intent_detection.batch_test

    # Custom CSV
    python -m multidomain_intent_detection.batch_test --csv path/to/data.csv

    # Ensemble-only (skip the LLM pass — faster, no API calls)
    python -m multidomain_intent_detection.batch_test --no-llm

    # Verbose: show each row result inline
    python -m multidomain_intent_detection.batch_test --verbose

    # JSON summary output
    python -m multidomain_intent_detection.batch_test --json
"""

import sys
import os
import logging
import argparse
import json
import time
import warnings
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

# Suppress sklearn joblib parallel warnings (harmless, from ExtraTreesClassifier)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*sklearn.*")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import pandas as pd

# Fix Windows console encoding for box-drawing characters
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from multidomain_intent_detection.config import INTENT_TO_DOMAIN
from multidomain_intent_detection.normalizer import normalize_query
from multidomain_intent_detection.embeddings import get_embedder
from multidomain_intent_detection.llm_fallback import llm_classify

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _find_default_csv() -> str:
    """Search for testingFinalDataset.csv in standard locations."""
    candidates = [
        os.path.join(BASE_DIR, "testingFinalDataset.csv"),
        os.path.join(os.getcwd(), "testingFinalDataset.csv"),
        os.path.join(BASE_DIR, "Testdata_corrected.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def _resolve_outputs_dir() -> str:
    """Resolve outputs directory (mirrors training.py pattern)."""
    project_root = os.path.dirname(BASE_DIR)
    ids_dir = os.path.join(project_root, "Intent_detection_system")
    if os.path.isdir(ids_dir):
        return os.path.join(ids_dir, "outputs")
    return os.path.join(BASE_DIR, "outputs")


def _load_pipeline():
    """Load the trained pipeline and embedder (mirrors training.py).

    Returns (pipeline, embedder).
    """
    from multidomain_intent_detection.pipeline import IntentPipeline

    # Register IntentPipeline on __main__ for pickle compat
    import __main__
    if not hasattr(__main__, "IntentPipeline"):
        __main__.IntentPipeline = IntentPipeline

    import pickle

    # Search for model file
    candidates = [
        os.path.join(BASE_DIR, "artifacts", "v3_pipeline.pkl"),
        os.path.join(os.path.dirname(BASE_DIR), "Intent_detection_system", "artifacts", "v3_pipeline.pkl"),
        os.path.join(os.getcwd(), "Intent_detection_system", "artifacts", "v3_pipeline.pkl"),
        os.path.join(os.getcwd(), "artifacts", "v3_pipeline.pkl"),
    ]
    model_path = None
    for path in candidates:
        if os.path.isfile(path):
            model_path = path
            break
    if model_path is None:
        raise FileNotFoundError(
            f"Trained pipeline not found. Searched:\n"
            + "\n".join(f"  - {c}" for c in candidates)
            + "\nRun training first: python -m multidomain_intent_detection.training"
        )

    t0 = time.time()
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    load_ms = (time.time() - t0) * 1000

    embedder = get_embedder()

    print(f"  Pipeline loaded: {len(pipeline.label_names)} intents, "
          f"PCA-{pipeline.n_pca}, {load_ms:.0f}ms")
    print(f"  Model: {model_path}")
    return pipeline, embedder


def load_csv(path: str) -> List[Dict]:
    """Load labeled CSV (Prompt,Intent,domain). Validates headers."""
    df = pd.read_csv(path)
    required = {"Prompt", "Intent", "domain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Expected: Prompt, Intent, domain"
        )
    rows = []
    for idx, row in df.iterrows():
        prompt = str(row["Prompt"]).strip() if pd.notna(row["Prompt"]) else ""
        if not prompt:
            logger.warning(f"Skipping empty row at line {idx + 2}")
            continue
        rows.append({
            "text": prompt,
            "actual_intent": str(row["Intent"]).strip(),
            "actual_domain": str(row["domain"]).strip(),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation (mirrors training.py evaluate() exactly)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    test_data: List[Dict],
    pipeline,
    embedder,
    *,
    use_llm: bool = True,
    conf_t: float = None,
    margin_t: float = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate pipeline with confusion-aware gating.

    Mirrors training.py's evaluate() exactly — same gate logic,
    same metrics, same output format.

    Returns {intent_accuracy, domain_accuracy, gate_stats, df, saved_path}.
    """
    from multidomain_intent_detection.pipeline import CONFUSION_PRONE_INTENTS

    # Load gating thresholds from tuning_config.json
    import json as _json
    _gate_cfg = {}
    _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_config.json")
    if os.path.exists(_cfg_path):
        try:
            with open(_cfg_path) as _f:
                _gate_cfg = _json.load(_f).get("gating", {})
        except (ValueError, IOError):
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

    results = []
    llm_n = 0
    gate_stats = {
        "ensemble_pass": 0,
        "disagreement_to_llm": 0,
        "low_conf_to_llm": 0,
        "confusion_pair_to_llm": 0,
    }
    total = len(test_data)
    t_start = time.time()

    for idx, rec in enumerate(test_data):
        normalized_text = normalize_query(rec["text"])
        vec = np.array(embedder.embed(normalized_text))
        pred = pipeline.predict_single(vec)

        # ── Gating (mirrors classifier.py exactly) ───────────────────
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

        intent_match = rec["actual_intent"] == final
        predicted_domain = INTENT_TO_DOMAIN.get(final, "unknown")
        domain_match = rec["actual_domain"] == predicted_domain

        results.append({
            "Prompt": rec["text"],
            "actual_intent": rec["actual_intent"],
            "predicted_intent": final,
            "intent_match": intent_match,
            "actual_domain": rec["actual_domain"],
            "predicted_domain": predicted_domain,
            "domain_match": domain_match,
            "confidence": pred["confidence"],
            "raw_confidence": pred.get("raw_confidence", pred["confidence"]),
            "margin": pred["margin"],
            "source": src,
            "agreement": pred["agreement"],
            "is_confusion_pair": pred.get("is_confusion_pair", False),
            "gate_reason": gate_reason,
        })

        if verbose:
            status = "✓" if (intent_match and domain_match) else "✗"
            print(
                f"  {status}  {idx + 1:>4}/{total}  "
                f"{rec['actual_intent']:<28} → {final:<28} "
                f"({predicted_domain:>22})  "
                f"conf={pred['confidence']:.2f}  src={src}"
            )

        if (idx + 1) % 50 == 0 and not verbose:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  ... {idx + 1}/{total} ({rate:.1f} rows/sec)")

    elapsed = time.time() - t_start
    df = pd.DataFrame(results)

    # Save results CSV
    mode = "hybrid" if use_llm else "ensemble_only"
    ts = _timestamp()
    outputs_dir = _resolve_outputs_dir()
    os.makedirs(outputs_dir, exist_ok=True)
    saved_path = os.path.join(outputs_dir, f"batch_results_{mode}_{ts}.csv")
    df.to_csv(saved_path, index=False)

    # ── Print report (same format as training.py) ────────────────────
    ia = df["intent_match"].mean() * 100
    da = df["domain_match"].mean() * 100
    errors = df[~df["intent_match"]]
    high_conf_errors = errors[errors["raw_confidence"] >= 0.85]

    print(f"\n{'=' * 70}")
    print(f"  {'ENSEMBLE + LLM FALLBACK' if use_llm else 'ENSEMBLE ONLY'}")
    print(f"{'=' * 70}")
    print(f"  Intent Accuracy    : {ia:.2f}%  ({int(df['intent_match'].sum())}/{total})")
    print(f"  Domain Accuracy    : {da:.2f}%  ({int(df['domain_match'].sum())}/{total})")
    print(f"  Total Errors       : {len(errors)}")
    print(f"  High-conf Errors   : {len(high_conf_errors)} (raw conf >= 0.85)")
    if use_llm:
        ep = (total - llm_n) / total * 100
        print(f"  Ensemble resolved  : {ep:.1f}% ({total - llm_n}/{total})")
        print(f"  LLM calls          : {llm_n} ({llm_n / total * 100:.1f}%)")
    print(f"  Elapsed            : {elapsed:.1f}s ({total / elapsed:.1f} rows/sec)")

    print(f"\n  Gate Statistics:")
    for k, v in gate_stats.items():
        pct = v / total * 100 if total else 0
        print(f"    {k:25s}: {v:>5}  ({pct:5.1f}%)")

    # Per-domain breakdown
    print(f"\n  {'Domain':<28} {'Intent%':>8} {'Domain%':>8} {'Count':>6}")
    print(f"  {'─' * 55}")
    for dom in sorted(df["actual_domain"].unique()):
        s = df[df["actual_domain"] == dom]
        print(f"  {dom:<28} {s['intent_match'].mean() * 100:>6.1f}%"
              f" {s['domain_match'].mean() * 100:>6.1f}% {len(s):>6}")

    # Per-intent breakdown (worst first)
    print(f"\n  Per-Intent Breakdown (worst first):")
    print(f"  {'Intent':<30} {'Accuracy':>9} {'Errors':>7} {'Total':>6}")
    print(f"  {'─' * 55}")
    intent_stats = []
    for intent in sorted(df["actual_intent"].unique()):
        mask = df["actual_intent"] == intent
        sub = df[mask]
        n = len(sub)
        errs = int(n - sub["intent_match"].sum())
        acc = sub["intent_match"].mean() * 100
        intent_stats.append((intent, acc, errs, n))
    intent_stats.sort(key=lambda x: (x[1], -x[2]))
    for intent, acc, errs, n in intent_stats:
        marker = " **" if errs > 0 else ""
        print(f"  {intent:<30} {acc:8.1f}% {errs:>7} {n:>6}{marker}")

    # Top confusion pairs
    if len(errors) > 0:
        print(f"\n  Top Confusion Pairs ({len(errors)} errors):")
        conf = (
            errors.groupby(["actual_intent", "predicted_intent"])
            .size()
            .sort_values(ascending=False)
            .head(15)
        )
        for (a, p), c in conf.items():
            print(f"    {a} → {p}: {c}")

    # LLM-only accuracy
    if use_llm and llm_n > 0:
        lr = df[df["source"] == "llm"]
        llm_acc = lr["intent_match"].mean() * 100
        print(f"\n  LLM-only accuracy: {llm_acc:.1f}% "
              f"({int(lr['intent_match'].sum())}/{llm_n})")

    # Failure details
    if len(errors) > 0:
        print(f"\n  Failure Details ({len(errors)} failures):")
        print(f"  {'─' * 70}")
        for _, row in errors.iterrows():
            query = str(row["Prompt"])[:65]
            print(f"    {query}")
            print(
                f"      Expected: {row['actual_intent']} ({row['actual_domain']})")
            print(
                f"      Got:      {row['predicted_intent']} "
                f"({row['predicted_domain']})  "
                f"conf={row['confidence']:.2f}  margin={row['margin']:.2f}  "
                f"src={row['source']}  gate={row['gate_reason']}")
    else:
        print(f"\n  All {total} test cases PASSED!")

    print(f"\n  Saved → {saved_path}")
    print(f"{'=' * 70}\n")

    return {
        "intent_accuracy": ia,
        "domain_accuracy": da,
        "gate_stats": gate_stats,
        "llm_calls": llm_n,
        "high_conf_errors": len(high_conf_errors),
        "df": df,
        "saved_path": saved_path,
    }


def build_summary_json(m1: Dict, m2: Dict = None) -> Dict[str, Any]:
    """Build JSON-serializable summary from one or two evaluation passes."""
    def _pass_json(m: Dict) -> Dict:
        df = m["df"]
        total = len(df)
        errors_df = df[~df["intent_match"]]

        per_domain = []
        for domain in sorted(df["actual_domain"].unique()):
            sub = df[df["actual_domain"] == domain]
            per_domain.append({
                "domain": domain,
                "intent_accuracy": round(sub["intent_match"].mean() * 100, 2),
                "domain_accuracy": round(sub["domain_match"].mean() * 100, 2),
                "count": len(sub),
            })

        per_intent = []
        for intent in sorted(df["actual_intent"].unique()):
            sub = df[df["actual_intent"] == intent]
            errs = int(len(sub) - sub["intent_match"].sum())
            per_intent.append({
                "intent": intent,
                "domain": INTENT_TO_DOMAIN.get(intent, "unknown"),
                "accuracy": round(sub["intent_match"].mean() * 100, 2),
                "errors": errs,
                "total": len(sub),
            })
        per_intent.sort(key=lambda x: x["accuracy"])

        top_confusions = []
        if len(errors_df) > 0:
            confusion = (
                errors_df.groupby(["actual_intent", "predicted_intent"])
                .size().sort_values(ascending=False).head(15)
            )
            for (a, p), c in confusion.items():
                top_confusions.append({
                    "actual_intent": a, "predicted_intent": p, "count": int(c),
                })

        source_counts = df["source"].value_counts()
        llm_rows = df[df["source"] == "llm"]
        return {
            "intent_accuracy": round(m["intent_accuracy"], 2),
            "domain_accuracy": round(m["domain_accuracy"], 2),
            "total_rows": total,
            "total_errors": int((~df["intent_match"]).sum()),
            "high_conf_errors": m["high_conf_errors"],
            "gate_stats": m["gate_stats"],
            "source_breakdown": {
                "ensemble": int(source_counts.get("ensemble", 0)),
                "llm": int(source_counts.get("llm", 0)),
            },
            "llm_accuracy": (
                round(llm_rows["intent_match"].mean() * 100, 2)
                if len(llm_rows) > 0 else None
            ),
            "per_domain": per_domain,
            "per_intent": per_intent,
            "top_confusions": top_confusions,
            "output_file": m["saved_path"],
        }

    result = {"ensemble_only": _pass_json(m1)}
    if m2:
        result["hybrid"] = _pass_json(m2)
        result["delta"] = round(m2["intent_accuracy"] - m1["intent_accuracy"], 2)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch Intent Classification Test Runner — "
                    "dual-pass evaluation (ensemble-only + hybrid)",
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to labeled CSV (Prompt,Intent,domain)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Run ensemble-only pass (skip LLM hybrid pass)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each row result inline")
    parser.add_argument("--json", action="store_true",
                        help="Print summary as JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # 1. Resolve CSV path
    csv_path = args.csv or _find_default_csv()
    if not os.path.exists(csv_path):
        print(f"  CSV not found: {csv_path}")
        print(f"  Use --csv <path> to specify a labeled CSV file.")
        sys.exit(1)

    # 2. Load data
    test_data = load_csv(csv_path)
    print(f"\n{'=' * 70}")
    print(f"  Batch Test — {os.path.basename(csv_path)}")
    print(f"  {len(test_data)} test queries, "
          f"{len(set(r['actual_intent'] for r in test_data))} intents")
    print(f"{'=' * 70}")

    # 3. Validate intent labels
    unknown_intents = set()
    for row in test_data:
        if row["actual_intent"] not in INTENT_TO_DOMAIN:
            unknown_intents.add(row["actual_intent"])
    if unknown_intents:
        print(f"\n  WARNING: {len(unknown_intents)} unknown intent labels "
              f"not in INTENT_TO_DOMAIN: {sorted(unknown_intents)}")

    # 4. Load pipeline + embedder directly (not through classifier)
    print(f"\n  Loading pipeline and embedder...")
    pipeline, embedder = _load_pipeline()

    # 5. Pass 1 — Ensemble only (always runs)
    print(f"\n{'─' * 70}")
    print(f"  Pass 1 — Ensemble Only")
    print(f"{'─' * 70}")
    m1 = evaluate(
        test_data, pipeline, embedder,
        use_llm=False, verbose=args.verbose,
    )

    # 6. Pass 2 — Hybrid (ensemble + LLM) unless --no-llm
    m2 = None
    if not args.no_llm:
        print(f"\n{'─' * 70}")
        print(f"  Pass 2 — Ensemble + LLM Fallback")
        print(f"{'─' * 70}")
        m2 = evaluate(
            test_data, pipeline, embedder,
            use_llm=True, verbose=args.verbose,
        )

        # Final delta
        print(f"{'=' * 70}")
        print(f"  FINAL: Ensemble {m1['intent_accuracy']:.1f}%"
              f" → +LLM {m2['intent_accuracy']:.1f}%"
              f"  (Δ {m2['intent_accuracy'] - m1['intent_accuracy']:+.1f}%)")
        print(f"{'=' * 70}")

    # 7. JSON output
    if args.json:
        summary = build_summary_json(m1, m2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
