"""
Multidomain Intent Detection — LLM-Only Evaluation
====================================================

Every query is sent to the LLM regardless of ensemble confidence.
The ensemble is still used to compute the top-5 candidate list that
provides context to the LLM, but it never makes the final decision.

Differences from training.py:
  - No gating logic (confident / disagreement / confusion checks skipped)
  - LLM called for every single query
  - Thinking always captured (capture_thinking=True)
  - Results streamed to CSV row-by-row; file is always readable even if
    the run is interrupted
  - Loads an already-trained pipeline from artifacts/v3_pipeline.pkl
    (run training.py first if the file doesn't exist)

Run:
    python -m multidomain_intent_detection.training_llm_only
"""

import os
import sys
import csv
import json
import pickle
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from multidomain_intent_detection.config import INTENT_TO_DOMAIN
from multidomain_intent_detection.normalizer import normalize_query
from multidomain_intent_detection.embeddings import get_embedder
from multidomain_intent_detection.llm_fallback import llm_classify

logger = logging.getLogger(__name__)

# ── Paths (mirrors training.py) ───────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_IDS_DIR  = os.path.join(os.path.dirname(BASE_DIR), "Intent_detection_system")

ARTIFACTS = (
    os.path.join(_IDS_DIR, "artifacts")
    if os.path.isdir(os.path.join(_IDS_DIR, "artifacts"))
    else os.path.join(BASE_DIR, "artifacts")
)
OUTPUTS = (
    os.path.join(_IDS_DIR, "outputs")
    if os.path.isdir(_IDS_DIR)
    else os.path.join(BASE_DIR, "outputs")
)
MODEL_PKL = os.path.join(ARTIFACTS, "v3_pipeline.pkl")

os.makedirs(OUTPUTS, exist_ok=True)

# ── CSV schema ────────────────────────────────────────────────────────────────
_FIELDNAMES = [
    "text", "actual_intent", "predicted_intent", "intent_match",
    "actual_domain", "predicted_domain", "domain_match",
    # Ensemble signal (context only — not used for the decision)
    "ensemble_intent", "ensemble_confidence", "ensemble_margin", "agreement",
    # Top-5 candidates passed to the LLM
    "top1_intent", "top1_conf", "top1_domain",
    "top2_intent", "top2_conf", "top2_domain",
    "top3_intent", "top3_conf", "top3_domain",
    "top4_intent", "top4_conf", "top4_domain",
    "top5_intent", "top5_conf", "top5_domain",
    # LLM output
    "llm_source",
    "llm_reasoning",
    "llm_thinking",
]


def _timestamp() -> str:
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_llm_only(test_data, pipeline, embedder) -> dict:
    """
    Run LLM-only evaluation.  Every query goes to the LLM; results are
    written to the CSV immediately after each query is processed.

    Returns a summary dict: {intent_accuracy, domain_accuracy, csv_path}.
    """
    ts = _timestamp()
    csv_path = os.path.join(OUTPUTS, f"results_llm_only_{ts}.csv")

    n_total   = len(test_data)
    n_correct_intent = 0
    n_correct_domain = 0
    n_llm_errors     = 0

    print(f"\n  Output  : {csv_path}")
    print(f"  Queries : {n_total}\n")
    print(f"  {'#':>5}  {'OK':>2}  {'Actual intent':<32} {'Predicted intent':<32}  {'Acc':>5}")
    print(f"  {'-'*90}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        f.flush()

        for idx, rec in enumerate(test_data):
            # ── Embed + ensemble top-5 (context for LLM only) ────────────
            normalized = normalize_query(rec["text"])
            vec  = np.array(embedder.embed(normalized))
            pred = pipeline.predict_single(vec)

            candidates = [name for name, _ in pred["top_5"]]

            # ── Always call LLM ───────────────────────────────────────────
            llm_result = llm_classify(
                rec["text"],
                candidates,
                ensemble_intent=pred["intent"],
                ensemble_confidence=pred["confidence"],
                return_details=True,
                capture_thinking=True,
            )

            final          = llm_result["intent"]
            llm_reasoning  = llm_result.get("reasoning")
            llm_thinking   = llm_result.get("thinking")
            llm_source     = llm_result.get("source", "llm")

            if llm_source.endswith("_error"):
                n_llm_errors += 1

            predicted_domain = INTENT_TO_DOMAIN.get(final, "unknown")
            intent_match     = rec["actual_intent"] == final
            domain_match     = rec["actual_domain"] == predicted_domain

            if intent_match:
                n_correct_intent += 1
            if domain_match:
                n_correct_domain += 1

            # ── Flatten top-5 columns ─────────────────────────────────────
            top5_cols = {}
            for rank, (t_intent, t_conf) in enumerate(pred.get("top_5", []), start=1):
                top5_cols[f"top{rank}_intent"] = t_intent
                top5_cols[f"top{rank}_conf"]   = round(t_conf, 4)
                top5_cols[f"top{rank}_domain"]  = INTENT_TO_DOMAIN.get(t_intent, "unknown")

            # ── Build row and stream to CSV ───────────────────────────────
            row = {
                "text":                rec["text"],
                "actual_intent":       rec["actual_intent"],
                "predicted_intent":    final,
                "intent_match":        intent_match,
                "actual_domain":       rec["actual_domain"],
                "predicted_domain":    predicted_domain,
                "domain_match":        domain_match,
                "ensemble_intent":     pred["intent"],
                "ensemble_confidence": round(pred["confidence"], 4),
                "ensemble_margin":     round(pred["margin"], 4),
                "agreement":           pred["agreement"],
                **top5_cols,
                "llm_source":          llm_source,
                "llm_reasoning":       llm_reasoning,
                "llm_thinking":        llm_thinking,
            }
            writer.writerow(row)
            f.flush()

            # ── Per-row console progress ──────────────────────────────────
            running_acc = n_correct_intent / (idx + 1) * 100
            ok_mark = "✓" if intent_match else "✗"
            print(
                f"  {idx+1:>5}  {ok_mark:>2}  "
                f"{rec['actual_intent']:<32} {final:<32}  {running_acc:>4.1f}%"
            )

    # ── Final summary ─────────────────────────────────────────────────────────
    ia = n_correct_intent / n_total * 100
    da = n_correct_domain / n_total * 100

    print(f"\n{'=' * 65}")
    print(f"  LLM-ONLY EVALUATION RESULTS")
    print(f"  Intent Accuracy : {ia:.2f}%  ({n_correct_intent}/{n_total})")
    print(f"  Domain Accuracy : {da:.2f}%  ({n_correct_domain}/{n_total})")
    print(f"  LLM Errors      : {n_llm_errors}  (fell back to ensemble top-1)")
    print(f"  Output CSV      : {csv_path}")
    print(f"{'=' * 65}\n")

    # Per-domain breakdown
    df = pd.read_csv(csv_path)
    print(f"  {'Domain':<28} {'Intent acc':>10} {'Domain acc':>10} {'Count':>6}")
    print(f"  {'-'*58}")
    for dom in sorted(df["actual_domain"].unique()):
        s = df[df["actual_domain"] == dom]
        print(
            f"  {dom:<28} "
            f"{s['intent_match'].mean()*100:>8.1f}%  "
            f"{s['domain_match'].mean()*100:>8.1f}%  "
            f"{len(s):>6}"
        )

    errors = df[~df["intent_match"]]
    if len(errors) > 0:
        print(f"\n  Top Confusions:")
        conf = (
            errors.groupby(["actual_intent", "predicted_intent"])
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        for (a, p), c in conf.items():
            print(f"    {a} → {p}: {c}")

    return {"intent_accuracy": ia, "domain_accuracy": da, "csv_path": csv_path}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 65)
    print("  Multidomain Intent Detection — LLM-Only Evaluation")
    print("  Every query → LLM  (ensemble used for top-5 context only)")
    print("=" * 65)

    # ── Load test data ────────────────────────────────────────────────────────
    _project_root = os.path.dirname(BASE_DIR)
    for candidate in [
        os.path.join(_project_root, "combinedtestset.csv"),
        os.path.join(_IDS_DIR, "combinedtestset.csv"),
        os.path.join(_IDS_DIR, "testingFinalDataset.csv"),
        os.path.join(BASE_DIR, "testingFinalDataset.csv"),
    ]:
        if os.path.exists(candidate):
            TESTDATA = candidate
            break
    else:
        print("No test CSV found (combinedtestset.csv or testingFinalDataset.csv).")
        sys.exit(1)

    print(f"\nTest data : {os.path.basename(TESTDATA)}")
    tdf = pd.read_csv(TESTDATA)

    col_text   = "prompt"           if "prompt"           in tdf.columns else "Prompt"
    col_intent = "expected_intent"  if "expected_intent"  in tdf.columns else "Intent"
    col_domain = "expected_domain"  if "expected_domain"  in tdf.columns else "domain"

    test_data = [
        {
            "text":          r[col_text],
            "actual_intent": r[col_intent],
            "actual_domain": r[col_domain],
        }
        for _, r in tdf.iterrows()
    ]
    print(f"  {len(test_data)} queries, {tdf[col_intent].nunique()} unique intents")

    # ── Load trained pipeline ─────────────────────────────────────────────────
    if not os.path.exists(MODEL_PKL):
        print(f"\nPipeline not found at {MODEL_PKL}")
        print("Run first:  python -m multidomain_intent_detection.training")
        sys.exit(1)

    print(f"\nLoading pipeline: {MODEL_PKL}")
    with open(MODEL_PKL, "rb") as f:
        pipeline = pickle.load(f)

    embedder = get_embedder()

    # ── Run evaluation ────────────────────────────────────────────────────────
    evaluate_llm_only(test_data, pipeline, embedder)


if __name__ == "__main__":
    main()
