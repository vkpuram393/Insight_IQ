"""
Multidomain Intent Detection — Diagnostics & Auto-Fix Recommendations
======================================================================

Analyzes batch test results and generates actionable recommendations
for improving accuracy and confidence.  Instead of manually analyzing
the HTML report and editing code, run this tool after each batch test.

What it does:
  1. Loads batch test result CSV (from batch_test.py or training.py)
  2. Identifies worst-performing intents
  3. Detects undiscovered confusion pairs
  4. Analyzes confidence distribution and calibration
  5. Generates prioritized action items
  6. Optionally auto-patches: updates CONFUSION_PAIRS, augmented_examples
  7. Outputs a concise report with exact next steps

Usage:
    # Analyze latest batch results
    python -m multidomain_intent_detection.diagnostics

    # Analyze a specific results CSV
    python -m multidomain_intent_detection.diagnostics --csv outputs/batch_results_hybrid_20260501.csv

    # Auto-patch: update confusion pairs and generate augmented examples
    python -m multidomain_intent_detection.diagnostics --auto-fix

    # Set minimum acceptable accuracy threshold
    python -m multidomain_intent_detection.diagnostics --min-accuracy 90
"""

import os
import sys
import json
import glob
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_latest_results() -> Optional[str]:
    """Find the most recent batch results CSV."""
    search_dirs = [
        os.path.join(BASE_DIR, "outputs"),
        os.path.join(os.path.dirname(BASE_DIR), "Intent_detection_system", "outputs"),
    ]
    all_csvs = []
    for d in search_dirs:
        all_csvs.extend(glob.glob(os.path.join(d, "batch_results_*.csv")))
        all_csvs.extend(glob.glob(os.path.join(d, "results_*.csv")))
    if not all_csvs:
        return None
    # Sort by modification time, newest first
    all_csvs.sort(key=os.path.getmtime, reverse=True)
    return all_csvs[0]


def load_results(csv_path: str) -> pd.DataFrame:
    """Load batch results CSV."""
    df = pd.read_csv(csv_path)
    # Normalize column names (batch_test uses "Prompt", training uses "text")
    if "Prompt" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"Prompt": "text"})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────

def analyze_overall(df: pd.DataFrame) -> Dict:
    """Overall accuracy and confidence stats."""
    total = len(df)
    intent_correct = df["intent_match"].sum()
    domain_correct = df["domain_match"].sum()

    # Confidence stats
    confs = df["confidence"].values
    correct_confs = df[df["intent_match"]]["confidence"].values
    wrong_confs = df[~df["intent_match"]]["confidence"].values

    return {
        "total": total,
        "intent_accuracy": round(intent_correct / total * 100, 2),
        "domain_accuracy": round(domain_correct / total * 100, 2),
        "intent_errors": int(total - intent_correct),
        "domain_errors": int(total - domain_correct),
        "confidence_mean": round(float(confs.mean()), 4),
        "confidence_median": round(float(np.median(confs)), 4),
        "confidence_p10": round(float(np.percentile(confs, 10)), 4),
        "confidence_p90": round(float(np.percentile(confs, 90)), 4),
        "correct_conf_mean": round(float(correct_confs.mean()), 4) if len(correct_confs) > 0 else 0,
        "wrong_conf_mean": round(float(wrong_confs.mean()), 4) if len(wrong_confs) > 0 else 0,
        "high_conf_errors": int((df[~df["intent_match"]]["confidence"] >= 0.80).sum()),
        "low_conf_correct": int((df[df["intent_match"]]["confidence"] < 0.30).sum()),
        "pct_above_80": round(float((confs >= 0.80).mean() * 100), 1),
        "pct_above_60": round(float((confs >= 0.60).mean() * 100), 1),
    }


def analyze_worst_intents(df: pd.DataFrame, min_accuracy: float = 90.0) -> List[Dict]:
    """Identify intents performing below threshold."""
    results = []
    for intent in sorted(df["actual_intent"].unique()):
        mask = df["actual_intent"] == intent
        sub = df[mask]
        n = len(sub)
        correct = int(sub["intent_match"].sum())
        acc = correct / n * 100 if n > 0 else 0
        avg_conf = float(sub["confidence"].mean())
        med_conf = float(sub["confidence"].median())

        if acc < min_accuracy:
            # Find what it's being confused with
            errors = sub[~sub["intent_match"]]
            confused_with = {}
            if len(errors) > 0:
                for _, row in errors.iterrows():
                    pred = row["predicted_intent"]
                    confused_with[pred] = confused_with.get(pred, 0) + 1

            results.append({
                "intent": intent,
                "domain": sub["actual_domain"].iloc[0] if "actual_domain" in sub.columns else "unknown",
                "accuracy": round(acc, 1),
                "correct": correct,
                "total": n,
                "avg_confidence": round(avg_conf, 4),
                "median_confidence": round(med_conf, 4),
                "confused_with": dict(sorted(confused_with.items(), key=lambda x: -x[1])),
            })

    results.sort(key=lambda x: x["accuracy"])
    return results


def analyze_confusion_pairs(df: pd.DataFrame, min_count: int = 2) -> List[Dict]:
    """Detect systematic confusion pairs from errors."""
    errors = df[~df["intent_match"]]
    if len(errors) == 0:
        return []

    pairs = (
        errors.groupby(["actual_intent", "predicted_intent"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    results = []
    for _, row in pairs.iterrows():
        if row["count"] < min_count:
            continue
        actual = row["actual_intent"]
        predicted = row["predicted_intent"]

        # Check if this is cross-domain
        actual_domain = errors[errors["actual_intent"] == actual]["actual_domain"].iloc[0] if "actual_domain" in errors.columns else ""
        pred_domain = errors[errors["predicted_intent"] == predicted]["predicted_domain"].iloc[0] if "predicted_domain" in errors.columns else ""
        is_cross_domain = actual_domain != pred_domain

        results.append({
            "actual": actual,
            "predicted": predicted,
            "count": int(row["count"]),
            "actual_domain": actual_domain,
            "predicted_domain": pred_domain,
            "is_cross_domain": is_cross_domain,
        })
    return results


def analyze_confidence_bands(df: pd.DataFrame) -> List[Dict]:
    """Break down accuracy by confidence band."""
    bands = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    results = []
    for lo, hi in bands:
        mask = (df["confidence"] >= lo) & (df["confidence"] < hi)
        sub = df[mask]
        if len(sub) == 0:
            continue
        results.append({
            "band": f"{lo:.0%}-{hi:.0%}",
            "count": len(sub),
            "pct_of_total": round(len(sub) / len(df) * 100, 1),
            "intent_accuracy": round(sub["intent_match"].mean() * 100, 1),
            "domain_accuracy": round(sub["domain_match"].mean() * 100, 1),
        })
    return results


def analyze_gate_stats(df: pd.DataFrame) -> Dict:
    """Analyze LLM gating effectiveness."""
    if "gate_reason" not in df.columns or "source" not in df.columns:
        return {}

    gate_dist = df["gate_reason"].value_counts().to_dict()
    source_dist = df["source"].value_counts().to_dict()

    # LLM effectiveness
    llm_rows = df[df["source"] == "llm"]
    ensemble_rows = df[df["source"] == "ensemble"]

    return {
        "gate_distribution": gate_dist,
        "source_distribution": source_dist,
        "ensemble_accuracy": round(ensemble_rows["intent_match"].mean() * 100, 2) if len(ensemble_rows) > 0 else 0,
        "llm_accuracy": round(llm_rows["intent_match"].mean() * 100, 2) if len(llm_rows) > 0 else 0,
        "llm_call_rate": round(len(llm_rows) / len(df) * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Auto-Fix Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_confusion_pair_patch(confusion_pairs: List[Dict]) -> str:
    """Generate Python code to add to CONFUSION_PAIRS in pipeline.py."""
    lines = ["# ── AUTO-GENERATED: New confusion pairs from diagnostics ──"]
    existing = set()

    for pair in confusion_pairs:
        a, b = pair["actual"], pair["predicted"]
        if (a, b) in existing or (b, a) in existing:
            continue
        existing.add((a, b))
        lines.append(f'    "{a}": {{"{b}"}},  # {pair["count"]}x confusion')
        lines.append(f'    "{b}": {{"{a}"}},  # reverse')

    return "\n".join(lines)


def generate_augmentation_suggestions(worst_intents: List[Dict]) -> List[Dict]:
    """Generate training data augmentation suggestions for worst intents."""
    suggestions = []
    for item in worst_intents:
        if item["accuracy"] >= 90:
            continue

        confused_with = item.get("confused_with", {})
        top_confusion = list(confused_with.keys())[:3] if confused_with else []

        priority = "CRITICAL" if item["accuracy"] < 50 else \
                   "HIGH" if item["accuracy"] < 70 else \
                   "MEDIUM" if item["accuracy"] < 80 else "LOW"

        suggestion = {
            "intent": item["intent"],
            "domain": item["domain"],
            "accuracy": item["accuracy"],
            "priority": priority,
            "action": f"Add 10-15 augmented training examples to AUGMENTED_EXAMPLES['{item['intent']}']",
            "confused_with": top_confusion,
            "guidance": [],
        }

        if top_confusion:
            suggestion["guidance"].append(
                f"Focus on phrases that DISTINGUISH '{item['intent']}' from "
                f"{', '.join(repr(c) for c in top_confusion)}"
            )

        if item["median_confidence"] < 0.30:
            suggestion["guidance"].append(
                "Very low confidence — embeddings are not separating well. "
                "Add more diverse example phrasings."
            )

        suggestions.append(suggestion)

    return suggestions


# ─────────────────────────────────────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    csv_path: str,
    overall: Dict,
    worst_intents: List[Dict],
    confusion_pairs: List[Dict],
    conf_bands: List[Dict],
    gate_stats: Dict,
    augmentation_suggestions: List[Dict],
    min_accuracy: float = 90.0,
):
    """Print human-readable diagnostic report."""
    print(f"\n{'=' * 70}")
    print(f"  INTENT DETECTION DIAGNOSTICS REPORT")
    print(f"  Source: {os.path.basename(csv_path)}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    # ── Overall ──────────────────────────────────────────────────────
    print(f"\n  OVERALL PERFORMANCE")
    print(f"  {'─' * 50}")
    print(f"  Intent Accuracy    : {overall['intent_accuracy']}%  ({overall['total'] - overall['intent_errors']}/{overall['total']})")
    print(f"  Domain Accuracy    : {overall['domain_accuracy']}%")
    print(f"  Total Errors       : {overall['intent_errors']}")

    # Status indicator
    if overall["intent_accuracy"] >= 95:
        status = "EXCELLENT"
    elif overall["intent_accuracy"] >= 90:
        status = "GOOD"
    elif overall["intent_accuracy"] >= 85:
        status = "NEEDS IMPROVEMENT"
    else:
        status = "POOR — ACTION REQUIRED"
    print(f"  Status             : {status}")

    # ── Confidence Quality ───────────────────────────────────────────
    print(f"\n  CONFIDENCE QUALITY")
    print(f"  {'─' * 50}")
    print(f"  Mean confidence    : {overall['confidence_mean']}")
    print(f"  Median confidence  : {overall['confidence_median']}")
    print(f"  P10 / P90          : {overall['confidence_p10']} / {overall['confidence_p90']}")
    print(f"  Correct avg conf   : {overall['correct_conf_mean']}")
    print(f"  Wrong avg conf     : {overall['wrong_conf_mean']}")
    print(f"  % above 80%        : {overall['pct_above_80']}%")
    print(f"  % above 60%        : {overall['pct_above_60']}%")
    print(f"  High-conf errors   : {overall['high_conf_errors']} (conf >= 80% but wrong)")
    print(f"  Low-conf correct   : {overall['low_conf_correct']} (conf < 30% but correct)")

    # Production readiness assessment
    if overall["pct_above_80"] >= 70 and overall["high_conf_errors"] <= 5:
        prod_status = "PRODUCTION READY"
    elif overall["pct_above_60"] >= 80:
        prod_status = "NEEDS CALIBRATION"
    else:
        prod_status = "NOT PRODUCTION READY — confidence too low"
    print(f"\n  Production Status  : {prod_status}")

    # ── Confidence Bands ─────────────────────────────────────────────
    print(f"\n  CONFIDENCE BAND ANALYSIS")
    print(f"  {'─' * 50}")
    print(f"  {'Band':<12} {'Count':>6} {'% Total':>8} {'Intent%':>9} {'Domain%':>9}")
    for band in conf_bands:
        print(f"  {band['band']:<12} {band['count']:>6} {band['pct_of_total']:>7.1f}% "
              f"{band['intent_accuracy']:>8.1f}% {band['domain_accuracy']:>8.1f}%")

    # ── Gate Stats ───────────────────────────────────────────────────
    if gate_stats:
        print(f"\n  LLM GATING ANALYSIS")
        print(f"  {'─' * 50}")
        print(f"  Ensemble accuracy  : {gate_stats.get('ensemble_accuracy', 0)}%")
        print(f"  LLM accuracy       : {gate_stats.get('llm_accuracy', 0)}%")
        print(f"  LLM call rate      : {gate_stats.get('llm_call_rate', 0)}%")
        if gate_stats.get("gate_distribution"):
            print(f"  Gate distribution:")
            for gate, count in gate_stats["gate_distribution"].items():
                print(f"    {gate:25s}: {count}")

    # ── Worst Intents ────────────────────────────────────────────────
    if worst_intents:
        print(f"\n  WORST PERFORMING INTENTS (< {min_accuracy}% accuracy)")
        print(f"  {'─' * 60}")
        print(f"  {'Intent':<30} {'Acc':>6} {'N':>4} {'Domain':<22} {'Confused With'}")
        for item in worst_intents[:15]:
            confused_str = ", ".join(
                f"{k}({v})" for k, v in list(item["confused_with"].items())[:3]
            )
            print(f"  {item['intent']:<30} {item['accuracy']:>5.1f}% {item['total']:>4} "
                  f"{item['domain']:<22} {confused_str}")

    # ── Top Confusion Pairs ──────────────────────────────────────────
    if confusion_pairs:
        print(f"\n  TOP CONFUSION PAIRS")
        print(f"  {'─' * 60}")
        print(f"  {'Actual':<25} {'Predicted':<25} {'Count':>5} {'Cross-Domain':>13}")
        for pair in confusion_pairs[:12]:
            xd = "YES" if pair["is_cross_domain"] else ""
            print(f"  {pair['actual']:<25} {pair['predicted']:<25} {pair['count']:>5} {xd:>13}")

    # ── Action Items ─────────────────────────────────────────────────
    if augmentation_suggestions:
        print(f"\n  ACTION ITEMS (sorted by priority)")
        print(f"  {'─' * 60}")
        for i, s in enumerate(augmentation_suggestions[:10], 1):
            print(f"\n  {i}. [{s['priority']}] {s['intent']} ({s['accuracy']}% accuracy)")
            print(f"     Domain: {s['domain']}")
            print(f"     Action: {s['action']}")
            for g in s["guidance"]:
                print(f"     -> {g}")

    # ── New Confusion Pairs to Add ───────────────────────────────────
    new_pairs = [p for p in confusion_pairs if p["count"] >= 2]
    if new_pairs:
        print(f"\n  SUGGESTED CONFUSION_PAIRS ADDITIONS")
        print(f"  {'─' * 60}")
        print(f"  Add these to pipeline.py CONFUSION_PAIRS dict:")
        for pair in new_pairs[:10]:
            print(f'    "{pair["actual"]}": {{"{pair["predicted"]}"}},')
            print(f'    "{pair["predicted"]}": {{"{pair["actual"]}"}},')

    print(f"\n{'=' * 70}")
    print(f"  NEXT STEPS:")
    print(f"    1. Add training examples for worst intents (see Action Items)")
    print(f"    2. Add confusion pairs to pipeline.py")
    print(f"    3. Run: python -m multidomain_intent_detection.auto_tuner --retrain")
    print(f"    4. Re-test: python -m multidomain_intent_detection.batch_test")
    print(f"{'=' * 70}\n")


def save_report_json(
    csv_path: str,
    overall: Dict,
    worst_intents: List[Dict],
    confusion_pairs: List[Dict],
    conf_bands: List[Dict],
    gate_stats: Dict,
    augmentation_suggestions: List[Dict],
) -> str:
    """Save diagnostic report as JSON for programmatic consumption."""
    report = {
        "source_csv": csv_path,
        "generated_at": datetime.now().isoformat(),
        "overall": overall,
        "worst_intents": worst_intents,
        "confusion_pairs": confusion_pairs,
        "confidence_bands": conf_bands,
        "gate_stats": gate_stats,
        "action_items": augmentation_suggestions,
    }
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    out_path = os.path.join(outputs_dir, f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return out_path


def auto_fix_confusion_pairs(confusion_pairs: List[Dict]):
    """Automatically add new confusion pairs to pipeline.py."""
    from multidomain_intent_detection.pipeline import CONFUSION_PAIRS

    new_pairs_added = 0
    pairs_to_add = {}

    for pair in confusion_pairs:
        if pair["count"] < 2:
            continue
        a, b = pair["actual"], pair["predicted"]

        # Check if already exists
        existing_set = CONFUSION_PAIRS.get(a, set())
        if b in existing_set:
            continue

        if a not in pairs_to_add:
            pairs_to_add[a] = set()
        pairs_to_add[a].add(b)

        if b not in pairs_to_add:
            pairs_to_add[b] = set()
        pairs_to_add[b].add(a)
        new_pairs_added += 1

    if new_pairs_added == 0:
        print("  No new confusion pairs to add.")
        return

    # Read pipeline.py and patch
    pipeline_path = os.path.join(BASE_DIR, "pipeline.py")
    with open(pipeline_path, "r") as f:
        content = f.read()

    # Find the CONFUSION_PAIRS dict end (closing brace before CONFUSION_PRONE_INTENTS)
    marker = "}\n\nCONFUSION_PRONE_INTENTS"
    if marker not in content:
        print("  WARNING: Could not find insertion point in pipeline.py")
        print("  Add these manually:")
        for intent, partners in pairs_to_add.items():
            print(f'    "{intent}": {partners},')
        return

    # Build new entries
    new_entries = []
    for intent, partners in sorted(pairs_to_add.items()):
        partner_str = ", ".join(f'"{p}"' for p in sorted(partners))
        new_entries.append(f'    "{intent}":' + " " * max(1, 20 - len(intent)) + f'{{{partner_str}}},  # auto-added by diagnostics')

    insert_point = content.index(marker)
    patched = (
        content[:insert_point]
        + "    # ── AUTO-ADDED by diagnostics ──\n"
        + "\n".join(new_entries) + "\n"
        + content[insert_point:]
    )

    with open(pipeline_path, "w") as f:
        f.write(patched)

    print(f"  Added {new_pairs_added} new confusion pairs to pipeline.py")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose intent detection performance and generate fixes",
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to batch results CSV")
    parser.add_argument("--min-accuracy", type=float, default=90.0,
                        help="Minimum acceptable intent accuracy (default: 90)")
    parser.add_argument("--auto-fix", action="store_true",
                        help="Auto-patch confusion pairs in pipeline.py")
    parser.add_argument("--json", action="store_true",
                        help="Also save report as JSON")
    args = parser.parse_args()

    # Find CSV
    csv_path = args.csv
    if csv_path is None:
        csv_path = _find_latest_results()
    if csv_path is None or not os.path.exists(csv_path):
        print(f"  No results CSV found. Run batch_test first:")
        print(f"    python -m multidomain_intent_detection.batch_test")
        sys.exit(1)

    print(f"  Loading: {csv_path}")
    df = load_results(csv_path)

    # Run analyses
    overall = analyze_overall(df)
    worst_intents = analyze_worst_intents(df, min_accuracy=args.min_accuracy)
    confusion_pairs = analyze_confusion_pairs(df)
    conf_bands = analyze_confidence_bands(df)
    gate_stats = analyze_gate_stats(df)
    augmentation_suggestions = generate_augmentation_suggestions(worst_intents)

    # Print report
    print_report(
        csv_path, overall, worst_intents, confusion_pairs,
        conf_bands, gate_stats, augmentation_suggestions,
        min_accuracy=args.min_accuracy,
    )

    # Save JSON if requested
    if args.json:
        json_path = save_report_json(
            csv_path, overall, worst_intents, confusion_pairs,
            conf_bands, gate_stats, augmentation_suggestions,
        )
        print(f"  JSON report saved: {json_path}")

    # Auto-fix if requested
    if args.auto_fix:
        print(f"\n  Auto-fixing confusion pairs...")
        auto_fix_confusion_pairs(confusion_pairs)


if __name__ == "__main__":
    main()
