"""
Test Multidomain Intent Detection — Interactive CLI
=====================================================

Takes a prompt from the user and identifies:
  - Intent (e.g. "pricing_info", "Status", "pa_summary")
  - Domain (e.g. "cap_api", "claim_history_search", "override_domain")
  - Confidence score (0.0–1.0)
  - Margin (gap between top-1 and top-2 predictions)
  - Source ("ensemble" or "llm" fallback)
  - Extracted entities (claim_number, sequence_number, NPI, NDC, etc.)
  - API endpoint for the domain
  - Top-5 candidate intents with probabilities
  - Sub-classifier agreement (SVM / LogReg / kNN)

Usage:
    # Interactive mode (REPL)
    python -m test_multidomain_intent_detection

    # Single query
    python -m test_multidomain_intent_detection "What is the copay on claim 132435151040074?"

    # Batch file (one query per line)
    python -m test_multidomain_intent_detection --file queries.txt

    # Run built-in test suite
    python -m test_multidomain_intent_detection --test

    # Show model info
    python -m test_multidomain_intent_detection --info
"""

import sys
import os
import logging
import argparse
import json
from typing import List, Dict, Any

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from multidomain_intent_detection import (
    get_classifier,
    get_all_intents,
    get_all_domains,
    get_intents_for_domain,
    INTENT_TO_DOMAIN,
)
from multidomain_intent_detection.normalizer import normalize_query, extract_entities


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_bar(confidence: float, width: int = 20) -> str:
    filled = int(confidence * width)
    return "█" * filled + "░" * (width - filled)


def print_result(query: str, result: Dict[str, Any], verbose: bool = True):
    """Pretty-print a classification result."""
    bar = _confidence_bar(result["confidence"])
    entities_str = (
        ", ".join(f"{k}={v}" for k, v in result["entities"].items())
        if result["entities"]
        else "—"
    )

    print(f"\n  ┌─ Query: {query}")
    print(f"  │")
    print(f"  ├─ Intent:     {result['intent']}")
    print(f"  ├─ Domain:     {result['domain']} ({result['domain_name']})")
    print(f"  ├─ Confidence: {bar} {result['confidence']:.2%}")
    print(f"  ├─ Margin:     {result['margin']:.4f}")
    print(f"  ├─ Source:     {result['source']}")
    print(f"  ├─ Agreement:  {result['agreement']}")
    print(f"  ├─ Entities:   {entities_str}")
    print(f"  ├─ Endpoint:   {result['api_endpoint'] or '(none)'}")
    print(f"  ├─ Clarify?    {result['needs_clarification']}")
    print(f"  ├─ Latency:    {result['latency_ms']}ms")

    if verbose and result.get("top_5"):
        print(f"  │")
        print(f"  ├─ Top-5 Candidates:")
        for i, (intent, prob) in enumerate(result["top_5"], 1):
            marker = " ◀" if intent == result["intent"] else ""
            domain = INTENT_TO_DOMAIN.get(intent, "?")
            print(f"  │   {i}. {intent:<25} {prob:.4f}  ({domain}){marker}")

    print(f"  └{'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Built-in test suite
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_TEST_QUERIES: List[Dict[str, str]] = [
    # ── cap_api ──────────────────────────────────────────────────────────
    {"query": "What is the copay on claim 132435151040074 sequence 001?", "expected_domain": "cap_api", "expected_intent": "pricing_info"},
    {"query": "Prescriber details for claim 220133725669000 sequence 001.", "expected_domain": "cap_api", "expected_intent": "prescriber_info"},
    {"query": "Show the settlement codes for this claim.", "expected_domain": "cap_api", "expected_intent": "settlement_info"},
    {"query": "Why was this claim rejected?", "expected_domain": "cap_api", "expected_intent": "rejection_reasons"},
    {"query": "Which pharmacy dispensed claim 132435151040074 sequence 001?", "expected_domain": "cap_api", "expected_intent": "pharmacy_info"},
    {"query": "R&R information for claim 242905816136000 sequence 001.", "expected_domain": "cap_api", "expected_intent": "reversal_info"},
    # ── benefits_api ─────────────────────────────────────────────────────
    {"query": "Show the current benefit plan overview for this member.", "expected_domain": "benefits_api", "expected_intent": "plan_summary"},
    {"query": "Display the audit trail of plan changes.", "expected_domain": "benefits_api", "expected_intent": "audit_info"},
    {"query": "When was claim 132435151040074 sequence 001 first created?", "expected_domain": "benefits_api", "expected_intent": "audit_info"},
    # ── claim_history_search ─────────────────────────────────────────────
    {"query": "Show all rejected claims for this member.", "expected_domain": "claim_history_search", "expected_intent": "Status"},
    {"query": "How much did the member pay for METFORMIN across all fills?", "expected_domain": "claim_history_search", "expected_intent": "Pricing"},
    {"query": "List claims filled at CVS PHARMACY 00610.", "expected_domain": "claim_history_search", "expected_intent": "Pharmacy"},
    {"query": "When was ATORVASTATIN last dispensed?", "expected_domain": "claim_history_search", "expected_intent": "DrugLast"},
    {"query": "Show claims with reject code 79.", "expected_domain": "claim_history_search", "expected_intent": "RejectCode"},
    {"query": "Which claims required prior authorization?", "expected_domain": "claim_history_search", "expected_intent": "PriorAuth"},
    {"query": "NDC 33342-0395-44", "expected_domain": "claim_history_search", "expected_intent": "NDC"},
    # ── member_domain ────────────────────────────────────────────────────
    {"query": "Does this member have active coverage as of today?", "expected_domain": "member_domain", "expected_intent": "member_coverage"},
    {"query": "What is the CVS ID for this member?", "expected_domain": "member_domain", "expected_intent": "cvs_id_lookup"},
    {"query": "Is this member LICS?", "expected_domain": "member_domain", "expected_intent": "lics_status"},
    # ── override_domain ──────────────────────────────────────────────────
    {"query": "Will this PA override a reject 75?", "expected_domain": "override_domain", "expected_intent": "pa_override_reject"},
    {"query": "What drugs will this PA cover?", "expected_domain": "override_domain", "expected_intent": "pa_drug_coverage"},
    {"query": "How many claims used this PA?", "expected_domain": "override_domain", "expected_intent": "pa_claim_usage"},
    # ── general ──────────────────────────────────────────────────────────
    {"query": "Hello", "expected_domain": "general", "expected_intent": "greeting"},
    {"query": "What's the weather today?", "expected_domain": "general", "expected_intent": "out_of_scope"},
]


def run_test_suite(classifier, verbose: bool = True):
    """Run the built-in test suite and report accuracy."""
    print("\n" + "=" * 72)
    print("  MULTIDOMAIN INTENT DETECTION — TEST SUITE")
    print("=" * 72)

    intent_correct = 0
    domain_correct = 0
    total = len(BUILTIN_TEST_QUERIES)
    failures: List[Dict] = []

    for tc in BUILTIN_TEST_QUERIES:
        result = classifier.classify(tc["query"])

        i_match = result["intent"] == tc["expected_intent"]
        d_match = result["domain"] == tc["expected_domain"]

        if i_match:
            intent_correct += 1
        if d_match:
            domain_correct += 1

        if verbose:
            status = "✓" if (i_match and d_match) else "✗"
            print(f"  {status}  {tc['query'][:55]:<55}  → {result['intent']:<22} ({result['domain']})")

        if not i_match or not d_match:
            failures.append({
                "query": tc["query"],
                "expected_intent": tc["expected_intent"],
                "predicted_intent": result["intent"],
                "expected_domain": tc["expected_domain"],
                "predicted_domain": result["domain"],
                "confidence": result["confidence"],
            })

    print(f"\n{'─' * 72}")
    print(f"  Intent Accuracy:  {intent_correct}/{total} ({intent_correct / total * 100:.1f}%)")
    print(f"  Domain Accuracy:  {domain_correct}/{total} ({domain_correct / total * 100:.1f}%)")

    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f in failures:
            print(f"    {f['query'][:50]}")
            print(f"      Expected: {f['expected_intent']} ({f['expected_domain']})")
            print(f"      Got:      {f['predicted_intent']} ({f['predicted_domain']}) conf={f['confidence']:.2f}")
    else:
        print(f"\n  All {total} test cases PASSED!")

    print("=" * 72)
    return {"intent_accuracy": intent_correct / total, "domain_accuracy": domain_correct / total}


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction standalone test
# ─────────────────────────────────────────────────────────────────────────────

def test_entity_extraction():
    """Quick sanity check on entity extraction."""
    print("\n" + "=" * 72)
    print("  ENTITY EXTRACTION TEST")
    print("=" * 72)

    test_cases = [
        ("What is the copay on claim 132435151040074 sequence 001?",
         {"claim_number": "132435151040074", "sequence_number": "001"}),
        ("Prescriber NPI 1234567890 for this claim.",
         {"npi": "1234567890"}),
        ("NDC 33342-0395-44 details",
         {"ndc": "33342-0395-44"}),
        ("Show claims with reject code 79.",
         {"reject_code": "79"}),
        ("Claims filled at CVS PHARMACY 00610",
         {"pharmacy_name": "CVS PHARMACY 00610"}),
        ("How much did the member pay for METFORMIN across all fills?",
         {"drug_name": "METFORMIN"}),
        ("Show claims with settlement code 358.",
         {"settlement_code": "358"}),
        ("Claims from last month",
         {"date_reference": "last month"}),
    ]

    passed = 0
    for query, expected in test_cases:
        entities = extract_entities(query)
        match = all(entities.get(k) == v for k, v in expected.items())
        status = "✓" if match else "✗"
        print(f"  {status}  {query[:50]:<50}  → {entities}")
        if not match:
            print(f"      Expected: {expected}")
        if match:
            passed += 1

    print(f"\n  Passed: {passed}/{len(test_cases)}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Model info
# ─────────────────────────────────────────────────────────────────────────────

def show_model_info(classifier):
    """Display model metadata."""
    info = classifier.model_info
    print("\n" + "=" * 72)
    print("  MODEL INFO")
    print("=" * 72)
    print(f"  Intents:       {info['n_intents']}")
    print(f"  Domains:       {info['n_domains']} — {', '.join(info['domains'])}")
    print(f"  PCA dims:      {info['pca_dims']}")
    print(f"  Temperature:   {info['temperature']}")
    print(f"  Weights:       {info['ensemble_weights']}")
    print(f"  Conf thresh:   {info['confidence_threshold']}")
    print(f"  Margin thresh: {info['margin_threshold']}")
    print(f"  LLM fallback:  {info['use_llm_fallback']}")
    print(f"  Model path:    {info['model_path']}")
    if info['load_time_ms']:
        print(f"  Load time:     {info['load_time_ms']}ms")

    print(f"\n  Intents by domain:")
    for domain in sorted(info['domains']):
        intents = get_intents_for_domain(domain)
        print(f"    {domain} ({len(intents)}): {', '.join(intents[:6])}{'…' if len(intents) > 6 else ''}")

    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────────────────────────────────────

def interactive_repl(classifier):
    """Interactive loop — type queries, see classification results."""
    print("\n" + "=" * 72)
    print("  MULTIDOMAIN INTENT DETECTION — Interactive Mode")
    print("=" * 72)
    print("  Type a query to classify.  Commands:")
    print("    :info    — Show model information")
    print("    :test    — Run built-in test suite")
    print("    :entity  — Test entity extraction")
    print("    :json    — Toggle JSON output")
    print("    :quit    — Exit")
    print("=" * 72)

    json_mode = False

    while True:
        try:
            query = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not query:
            continue

        if query.lower() in (":quit", ":exit", ":q"):
            print("  Goodbye!")
            break
        elif query.lower() == ":info":
            show_model_info(classifier)
            continue
        elif query.lower() == ":test":
            run_test_suite(classifier)
            continue
        elif query.lower() == ":entity":
            test_entity_extraction()
            continue
        elif query.lower() == ":json":
            json_mode = not json_mode
            print(f"  JSON output: {'ON' if json_mode else 'OFF'}")
            continue

        result = classifier.classify(query)

        if json_mode:
            # Serializable copy
            output = dict(result)
            output["top_5"] = [{"intent": n, "probability": p} for n, p in result["top_5"]]
            print(json.dumps(output, indent=2))
        else:
            print_result(query, result)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multidomain Intent Detection — Classify PBM queries",
    )
    parser.add_argument("query", nargs="?", help="Single query to classify")
    parser.add_argument("--test", action="store_true", help="Run built-in test suite")
    parser.add_argument("--info", action="store_true", help="Show model info")
    parser.add_argument("--entity", action="store_true", help="Test entity extraction")
    parser.add_argument("--file", type=str, help="Classify queries from a file (one per line)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM fallback")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    classifier = get_classifier(use_llm_fallback=not args.no_llm)

    if args.entity:
        test_entity_extraction()
        return

    if args.info:
        show_model_info(classifier)
        return

    if args.test:
        run_test_suite(classifier, verbose=args.verbose)
        return

    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            sys.exit(1)
        with open(args.file) as f:
            queries = [line.strip() for line in f if line.strip()]
        results = []
        for q in queries:
            result = classifier.classify(q)
            if args.json:
                output = dict(result)
                output["top_5"] = [{"intent": n, "probability": p} for n, p in result["top_5"]]
                results.append(output)
            else:
                print_result(q, result, verbose=args.verbose)
        if args.json:
            print(json.dumps(results, indent=2))
        return

    if args.query:
        result = classifier.classify(args.query)
        if args.json:
            output = dict(result)
            output["top_5"] = [{"intent": n, "probability": p} for n, p in result["top_5"]]
            print(json.dumps(output, indent=2))
        else:
            print_result(args.query, result)
        return

    # No args → interactive REPL
    interactive_repl(classifier)


if __name__ == "__main__":
    main()
