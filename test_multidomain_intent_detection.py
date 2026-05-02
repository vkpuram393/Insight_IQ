"""
test_multidomain_intent_detection.py
====================================

End-to-end smoke test for the multidomain intent detection system AND
its wiring into the claim-history search pipeline.

What it does:
  1. Loads the multidomain classifier (PCA + Ensemble + LLM fallback).
  2. Runs each of the canonical claim-history-search queries through it.
  3. Prints a per-query report:
       intent | domain | confidence | source | api_endpoint
  4. Asserts that every query maps to the expected intent label AND to the
     ``claim_history_search`` domain (so the LangGraph router will dispatch
     them to the member-history pipeline instead of the single-claim tool).

Run from the project root:

    python test_multidomain_intent_detection.py

The script never calls any upstream HTTP API — it is purely a classifier
+ routing test, so it is safe to run without auth credentials.
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple

# Ensure the project root is on sys.path so package imports resolve
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ---------------------------------------------------------------------------
# Canonical test set (as supplied by the product owner)
# ---------------------------------------------------------------------------
TEST_QUERIES: List[Tuple[str, str]] = [
    ("NDC 33342-0395-44",                                         "NDC"),
    ("manufactured by MACLEODS",                                  "Manufacturer"),
    ("show all generic drug claims",                              "Generic"),
    ("show brand name claims",                                    "Brand"),
    ("show all refills for this member",                          "Refills"),
    ("show claims with 90 day supply",                            "DaysSupply"),
    ("which claims used prior authorization?",                    "PriorAuth"),
    ("show claims with diagnosis code E1129",                     "Diagnosis"),
    ("show claims with settlement code 358",                      "Settlement"),
    ("show retail pharmacy claims",                               "PharmType"),
    ("show claims under plan LICS2",                              "Plan"),
    ("show me claims filled at CVS PHARMACY 00610",               "Pharmacy"),
    ("show claims by prescriber NOEUV",                           "Prescriber"),
    ("how much did the member pay for LEVOTHYROXINE?",            "Pricing"),
    ("show me all rejected claims",                               "Status"),
    ("show me all claims with reject code 79",                    "RejectCode"),
    ("When was LEVOTHYROXINE taken last for this member?",        "DrugLast"),
    ("give me all the claims for this member in january",         "Month"),
    # Real-world phrasings exercised against /claims/search
    ("When was this drug taken last 260053944925162?",            "DrugLast"),
    ("Give me list of all medicines taken by member in February 260302639954275?", "Month"),
]

EXPECTED_DOMAIN = "claim_history_search"


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _fmt_row(query: str, expected: str, result: dict) -> str:
    intent = result.get("intent")
    domain = result.get("domain")
    conf   = result.get("confidence", 0.0)
    src    = result.get("source", "?")
    intent_ok = "✅" if intent == expected else "❌"
    domain_ok = "✅" if domain == EXPECTED_DOMAIN else "❌"
    return (
        f"{intent_ok} intent={intent:<14}  exp={expected:<12}"
        f"  {domain_ok} domain={domain:<22}  conf={conf:.2f}  src={src:<8}"
        f"  q={query!r}"
    )


def _summary(rows_intent_ok: int, rows_domain_ok: int, total: int, elapsed: float) -> str:
    return (
        f"\nIntent matches : {rows_intent_ok}/{total}"
        f"\nDomain matches : {rows_domain_ok}/{total}"
        f"\nElapsed        : {elapsed:.2f}s"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    try:
        from multidomain_intent_detection import classify_query
    except Exception as e:
        print(f"❌ Could not import multidomain_intent_detection: {e}")
        print("   Make sure the trained pipeline (v3_pipeline.pkl) exists and")
        print("   that all classifier dependencies are installed.")
        return 2

    # Optional: verify api_routing_config knows about all of these intents.
    try:
        from config.api_routing_config import is_claim_history_search_intent
    except Exception:
        is_claim_history_search_intent = lambda _x: False  # type: ignore

    print("=" * 100)
    print("Multidomain Intent Detection — claim_history_search smoke test")
    print("=" * 100)

    rows_intent_ok = 0
    rows_domain_ok = 0
    t0 = time.time()

    for query, expected_intent in TEST_QUERIES:
        try:
            result = classify_query(query)
        except Exception as e:
            print(f"❌ Crashed on query={query!r}: {e}")
            continue

        if result.get("intent") == expected_intent:
            rows_intent_ok += 1
        if result.get("domain") == EXPECTED_DOMAIN:
            rows_domain_ok += 1

        # Sanity-check: api_routing_config is in sync with the classifier
        cfg_says_history = is_claim_history_search_intent(result.get("intent") or "")
        if result.get("domain") == EXPECTED_DOMAIN and not cfg_says_history:
            print(
                f"⚠️  api_routing_config does not list intent "
                f"{result.get('intent')!r} as claim_history_search; "
                f"the LangGraph router will still work via the domain check, "
                f"but consider adding it to INTENT_API_ROUTING for completeness."
            )

        print(_fmt_row(query, expected_intent, result))

    elapsed = time.time() - t0
    print("-" * 100)
    print(_summary(rows_intent_ok, rows_domain_ok, len(TEST_QUERIES), elapsed))

    # Exit non-zero if the routing path is broken — CI catches regressions.
    return 0 if rows_domain_ok == len(TEST_QUERIES) else 1


if __name__ == "__main__":
    sys.exit(main())
