"""
Render Scenario Runner
======================
Sends real prompts to the live API and saves full responses to disk.

Usage:
  python scripts/run_render_scenarios.py                    # all scenarios
  python scripts/run_render_scenarios.py --filter text      # text-only scenarios
  python scripts/run_render_scenarios.py --filter table     # table scenarios
  python scripts/run_render_scenarios.py --url http://localhost:8001
  python scripts/run_render_scenarios.py --out results/

The server must be running before you start this script.
Output goes to  scripts/render_scenario_results/<timestamp>/
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Prompt catalogue
# ---------------------------------------------------------------------------
# Each scenario has:
#   prompt        - natural-language text to send to /chat
#   expected      - "text" | "table"  (the render_format we expect)
#   tier          - "no_render" | "tier2_text" | "tier1_table" | "llm_decides"
#   intent_hint   - what intent we expect the LLM to classify this as
#   note          - why this scenario belongs to that expected output
#
# Scenarios are ordered: text → table, matching the two-tier system.

SCENARIOS = [
    # -------------------------------------------------------------------------
    # TEXT — NO_RENDER intents (Gate 1, never rendered)
    # -------------------------------------------------------------------------
    {
        "id": "T01",
        "prompt": "Hello",
        "expected": "text",
        "tier": "no_render",
        "intent_hint": "greeting",
        "note": "Greeting — Gate 1 fires, no tool call at all",
    },
    {
        "id": "T02",
        "prompt": "What can you help me with?",
        "expected": "text",
        "tier": "no_render",
        "intent_hint": "help",
        "note": "Help request — always plain text",
    },
    {
        "id": "T03",
        "prompt": "What is the weather today?",
        "expected": "text",
        "tier": "no_render",
        "intent_hint": "out_of_scope",
        "note": "Out-of-scope question — no rendering",
    },

    # -------------------------------------------------------------------------
    # TEXT — Tier 3 (LLM decides text_only for single-value answers)
    # -------------------------------------------------------------------------
    {
        "id": "T04",
        "prompt": "What is the status of my claim?",
        "expected": "text",
        "tier": "tier3_text",
        "intent_hint": "claim_status",
        "note": "Single-value answer — LLM should output render_mode=text_only",
    },
    {
        "id": "T05",
        "prompt": "How much did I pay for my last prescription?",
        "expected": "text",
        "tier": "tier3_text",
        "intent_hint": "claim_details",
        "note": "Single dollar amount — text_only",
    },
    {
        "id": "T06",
        "prompt": "When was my prescription filled?",
        "expected": "text",
        "tier": "tier3_text",
        "intent_hint": "rx_details",
        "note": "Single date — text_only",
    },
    {
        "id": "T07",
        "prompt": "What is the DAW code for this claim?",
        "expected": "text",
        "tier": "tier3_text",
        "intent_hint": "claim_details",
        "note": "Single code — text_only",
    },
    {
        "id": "T08",
        "prompt": "Was prior authorization required for this claim?",
        "expected": "text",
        "tier": "tier3_text",
        "intent_hint": "claim_status",
        "note": "Yes/No question — text_only",
    },
    {
        "id": "T09",
        "prompt": "Is this a compound drug claim?",
        "expected": "text",
        "tier": "tier3_text",
        "intent_hint": "claim_status",
        "note": "Yes/No question — text_only",
    },
    {
        "id": "T10",
        "prompt": "What is my member ID?",
        "expected": "text",
        "tier": "tier3_text",
        "intent_hint": "claim_status",
        "note": "Single identifier — text_only",
    },
    {
        "id": "T11",
        "prompt": "What pharmacy dispensed my prescription?",
        "expected": "text",
        "tier": "tier3_text",
        "intent_hint": "pharmacy_info",
        "note": "If classified as pharmacy_info → LLM decides (text for name, card for details); "
                "if classified as claim_status → text_only",
    },

    # -------------------------------------------------------------------------
    # LLM-decides intents (text or table based on question and data shape)
    # -------------------------------------------------------------------------
    {
        "id": "C06",
        "prompt": "What is the approval status for this claim?",
        "expected": "text",
        "tier": "llm_decides",
        "intent_hint": "approval_info",
        "note": "approval_info → LLM decides → text_only for single status value",
    },

    # -------------------------------------------------------------------------
    # TABLE — Tier 1 MUST_RENDER_INTENTS
    # -------------------------------------------------------------------------
    {
        "id": "TAB01",
        "prompt": "Show me the pricing breakdown for this claim",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "pricing_info",
        "note": "pricing_info → MUST_RENDER → html_table (multiple cost components)",
    },
    {
        "id": "TAB02",
        "prompt": "What are my copay amounts by drug tier?",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "copay_info",
        "note": "copay_info → MUST_RENDER → html_table (tier table)",
    },
    {
        "id": "TAB03",
        "prompt": "Show all my recent claims",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "claim_list",
        "note": "claim_list → MUST_RENDER → html_table (multiple rows)",
    },
    {
        "id": "TAB04",
        "prompt": "Why was my claim rejected? Show me the rejection codes.",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "rejection_reasons",
        "note": "rejection_reasons → MUST_RENDER → html_table (code + reason pairs)",
    },
    {
        "id": "TAB05",
        "prompt": "Show me the coordination of benefits details",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "cob_info",
        "note": "cob_info → MUST_RENDER → html_table (primary + secondary breakdown)",
    },
    {
        "id": "TAB06",
        "prompt": "What is my deductible status and accumulator amounts?",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "deductible_info",
        "note": "deductible_info → MUST_RENDER → html_table (accumulator breakdown)",
    },
    {
        "id": "TAB07",
        "prompt": "Give me a summary of all my claims this year",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "claim_summary",
        "note": "claim_summary → MUST_RENDER → html_table",
    },
    {
        "id": "TAB08",
        "prompt": "Show me claims from January to March",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "date_range_search",
        "note": "date_range_search → MUST_RENDER → html_table",
    },
    {
        "id": "TAB09",
        "prompt": "Which of my claims had the highest cost?",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "expensive_claims",
        "note": "expensive_claims → MUST_RENDER → html_table",
    },
    {
        "id": "TAB10",
        "prompt": "Show me the reversal details for this claim",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "reversal_info",
        "note": "reversal_info → MUST_RENDER → html_table",
    },
    {
        "id": "TAB11",
        "prompt": "What are the compound drug ingredients and costs?",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "compound_info",
        "note": "compound_info → MUST_RENDER → html_table (ingredient list)",
    },
    {
        "id": "TAB12",
        "prompt": "Show my Medicare Part D coverage stage and TrOOP amounts",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "medicare_part_d",
        "note": "medicare_part_d → MUST_RENDER → html_table (TrOOP stages)",
    },

    # -------------------------------------------------------------------------
    # SAFETY NET — LLM picks wrong render_mode, Python overrides
    # -------------------------------------------------------------------------
    {
        "id": "SN01",
        "prompt": "What is the copay for tier 1 drugs?",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "copay_info",
        "note": "SAFETY NET: LLM may output text_only (single copay value), "
                "but copay_info is MUST_RENDER → still html_table",
    },
    {
        "id": "SN03",
        "prompt": "How much was the ingredient cost for this claim?",
        "expected": "table",
        "tier": "tier1_table",
        "intent_hint": "pricing_info",
        "note": "SAFETY NET: LLM may output text_only (single dollar), "
                "but pricing_info is MUST_RENDER → still html_table",
    },
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def chat(base_url: str, prompt: str, session_id: str, timeout: int = 60) -> dict:
    """Send a single chat request; return the parsed JSON response."""
    payload = {
        "text": prompt,
        "session_id": session_id,
    }
    resp = requests.post(f"{base_url}/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def check_server(base_url: str) -> bool:
    """Return True if the server responds to a health check."""
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        return r.status_code < 500
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Result evaluation
# ---------------------------------------------------------------------------

PASS  = "PASS"
FAIL  = "FAIL"
WARN  = "WARN"    # correct category but mismatched render_mode value
SKIP  = "SKIP"


def evaluate(scenario: dict, response: dict) -> dict:
    """Compare expected vs actual render_format and render_mode."""
    actual_format = response.get("render_format", "text")
    actual_mode   = response.get("render_mode")
    actual_intent = response.get("intent", "")
    expected      = scenario["expected"]

    # Map expected → expected render_format string
    format_map = {
        "text":  "text",
        "table": "html_table",
    }
    expected_format = format_map[expected]

    status  = PASS
    reasons = []

    if actual_format != expected_format:
        status = FAIL
        reasons.append(
            f"render_format: expected={expected_format!r}  actual={actual_format!r}"
        )
    else:
        # Correct format — check render_mode consistency
        if expected == "text" and actual_mode not in (None, "text_only"):
            status = WARN
            reasons.append(f"render_mode expected text_only or None, got {actual_mode!r}")
        elif expected == "table" and actual_mode not in ("table", None):
            status = WARN
            reasons.append(f"render_mode expected 'table' or None, got {actual_mode!r}")

    if actual_intent and actual_intent != scenario["intent_hint"]:
        reasons.append(f"intent classified as {actual_intent!r} (expected {scenario['intent_hint']!r})")

    return {
        "status":        status,
        "expected":      expected_format,
        "actual_format": actual_format,
        "actual_mode":   actual_mode,
        "actual_intent": actual_intent,
        "reasons":       reasons,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

STATUS_ICON = {PASS: "✓", FAIL: "✗", WARN: "⚠", SKIP: "–"}
STATUS_COL  = {PASS: "\033[32m", FAIL: "\033[31m", WARN: "\033[33m", SKIP: "\033[90m"}
RESET       = "\033[0m"


def _colorize(status: str, text: str) -> bool:
    """Return colored text if stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{STATUS_COL[status]}{text}{RESET}"


def print_row(scenario: dict, eval_result: dict, elapsed_ms: float):
    icon = STATUS_ICON[eval_result["status"]]
    fmt  = eval_result["actual_format"][:10].ljust(10)
    mode = (eval_result["actual_mode"] or "—")[:12].ljust(12)
    sid  = scenario["id"].ljust(7)
    prompt_short = scenario["prompt"][:52].ljust(52)
    ms   = f"{elapsed_ms:6.0f}ms"
    line = f"  {icon} {sid} {prompt_short} {fmt} {mode} {ms}"
    print(_colorize(eval_result["status"], line))
    for r in eval_result["reasons"]:
        print(f"        → {r}")


def print_summary(results: list):
    counts = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
    for r in results:
        counts[r["eval"]["status"]] += 1
    total = len(results)
    print()
    print("─" * 80)
    print(f"  Total: {total}  "
          f"{_colorize(PASS, str(counts[PASS]) + ' pass')}  "
          f"{_colorize(FAIL, str(counts[FAIL]) + ' fail')}  "
          f"{_colorize(WARN, str(counts[WARN]) + ' warn')}  "
          f"{counts[SKIP]} skip")
    print("─" * 80)


# ---------------------------------------------------------------------------
# File saving
# ---------------------------------------------------------------------------

def save_results(out_dir: Path, results: list):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full JSON dump (all scenarios + raw API responses)
    full_path = out_dir / "full_results.json"
    full_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Summary CSV — easy to open in Excel
    csv_path = out_dir / "summary.csv"
    lines = ["id,prompt,expected,actual_format,render_mode,intent,status,elapsed_ms,reasons"]
    for r in results:
        s  = r["scenario"]
        ev = r["eval"]
        reasons_str = "; ".join(ev["reasons"]).replace('"', "'")
        lines.append(
            f'{s["id"]},'
            f'"{s["prompt"]}",'
            f'{s["expected"]},'
            f'{ev["actual_format"]},'
            f'{ev["actual_mode"] or ""},'
            f'{ev["actual_intent"]},'
            f'{ev["status"]},'
            f'{r["elapsed_ms"]:.0f},'
            f'"{reasons_str}"'
        )
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    # Per-scenario HTML files (for visual inspection in a browser)
    html_dir = out_dir / "html"
    html_dir.mkdir(exist_ok=True)
    for r in results:
        s    = r["scenario"]
        resp = r.get("response", {})
        html = resp.get("html_content") or ""
        css  = resp.get("css_content") or ""
        if not html:
            continue
        page = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>{css}</style></head><body>"
            f"<h3 style='font-family:sans-serif'>{s['id']} — {s['prompt']}</h3>"
            f"{html}"
            "</body></html>"
        )
        (html_dir / f"{s['id']}.html").write_text(page, encoding="utf-8")

    return full_path, csv_path, html_dir


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(base_url: str, filter_expected: Optional[str], out_dir: Path, delay_s: float):
    scenarios = SCENARIOS
    if filter_expected:
        scenarios = [s for s in scenarios if s["expected"] == filter_expected]

    print()
    print(f"  Render Scenario Runner  —  {base_url}")
    print(f"  Scenarios: {len(scenarios)}   Filter: {filter_expected or 'all'}")
    print(f"  Output:    {out_dir}")
    print()
    header = f"  {'St':2} {'ID':7} {'Prompt':52} {'Format':10} {'Mode':12} {'Time':8}"
    print(header)
    print("  " + "─" * 94)

    session_id = str(uuid.uuid4())
    all_results = []

    for scenario in scenarios:
        t0 = time.perf_counter()
        try:
            response = chat(base_url, scenario["prompt"], session_id)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            eval_result = evaluate(scenario, response)
        except requests.exceptions.ConnectionError:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            eval_result = {
                "status": SKIP,
                "expected": scenario["expected"],
                "actual_format": "—",
                "actual_mode": None,
                "actual_intent": "—",
                "reasons": ["ConnectionError — is the server running?"],
            }
            response = {}
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            eval_result = {
                "status": FAIL,
                "expected": scenario["expected"],
                "actual_format": "—",
                "actual_mode": None,
                "actual_intent": "—",
                "reasons": [f"{type(exc).__name__}: {exc}"],
            }
            response = {}

        print_row(scenario, eval_result, elapsed_ms)

        all_results.append({
            "scenario":   scenario,
            "eval":       eval_result,
            "elapsed_ms": elapsed_ms,
            "response":   {k: v for k, v in response.items() if k != "html_content"},
            "html_saved": bool(response.get("html_content")),
        })

        if delay_s > 0:
            time.sleep(delay_s)

    print_summary(all_results)

    full_path, csv_path, html_dir = save_results(out_dir, all_results)
    html_count = len(list(html_dir.glob("*.html")))
    print(f"\n  Saved:")
    print(f"    {full_path}")
    print(f"    {csv_path}")
    print(f"    {html_dir}  ({html_count} HTML files)")
    print()

    fail_count = sum(1 for r in all_results if r["eval"]["status"] == FAIL)
    return 0 if fail_count == 0 else 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run render scenarios against the live API.")
    p.add_argument("--url",    default="http://localhost:8000", help="Base URL of the running server")
    p.add_argument("--filter", choices=["text", "table"], default=None,
                   help="Only run scenarios with this expected output")
    p.add_argument("--out",    default=None,
                   help="Output directory (default: scripts/render_scenario_results/<timestamp>)")
    p.add_argument("--delay",  type=float, default=1.0,
                   help="Seconds to wait between requests (default: 1.0)")
    p.add_argument("--no-check", action="store_true",
                   help="Skip the server health-check before running")
    return p


def main():
    args = _build_parser().parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / "render_scenario_results" / timestamp
    )

    if not args.no_check:
        print(f"\n  Checking server at {args.url} …", end=" ", flush=True)
        if not check_server(args.url):
            print("UNREACHABLE")
            print("  Start the server first:  uvicorn main:app --reload")
            print("  Or skip the check:       --no-check")
            sys.exit(2)
        print("OK")

    sys.exit(run(args.url, args.filter, out_dir, args.delay))


if __name__ == "__main__":
    main()
