"""
Append synthetic test prompts for the 26 override_domain (pa_*) intents to
the training/eval CSV(s) used by `multidomain_intent_detection/training.py`.

Idempotent:
  - Loads existing CSV
  - Filters out any (Prompt, Intent, domain) triples we are about to insert
    that already exist
  - Appends only NEW rows
  - Writes back in-place (UTF-8, no index)

CSV schema: Prompt, Intent, domain
"""

from __future__ import annotations

import csv
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Both CSVs are kept in lock-step (verified identical via MD5).
TARGET_CSVS = [
    REPO_ROOT / "ImportantData" / "trainingcomplete_set.csv",
    REPO_ROOT / "multidomain_intent_detection" / "ImportantData" / "trainingcomplete_set.csv",
]

DOMAIN = "override_domain"

# 3 distinct, natural-phrasing eval prompts per intent (26 × 3 = 78 rows).
# These are intentionally short and conversational so they probe the
# ensemble's generalization, not memorization of the augmented templates.
OVERRIDE_TEST_PROMPTS: dict[str, list[str]] = {
    "pa_summary": [
        "Give me a quick summary of PA JW012726LC.",
        "What does this prior auth do, in a nutshell?",
        "Show me the high-level overview of this PA record.",
    ],
    "pa_override_reject": [
        "Will this PA take care of a reject 75?",
        "Can this prior auth bypass a non-formulary 70 rejection?",
        "Does this PA override plan-limits reject 76?",
    ],
    "pa_field_help": [
        "Explain what effectiveBeginDate means on a PA.",
        "What is the agentCode field used for?",
        "I don't understand this PA field — what does it control?",
    ],
    "pa_copay_pricing": [
        "Does the copay on this PA change what the member pays?",
        "How does this PA's copay override affect pricing?",
        "Will the PA copay reduce the patient's out-of-pocket cost?",
    ],
    "pa_drug_coverage": [
        "Which drugs does this PA cover?",
        "Does this PA include LISINOPRIL?",
        "Show the NDC list authorized by this prior auth.",
    ],
    "pa_claim_usage": [
        "How many claims have used this PA?",
        "Give me the utilization count for this prior auth.",
        "How frequently has this PA been applied to claims?",
    ],
    "pa_reason_code": [
        "What is the PA reason code on this override?",
        "Show the reason code assigned to this PA.",
        "Is the reason code on this PA set to U1 or LC?",
    ],
    "pa_effective_dates": [
        "When does this PA become effective and when does it expire?",
        "What are the begin and end dates for this PA?",
        "Is this prior authorization still active today?",
    ],
    "pa_agent_code": [
        "What agent code is set on this PA?",
        "Who created this PA — show the source/agent code.",
        "Is the agent code on this PA A, C, 3, or H?",
    ],
    "pa_ignore_status": [
        "What's the ignore status flag on this PA?",
        "Show the ignore status code (Y, P, or 3) for this PA.",
        "Is this PA set to ignore processing?",
    ],
    "pa_specialty_rx_override": [
        "Does this PA override the specialty Rx reject?",
        "Show the specialty Rx override indicator on this PA.",
        "Is specialty Rx rejection bypassed for this PA?",
    ],
    "pa_clinical_admin_code": [
        "What is the clinical administration code on this PA?",
        "Show the clinical admin code (A, C, or blank) for this PA.",
        "Retrieve the clinical administration setting on this PA.",
    ],
    "pa_transform_care": [
        "What transform care type is configured on this PA?",
        "Show the transform care setting for this prior auth.",
        "Does this PA participate in a care transformation program?",
    ],
    "pa_follow_me_logic": [
        "Is follow me logic enabled on this PA?",
        "Does this PA follow the member across plan changes?",
        "Show the follow-me indicator on this PA.",
    ],
    "pa_drug_type_indicator": [
        "Is the drug type indicator on this PA set to GPI or NDC?",
        "What authorized drug type does this PA use?",
        "Show the G/N drug-match method on this PA.",
    ],
    "pa_modification_history": [
        "When was this PA last modified?",
        "Show the last update timestamp on this prior auth.",
        "Retrieve the modifyDateTime on this PA record.",
    ],
    "pa_contingent_therapy_override": [
        "How do I bypass contingent therapy on this PA?",
        "Steps to override the contingent therapy flag on a PA.",
        "How do I flip the contingent therapy override on a PA?",
    ],
    "pa_smart_pa_override": [
        "How do I override Smart PA processing?",
        "Where do I enter the Smart PA criteria number?",
        "Steps to bypass Smart PA on a claim using a PA override.",
    ],
    "pa_part_b_override": [
        "How to make this claim pay under Medicare Part-B using a PA?",
        "Steps to set PA override reason MB for Part-B.",
        "How do I configure a PA so the claim adjudicates as Part-B?",
    ],
    "pa_esrd_override": [
        "How do I override the ESRD reject using a PA?",
        "Steps to set PA override reason ES for ESRD.",
        "How to bypass ESRD rejection on a claim with a PA override?",
    ],
    "pa_skip_deductible": [
        "How do I skip the deductible for this member using a PA?",
        "Steps to flip the skip DED flag on a PA override.",
        "How to configure a PA to bypass the deductible?",
    ],
    "pa_send_expiration": [
        "How do I send the PA expiration date on a claim?",
        "Steps to enable the send expiration date flag on a PA.",
        "How to configure the PA so the expiration date is transmitted?",
    ],
    "pa_tf_letter_setup": [
        "How do I set up the TF letter tag on a PA?",
        "Steps to configure transition fill letter type on a PA.",
        "How to attach a TF letter to a PA override?",
    ],
    "pa_copay_setup": [
        "How do I configure a different copay schedule on this PA?",
        "Steps to set up a custom copay on a PA override.",
        "How do I change the copay structure for this PA?",
    ],
    "pa_suggest_override": [
        "What PA should I enter to override this reject?",
        "Suggest the right PA override for this rejection.",
        "Which PA override clears this reject code?",
    ],
    "pa_reason_code_fields": [
        "What fields are required for PA override reason code U1?",
        "Which PA fields apply when the reason code is LC?",
        "List the applicable fields for reason code OA on a PA.",
    ],
}


def build_new_rows() -> list[dict]:
    rows: list[dict] = []
    for intent, prompts in OVERRIDE_TEST_PROMPTS.items():
        for p in prompts:
            rows.append({"Prompt": p, "Intent": intent, "domain": DOMAIN})
    return rows


def append_to(csv_path: Path, new_rows: list[dict]) -> tuple[int, int]:
    """Append rows not already present; return (added, skipped)."""
    df = pd.read_csv(csv_path)
    existing = set(zip(df["Prompt"].astype(str), df["Intent"].astype(str), df["domain"].astype(str)))
    to_add = [r for r in new_rows if (r["Prompt"], r["Intent"], r["domain"]) not in existing]
    if not to_add:
        return 0, len(new_rows)
    add_df = pd.DataFrame(to_add, columns=["Prompt", "Intent", "domain"])
    out = pd.concat([df, add_df], ignore_index=True)
    out.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    return len(to_add), len(new_rows) - len(to_add)


def main() -> int:
    new_rows = build_new_rows()
    print(f"Prepared {len(new_rows)} pa_* test rows across {len(OVERRIDE_TEST_PROMPTS)} intents.")

    for path in TARGET_CSVS:
        if not path.exists():
            print(f"  SKIP (missing): {path}")
            continue
        added, skipped = append_to(path, new_rows)
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        print(f"  {path}")
        print(f"     added={added}  already_present={skipped}  md5={md5}")

    # Quick verification
    print("\n=== Verification ===")
    for path in TARGET_CSVS:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        pa = df[df["Intent"].astype(str).str.startswith("pa_")]
        by_intent = pa["Intent"].value_counts().to_dict()
        print(f"  {path.name}: total={len(df)}, pa_*={len(pa)}, distinct pa intents={len(by_intent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
