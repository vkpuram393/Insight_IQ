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

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_IDS_DIR = os.path.join(os.path.dirname(BASE_DIR), "Intent_detection_system")
ARTIFACTS = os.path.join(_IDS_DIR, "artifacts") if os.path.isdir(os.path.join(_IDS_DIR, "artifacts")) else os.path.join(BASE_DIR, "artifacts")
OUTPUTS = os.path.join(_IDS_DIR, "outputs") if os.path.isdir(_IDS_DIR) else os.path.join(BASE_DIR, "outputs")
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
    from VamsiSir import embeddingVars
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


# ─────────────────────────────────────────────────────────────────────────────
# Augmented training examples  (real-world phrasing patterns)
# ─────────────────────────────────────────────────────────────────────────────

AUGMENTED_EXAMPLES: Dict[str, List[str]] = {
    # ── cap_api anchors ──────────────────────────────────────────────────
    "prescriber_info": [
        "Prescriber details for claim 132435151040074 sequence 001.",
        "Physician information for claim 150692388845000 sequence 001.",
        "Doctor's name for claim 160060096136030 sequence 001.",
        "Prescriber NPI for claim 191406379285000 sequence 001.",
        "Who prescribed the medication on claim 130041467416065 sequence 001?",
        "Which physician wrote the prescription for claim 180571470939000?",
        "Prescribing physician information for claim 221663541811000 sequence 001.",
        "Prescriber NPI and name for claim 221775171449000 sequence 003.",
        "Who ordered the medication on claim 221172865083001 sequence 001?",
        "Ordering provider information for claim 230624075311000 sequence 002.",
    ],
    "settlement_info": [
        "Settlement details for claim 220133725669000 sequence 001.",
        "Settlement information for claim 222492018072002 sequence 001.",
        "Settlement report for claim 222492117457002 sequence 001.",
        "Settlement status for claim 241774768475148 sequence 003.",
        "Settlement summary for claim 242831720377166 sequence 002.",
        "Settlement feedback for claim 243122443413000 sequence 001.",
        "Response information for claim 250023213779000 sequence 001.",
    ],
    "audit_info": [
        "When was claim 132435151040074 sequence 001 first created?",
        "Claim 201503823714118 sequence 001 add date and change date.",
        "Who last modified claim 201752592251000 sequence 001 and when?",
        "What is the creation timestamp of claim 211263773300000 sequence 004?",
        "When was claim 221172865083001 sequence 001 added to the system?",
    ],
    "reversal_info": [
        "R&R information for claim 242905816136000 sequence 001.",
        "R&R status for claim 242905816136000 sequence 001.",
        "R&R report for claim 253603736282009 sequence 001.",
        "Was claim 231181462825000 sequence 001 reversed?",
        "Claim modifications for claim 260021649904000 sequence 005.",
    ],
    "claim_status": [
        "What is the current status of claim 130041467416065 sequence 001?",
        "Is claim 220133725669000 sequence 001 paid, rejected, or pending?",
        "Quick status check on claim 230381673488000 sequence 001.",
        "What was the result of processing claim 230624075311000 sequence 002?",
        "Adjudication outcome for claim 191406379285000 sequence 001.",
    ],
    "pricing_info": [
        "What is the copay on claim 132435151040074 sequence 001?",
        "Show the pricing breakdown for claim 220133725669000 sequence 001.",
        "What did the patient pay on this specific claim?",
        "Ingredient cost and fees for claim 191406379285000 sequence 001.",
        "Copay calculation steps for this claim.",
    ],
    "pharmacy_info": [
        "Which pharmacy dispensed claim 132435151040074 sequence 001?",
        "Pharmacy details for claim 220133725669000 sequence 001.",
        "Where was this specific claim filled?",
        "Dispensing pharmacy name for this claim.",
        "Store location for claim 191406379285000 sequence 001.",
    ],
    # ── claim_history_search anchors ─────────────────────────────────────
    "Pricing": [
        "How much did the member pay for METFORMIN across all fills?",
        "Show me the total spent on ATORVASTATIN prescriptions.",
        "What was the copay trend for LISINOPRIL fills over time?",
        "Compare costs across multiple SERTRALINE claims.",
        "List the pricing for all GABAPENTIN claims this year.",
    ],
    "Settlement": [
        "Show claims with settlement code 358.",
        "Filter by settlement code 001 across all claims.",
        "Which claims returned settlement code 425?",
        "List fills that received settlement code 310.",
        "Retrieve claims with pharmacy settlement response 200.",
    ],
    "Pharmacy": [
        "Show claims filled at CVS PHARMACY 00610.",
        "List fills dispensed by WALGREENS 04528.",
        "Retrieve claims from RITE AID 11237.",
        "Which fills were filled at TARGET PHARMACY 01893?",
        "Give me claims processed at WALMART PHARMACY 10340.",
    ],
    "Prescriber": [
        "Show claims by prescriber NOEUV.",
        "List claims written by Dr. PATEL.",
        "Retrieve fills prescribed by NPI 1234567890.",
        "Which claims were ordered by prescriber SMITH?",
        "Display claims from prescriber Dr. JOHNSON.",
    ],
    "Status": [
        "Show all rejected claims for this member.",
        "List claims in paid status this year.",
        "Which claims are currently pending?",
        "Display all denied claims across all drugs.",
        "Give me claims in reversed status.",
    ],
    "RejectCode": [
        "Show claims with reject code 79.",
        "Filter claims by rejection code 75.",
        "Which claims have NCPDP reject code MR?",
        "List claims rejected under code 76.",
        "Retrieve claims with reject code 70.",
    ],
    "PriorAuth": [
        "Which claims required prior authorization?",
        "Show fills that went through a PA process.",
        "List claims where PA was approved.",
        "Retrieve prescriptions with an active prior auth on file.",
        "Display PA-approved claims for specialty drugs.",
    ],
    # ── general ──────────────────────────────────────────────────────────
    "greeting": [
        "Hello", "Hi there", "Welcome", "Hiya",
        "Hello, how are you?", "Hi, good to see you",
    ],
    "out_of_scope": [
        "What is the weather today?", "Tell me a joke.",
        "Who won the Super Bowl?", "What's up", "How do I cook pasta?",
    ],
    # ── member_domain anchors ────────────────────────────────────────────
    "member_coverage": [
        "Does this member have active coverage as of today?",
        "Show me the coverage eligibility windows for this member.",
        "What are the eligibility dates for member John Doe?",
        "When does this member's coverage begin and end?",
        "Is member 555123456 eligible right now?",
    ],
    "member_hierarchy": [
        "Which client does this member belong to?",
        "Show me the CAG hierarchy for this member.",
        "What client account group is this member under?",
        "Display the hierarchy information for this member.",
        "Give me the client and group assignment for this member.",
    ],
    "benefit_reset_date": [
        "What is the benefit reset date for this member?",
        "When does the benefit year reset for this member?",
        "Tell me when the accumulators reset for this member.",
        "Show me the plan year reset date for this member.",
        "When do the deductible and OOP accumulators reset?",
    ],
    "medicare_coverage": [
        "Does this member have Part D coverage?",
        "Is this member enrolled in Medicare?",
        "Show the Medicare Part D status for this member.",
        "Is this member a Medicare beneficiary?",
        "Tell me the Medicare coverage status for this member.",
    ],
    "lics_status": [
        "Is this member LICS?",
        "Does this member qualify for low income subsidy?",
        "Show the LICS status for this member.",
        "What LICS level is assigned to this member?",
        "Tell me if this member is receiving low income cost sharing.",
    ],
    "cvs_id_lookup": [
        "What is the CVS ID for this member?",
        "Show the CVS ID associated with this member.",
        "Retrieve the CVS ID for member John Doe.",
        "Give me the CVS identifier for this member.",
        "Look up the CVS ID for this member.",
    ],
    "alternate_ids": [
        "List all alternate IDs for this member.",
        "Show the alternate identifiers on file for this member.",
        "What alternate IDs are assigned to this member?",
        "Retrieve all alternate member IDs for this member.",
        "Give me all alternate IDs associated with this member.",
    ],
    # ── override_domain anchors ──────────────────────────────────────────
    "pa_summary": [
        "Give me a summary of this prior authorization.",
        "Summarize the key details of this PA.",
        "Show the most important fields on this PA.",
        "Display a high-level overview of this prior authorization.",
        "Provide the PA summary including effective dates and drug coverage.",
    ],
    "pa_override_reject": [
        "Will this PA override a reject 75 PA required?",
        "Does this PA handle reject code 75?",
        "Will this prior authorization bypass a reject 70 non-formulary?",
        "Does this PA override reject 70 plan exclusion?",
        "Show me which reject codes this PA can override.",
    ],
    "pa_field_help": [
        "What does the PA type field do?",
        "Explain the purpose of the effective date field on a PA.",
        "What is the GPI override field used for?",
        "Describe what the PA status indicator means.",
        "What does the quantity limit override field do on this PA?",
    ],
    "pa_copay_pricing": [
        "Does this copay override influence the price?",
        "How does the copay on this PA affect pricing?",
        "Will the PA copay change the member's out-of-pocket cost?",
        "Show me how the copay override impacts the final price.",
        "Does the copay field on this PA modify the claim price?",
    ],
    "pa_drug_coverage": [
        "What drugs will this PA cover?",
        "Show me the drug list covered by this prior authorization.",
        "Which medications are included under this PA?",
        "List the drugs that this PA authorizes.",
        "Display the GPI range covered by this PA.",
    ],
    "pa_claim_usage": [
        "How many claims used this PA?",
        "Show the claim count for this prior authorization.",
        "How many times has this PA been applied to claims?",
        "Display the number of claims processed under this PA.",
        "Retrieve the claim usage count for this PA.",
    ],
    # ── benefits_api anchors ─────────────────────────────────────────────
    "plan_summary": [
        "Show the current benefit plan overview for this member.",
        "Give me a snapshot of the member's active plan.",
        "What does this member's benefit plan cover?",
        "Display the current plan summary.",
        "Summarize the active benefit plan for this member.",
    ],
    "plan_history": [
        "Show the change log of this member's benefit plan.",
        "What modifications have been made to the plan over time?",
        "List past revisions of the benefits plan.",
        "Display the audit trail of plan changes.",
        "Give me the timeline of updates to the plan.",
    ],
    "plan_finder": [
        "Help me locate an available benefit plan.",
        "Search for plans that match this client.",
        "Which plans are offered to this member's group?",
        "Find a matching benefits plan.",
        "Look up what plans exist for this client.",
    ],
}


def augment_embeddings(embeddings: Dict) -> Dict:
    """Add augmented training examples to cached embeddings.

    Normalizes examples (strips claim numbers) before embedding so they
    match the normalized test query space.
    """
    aug_cache_path = os.path.join(ARTIFACTS, "augmented_embeddings_v3.json")

    if os.path.exists(aug_cache_path):
        with open(aug_cache_path) as f:
            aug_cached = json.load(f)
    else:
        aug_cached = {}

    needs_generation = False
    for intent, examples in AUGMENTED_EXAMPLES.items():
        if intent not in aug_cached or len(aug_cached[intent]) != len(examples):
            needs_generation = True
            break

    if needs_generation:
        logger.info("Generating augmented training embeddings (normalized)...")
        emb = get_embedder()
        for intent, examples in AUGMENTED_EXAMPLES.items():
            if intent not in aug_cached or len(aug_cached[intent]) != len(examples):
                normalized = [normalize_query(ex) for ex in examples]
                logger.info(f"  Augmenting '{intent}' with {len(examples)} normalized examples")
                aug_cached[intent] = [list(v) for v in emb.embed_batch(normalized)]
        with open(aug_cache_path, "w") as f:
            json.dump(aug_cached, f)
        logger.info(f"Augmented embeddings saved → {aug_cache_path}")

    # Merge
    merged = {k: list(v) for k, v in embeddings.items()}
    for intent, vecs in aug_cached.items():
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
    """Evaluate pipeline on test data.  Returns {intent_accuracy, domain_accuracy}."""
    results, llm_n = [], 0
    for idx, rec in enumerate(test_data):
        normalized_text = normalize_query(rec["text"])
        vec = np.array(embedder.embed(normalized_text))
        pred = pipeline.predict_single(vec)
        confident = pred["confidence"] >= conf_t and pred["margin"] >= margin_t

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
            "margin": pred["margin"],
            "source": src,
            "agreement": pred["agreement"],
        })
        if (idx + 1) % 50 == 0:
            logger.info(f"  {idx + 1}/{len(test_data)}")

    df = pd.DataFrame(results)
    mode = "hybrid" if use_llm else "ensemble_only"
    df.to_csv(os.path.join(OUTPUTS, f"results_{mode}.csv"), index=False)

    ia = df["intent_match"].mean() * 100
    da = df["domain_match"].mean() * 100
    print(f"\n{'=' * 60}")
    print(f"  {'ENSEMBLE + LLM' if use_llm else 'ENSEMBLE ONLY'}")
    print(f"  Intent Accuracy : {ia:.2f}%")
    print(f"  Domain Accuracy : {da:.2f}%")
    if use_llm:
        ep = (len(test_data) - llm_n) / len(test_data) * 100
        print(f"  Ensemble resolved : {ep:.1f}% ({len(test_data) - llm_n}/{len(test_data)})")
        print(f"  LLM calls         : {llm_n} ({llm_n / len(test_data) * 100:.1f}%)")
    print(f"{'=' * 60}")
    print()
    return {"intent_accuracy": ia, "domain_accuracy": da}


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
