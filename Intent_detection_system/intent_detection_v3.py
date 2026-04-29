"""
Intent Detection v3 — PCA + Ensemble Classifier + LLM Fallback

Architecture:
  ┌────────────────────────────────────────────────────────────────────┐
  │  STAGE 1: PCA Dimensionality Reduction (768 → optimal dims)      │
  │  Removes noise, concentrates signal. Fixes curse of              │
  │  dimensionality (20 samples in 768 dims → unlearnable).          │
  ├────────────────────────────────────────────────────────────────────┤
  │  STAGE 2: Calibrated Ensemble of 3 Classifiers                   │
  │    A) SVM-RBF — learns non-linear decision boundaries            │
  │    B) Logistic Regression — calibrated probabilities             │
  │    C) kNN (distance-weighted) — preserves decision boundaries    │
  │  Weighted soft voting with calibrated probabilities              │
  ├────────────────────────────────────────────────────────────────────┤
  │  STAGE 3: Confidence Gate → LLM Fallback                         │
  │    Confident → fast path (<1ms). Ambiguous → Gemini (~300ms).    │
  │    ~85-90% fast path, ~10-15% to LLM.                           │
  └────────────────────────────────────────────────────────────────────┘

No torch, no transformers, no GPU. Pure sklearn + numpy + Vertex AI.
"""

import os, sys, json, logging, time, pickle, re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from VamsiSir import embeddingVars

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")
OUTPUTS   = os.path.join(BASE_DIR, "outputs")
EMBEDDINGS_PATH = os.path.join(ARTIFACTS, "intent_embeddings.json")
MODEL_PKL = os.path.join(ARTIFACTS, "v3_pipeline.pkl")
os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# QUERY NORMALIZER — strips claim numbers so training and test align
# ═════════════════════════════════════════════════════════════════════════════
#
#  THE ROOT CAUSE OF 80% vs 95%:
#  Training: "Generate the audit log for this claim"
#  Test:     "Who last modified claim 201752592251000 sequence 001 and when?"
#
#  The 15-digit claim number DOMINATES the embedding vector, pushing the
#  test query into a completely different region than the training template.
#
#  FIX: Strip claim numbers, sequence numbers, and filler before embedding.
#  Both become: "who last modified claim and when" ≈ "generate audit log for claim"

_CLAIM_NUM_PATTERN = re.compile(r'\b\d{12,18}\b')                    # 12-18 digit claim numbers
_SEQ_PATTERN = re.compile(r'\bsequence\s+\d{1,3}\b', re.IGNORECASE)  # "sequence 001"
_SEQ_NUM = re.compile(r'\bseq\s+\d{1,3}\b', re.IGNORECASE)           # "seq 001"
_CLAIM_PREFIX = re.compile(r'\bclaim\s+\d{12,18}\b', re.IGNORECASE)   # "claim 123456..."
_PA_NUM_PATTERN = re.compile(r'\bPA\s+[A-Z0-9]{5,15}\b', re.IGNORECASE)  # "PA JW012726LC"
_WHITESPACE = re.compile(r'\s+')

def normalize_query(text: str) -> str:
    """Strip claim/sequence/PA numbers from query so embedding focuses on INTENT.
    
    Before: "Prescriber details for claim 132435151040074 sequence 001."
    After:  "prescriber details for claim"
    
    Also strips PA identifiers: "PA JW012726LC" → "pa"
    """
    t = text.lower().strip()
    t = _SEQ_PATTERN.sub('', t)     # remove "sequence 001"
    t = _SEQ_NUM.sub('', t)         # remove "seq 001"
    t = _CLAIM_NUM_PATTERN.sub('', t)  # remove bare claim numbers
    t = _PA_NUM_PATTERN.sub('pa', t)   # "PA JW012726LC" → "pa"
    t = t.replace('.', ' ').replace('?', ' ').replace('!', ' ')
    t = _WHITESPACE.sub(' ', t).strip()
    return t

# ── Embedding client ─────────────────────────────────────────────────────────
class VertexEmbeddings:
    def __init__(self):
        self.project = os.getenv("PROJECT_ID", "pbm-poc-coderev-genai-poc")
        self.location = os.getenv("LOCATION", "us-central1")
        from google import genai
        self.client = genai.Client(vertexai=True, project=self.project, location=self.location)
    def embed(self, text):
        from google.genai import types
        single = isinstance(text, str)
        texts = [text] if single else text
        out = []
        for i, t in enumerate(texts):
            if i > 0: time.sleep(0.3)
            if i > 0 and i % 20 == 0: time.sleep(5)
            bk = 2.0
            for att in range(5):
                try:
                    r = self.client.models.embed_content(model="text-embedding-005",
                        contents=[types.Part.from_text(text=t)])
                    out.append(r.embeddings[0].values); break
                except Exception as e:
                    if any(k in str(e).lower() for k in ("429","exhausted","quota")) and att < 4:
                        time.sleep(bk); bk *= 2
                    else: raise
        return out[0] if single else out

_emb = None
def get_embedder():
    global _emb
    if _emb is None: _emb = VertexEmbeddings()
    return _emb

# ── Domain mapping ───────────────────────────────────────────────────────────
INTENT_TO_DOMAIN = {
    # ── cap_api (single-claim operations) ─────────────────────────────────────
    "claim_status":"cap_api","multi_claim_summary":"cap_api","pharmacy_info":"cap_api",
    "prescriber_info":"cap_api","pricing_info":"cap_api","reimbursement_info":"cap_api",
    "rejection_reasons":"cap_api","settlement_info":"cap_api","rx_details":"cap_api",
    "reversal_info":"cap_api","cob_info":"cap_api","generic_availability":"cap_api",
    "daw_info":"cap_api","government_claim_type":"cap_api","mail_order_info":"cap_api",
    "medicare_part_d":"cap_api","network_info":"cap_api","prior_auth_info":"cap_api",
    # ── benefits_api ─────────────────────────────────────────────────────────
    "approval_info":"benefits_api","audit_info":"benefits_api","beneficiary_info":"benefits_api",
    "plan_summary":"benefits_api","plan_history":"benefits_api","plan_finder":"benefits_api",
    # ── claim_history_search ─────────────────────────────────────────────────
    "compound_info":"claim_history_search","date_range_claims":"claim_history_search",
    "drug_info":"claim_history_search","drug_interaction_info":"claim_history_search",
    "fill_date_info":"claim_history_search",
    "Refills":"claim_history_search","DaysSupply":"claim_history_search",
    "PriorAuth":"claim_history_search","Diagnosis":"claim_history_search",
    "Settlement":"claim_history_search","PharmType":"claim_history_search",
    "Plan":"claim_history_search","Pharmacy":"claim_history_search",
    "Prescriber":"claim_history_search","Pricing":"claim_history_search",
    "Status":"claim_history_search","RejectCode":"claim_history_search",
    "DrugLast":"claim_history_search","Month":"claim_history_search",
    "ClaimNum":"claim_history_search","NDC":"claim_history_search",
    "Manufacturer":"claim_history_search","Generic":"claim_history_search",
    "Brand":"claim_history_search",
    # ── general ──────────────────────────────────────────────────────────────
    "greeting":"general","help":"general","out_of_scope":"general",
    # ── member_domain ────────────────────────────────────────────────────────
    "member_coverage":"member_domain","member_hierarchy":"member_domain",
    "benefit_reset_date":"member_domain","family_type":"member_domain",
    "family_members":"member_domain","alternate_insurance":"member_domain",
    "medicare_coverage":"member_domain","lics_status":"member_domain",
    "stcob_linkage":"member_domain","cvs_id_lookup":"member_domain",
    "related_cagm":"member_domain","alternate_ids":"member_domain",
    "member_demographics":"member_domain","member_contact_info":"member_domain",
    "member_eligibility_copay":"member_domain","member_transition_status":"member_domain",
    "member_dur_config":"member_domain","member_mbi_number":"member_domain",
    "member_caretaker_info":"member_domain","member_language_pref":"member_domain",
    "member_discount_program":"member_domain","member_override_plan":"member_domain",
    # ── override_domain ──────────────────────────────────────────────────────
    "pa_summary":"override_domain","pa_override_reject":"override_domain",
    "pa_field_help":"override_domain","pa_copay_pricing":"override_domain",
    "pa_drug_coverage":"override_domain","pa_claim_usage":"override_domain",
    "pa_reason_code":"override_domain","pa_effective_dates":"override_domain",
    "pa_agent_code":"override_domain","pa_ignore_status":"override_domain",
    "pa_specialty_rx_override":"override_domain","pa_clinical_admin_code":"override_domain",
    "pa_transform_care":"override_domain","pa_follow_me_logic":"override_domain",
    "pa_drug_type_indicator":"override_domain","pa_modification_history":"override_domain",
}

INTENT_DESC = {
    # ── cap_api (single-claim) ────────────────────────────────────────────────
    "claim_status":"General claim status, adjudication outcome, paid/rejected/pending",
    "multi_claim_summary":"Summary of ALL/MULTIPLE claims for a member",
    "pharmacy_info":"Dispensing pharmacy name, location, address, NCPDP, store for ONE claim",
    "prescriber_info":"Prescribing physician/doctor name, NPI, credentials for ONE claim",
    "pricing_info":"Copay, ingredient cost, dispensing fee, patient pay breakdown for ONE claim",
    "reimbursement_info":"Amount paid TO pharmacy, reimbursement rationale, payment",
    "rejection_reasons":"Rejection codes, failed edits, denial reasons, how to resolve for ONE claim",
    "settlement_info":"Settlement codes, pharmacy response/feedback codes for ONE claim",
    "rx_details":"RX number, fill number, quantity, days supply, strength",
    "reversal_info":"Claim reversal, R&R, manual adjustments, resubmission",
    "cob_info":"Coordination of Benefits, other insurance, secondary payer, dual coverage",
    "generic_availability":"Generic alternatives, therapeutic equivalents, formulary substitutes",
    "daw_info":"DAW status, brand vs generic requirement, substitution",
    "government_claim_type":"Medicare/Medicaid claim type, government program",
    "mail_order_info":"Mail order/home delivery prescription, shipping",
    "medicare_part_d":"Medicare Part D summary, PDE, MEDD pricing, LICS for a claim",
    "network_info":"Pharmacy network details, which network processed claim",
    "prior_auth_info":"Prior authorization status, Smart PA, authorization requirements for ONE claim",
    # ── benefits_api ─────────────────────────────────────────────────────────
    "approval_info":"Claim approval, plan overrides, transition fill (TF), BPG, Smart PA",
    "audit_info":"Audit trail, change history, modification records, timestamps",
    "beneficiary_info":"Member benefit phase, coverage tier, eligibility, accumulations",
    "plan_summary":"Benefit plan overview, active plan snapshot, current coverage summary",
    "plan_history":"Plan change log, revision history, amendment timeline, past plan updates",
    "plan_finder":"Search/find available benefit plans, plan catalog lookup, plan matching",
    # ── claim_history_search (SEARCH/FILTER multiple claims) ─────────────────
    "compound_info":"Compound medication, MIC breakdown, ingredient costs",
    "date_range_claims":"Claims within date range, deductible claims, accumulation history",
    "drug_info":"Drug name, NDC, GPI, therapeutic class, formulary status",
    "drug_interaction_info":"DUR edits, drug interaction alerts, clinical screening",
    "fill_date_info":"Date prescription was filled, dispensing date, service date",
    "Refills":"Search claims by refill count, refill history, remaining refills",
    "DaysSupply":"Filter claims by days supply duration (7, 14, 30, 60, 90 days)",
    "PriorAuth":"Search claims that required prior authorization approval",
    "Diagnosis":"Filter claims by ICD-10 diagnosis code",
    "Settlement":"Search/filter claims by settlement response code NUMBER",
    "PharmType":"Filter claims by pharmacy type/channel (retail, mail-order, specialty)",
    "Plan":"Filter claims by insurance plan code",
    "Pharmacy":"Search claims from a specific pharmacy name/store/location",
    "Prescriber":"Search claims by prescriber name or NPI across claim history",
    "Pricing":"Search claims by cost/pricing for a specific DRUG across MULTIPLE claims",
    "Status":"Filter/list claims by status (paid, rejected, pending, reversed)",
    "RejectCode":"Search claims by NCPDP rejection code number",
    "DrugLast":"When was a specific drug last dispensed/filled for a member",
    "Month":"Filter claims by calendar month (January, February, etc.)",
    "ClaimNum":"Look up a specific claim by its claim number",
    "NDC":"Search claims by NDC (National Drug Code) number",
    "Manufacturer":"Filter claims by drug manufacturer name",
    "Generic":"Filter for generic drug claims only",
    "Brand":"Filter for brand name drug claims only",
    # ── general ──────────────────────────────────────────────────────────────
    "greeting":"Hello, hi, welcome, good morning/afternoon/evening",
    "help":"How to submit claims, steps to avoid rejection, filing guidance",
    "out_of_scope":"Unrelated to pharmacy — weather, recipes, sports, gibberish",
    # ── member_domain ────────────────────────────────────────────────────────
    "member_coverage":"Member coverage eligibility windows, active coverage status, enrollment dates",
    "member_hierarchy":"Client/CAG hierarchy, client-account-group membership, organizational structure",
    "benefit_reset_date":"Benefit year reset date, accumulator reset, plan year anniversary",
    "family_type":"Individual vs family plan classification, coverage tier type",
    "family_members":"List family members, dependents, subscriber and dependents on same plan",
    "alternate_insurance":"Other/secondary insurance on file, dual coverage, alternate payer",
    "medicare_coverage":"Medicare Part D enrollment status, Med-D plan assignment for a MEMBER",
    "lics_status":"Low Income Subsidy (LICS/LIS) status, subsidy level, cost-sharing reduction",
    "stcob_linkage":"Short-term COB linkage, STCOB member links and records",
    "cvs_id_lookup":"CVS ID associated with the member, CVS member identifier",
    "related_cagm":"Related CAGMs by CVS ID or family ID, linked CAGM records",
    "alternate_ids":"All alternate IDs on file for the member, cross-reference identifiers",
    "member_demographics":"Member name (first/last/middle), date of birth, gender, person code, relationship code",
    "member_contact_info":"Member email, phone number, mailing/postal address, city, state, zip code",
    "member_eligibility_copay":"Eligibility copay fields: copayBrand, copayGeneric, copay3, copay4",
    "member_transition_status":"Member transition fill status and start date from eligibility",
    "member_dur_config":"Drug utilization review (DUR) key and process flag, DUR configuration",
    "member_mbi_number":"Medicare Beneficiary Identifier (MBI) number from Part D record",
    "member_caretaker_info":"Caretaker name and address from Medicare Part D record",
    "member_language_pref":"Member language code/preference (mbrLangCode), communication language",
    "member_discount_program":"Discount program type assigned to the member",
    "member_override_plan":"Member-level override plan ID from eligibility (memberOverridePlan)",
    # ── override_domain ──────────────────────────────────────────────────────
    "pa_summary":"Prior authorization summary of key fields, PA overview and configuration",
    "pa_override_reject":"Will PA override specific reject codes (75, 70, 76), PA reject handling",
    "pa_field_help":"Explanation of what a specific PA field does, PA field documentation",
    "pa_copay_pricing":"PA copay override impact on pricing, copay influence on cost",
    "pa_drug_coverage":"Drugs covered by this PA (GPI/NDC lists), PA drug scope",
    "pa_claim_usage":"How many claims used/referenced this PA, PA utilization count",
    "pa_reason_code":"PA reason code (U1, LC, OD, OA, US, U3), override reason classification",
    "pa_effective_dates":"PA effective period begin/end dates, when PA is active or expired",
    "pa_agent_code":"Agent/source code on PA (A, C, 3, H), who created or modified the PA",
    "pa_ignore_status":"Ignore status code on PA (Y, P, 3), processing bypass indicator",
    "pa_specialty_rx_override":"Specialty Rx reject override indicator, bypass specialty rejection",
    "pa_clinical_admin_code":"Clinical administration code on PA (A, C, blank)",
    "pa_transform_care":"Transform care type on PA, care transformation program",
    "pa_follow_me_logic":"Follow me logic indicator, PA follows member across plan changes",
    "pa_drug_type_indicator":"Authorized drug type (G=GPI, N=NDC), drug matching method",
    "pa_modification_history":"PA modification date/time (modifyDateTime), last update timestamp",
}

# ── Load embeddings (with stale cache fix) ───────────────────────────────────
def load_embeddings():
    examples = embeddingVars.CVS_INTENT_EXAMPLES
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH) as f: cached = json.load(f)
        needed = set(INTENT_TO_DOMAIN.keys())
        missing = needed - set(cached.keys())
        if missing:
            logger.warning(f"Cache missing {len(missing)} intents: {sorted(missing)}")
            emb = get_embedder()
            for intent in sorted(missing):
                if intent in examples:
                    cached[intent] = [list(v) for v in emb.embed(examples[intent])]
            with open(EMBEDDINGS_PATH, "w") as f: json.dump(cached, f)
        return cached
    emb = get_embedder()
    cached = {i: [list(v) for v in emb.embed(s)] for i, s in examples.items()}
    with open(EMBEDDINGS_PATH, "w") as f: json.dump(cached, f)
    return cached

def build_Xy(embeddings, filter_intents=None):
    X, y, labels, lmap = [], [], [], {}
    for name in sorted(embeddings.keys()):
        if filter_intents and name not in filter_intents: continue
        if name not in lmap: lmap[name] = len(labels); labels.append(name)
        for vec in embeddings[name]: X.append(vec); y.append(lmap[name])
    return np.array(X), np.array(y), labels


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING DATA AUGMENTATION — fixes train/test distribution gap
# ═════════════════════════════════════════════════════════════════════════════
# Imported from the shared module to avoid duplication.
# To add/edit examples, modify: multidomain_intent_detection/augmented_examples.py

# Add parent directory to path so we can import multidomain_intent_detection
_parent_dir = os.path.dirname(BASE_DIR)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from multidomain_intent_detection.augmented_examples import AUGMENTED_EXAMPLES


def augment_embeddings(embeddings: dict) -> dict:
    """Add augmented training examples to cached embeddings.
    
    IMPORTANT: Normalizes examples (strips claim numbers) before embedding
    so they match the normalized test query space.
    """
    aug_cache_path = os.path.join(ARTIFACTS, "augmented_embeddings_v3.json")

    # Load previously generated augmented embeddings
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
                # Normalize before embedding — strip claim numbers
                normalized = [normalize_query(ex) for ex in examples]
                logger.info(f"  Augmenting '{intent}' with {len(examples)} normalized examples")
                aug_cached[intent] = [list(v) for v in emb.embed(normalized)]
        with open(aug_cache_path, "w") as f:
            json.dump(aug_cached, f)
        logger.info(f"Augmented embeddings saved → {aug_cache_path}")

    # Merge: add augmented examples to the main embeddings
    merged = {k: list(v) for k, v in embeddings.items()}
    for intent, vecs in aug_cached.items():
        if intent in merged:
            merged[intent] = merged[intent] + vecs
        else:
            merged[intent] = vecs

    return merged

class IntentPipeline:
    """PCA → Ensemble (SVM-RBF + LogReg + kNN) with learned temperature scaling."""

    # Known confusion pairs (intents with high embedding overlap)
    CONFUSION_PAIRS = {
        "approval_info":       {"prior_auth_info", "claim_status"},
        "prior_auth_info":     {"approval_info", "pa_summary"},
        "rejection_reasons":   {"help", "claim_status"},
        "help":                {"rejection_reasons"},
        "pricing_info":        {"compound_info", "medicare_part_d", "cob_info"},
        "compound_info":       {"pricing_info"},
        "claim_status":        {"approval_info", "rejection_reasons", "audit_info"},
        "generic_availability": {"Generic", "daw_info"},
        "fill_date_info":      {"rx_details", "audit_info"},
        "member_demographics": {"member_contact_info"},
        "member_contact_info": {"member_demographics"},
        "ClaimNum":            {"claim_status", "rx_details"},
        "settlement_info":     {"Settlement"},
        "Settlement":          {"settlement_info"},
        "beneficiary_info":    {"approval_info", "member_coverage"},
        "greeting":            {"out_of_scope"},
        "out_of_scope":        {"greeting"},
    }
    CONFUSION_PRONE_INTENTS = set(CONFUSION_PAIRS.keys())

    def __init__(self, n_pca=50, knn_k=5, temperature=1.5):
        self.n_pca = n_pca
        self.knn_k = knn_k
        self.temperature = temperature  # learned during fit(); >1 = softer
        self.pca = self.scaler = None
        self.clfs = {}
        self.label_names = []
        self.weights = {"svm": 0.40, "logreg": 0.35, "knn": 0.25}

    def fit(self, X_raw, y, label_names):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier

        self.label_names = label_names
        # L2 norm → PCA → scale
        X_n = X_raw / (np.linalg.norm(X_raw, axis=1, keepdims=True) + 1e-10)
        d = min(self.n_pca, X_raw.shape[0]-1, X_raw.shape[1])
        self.pca = PCA(n_components=d, whiten=True, random_state=42)
        X_p = self.pca.fit_transform(X_n)
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X_p)
        var_kept = self.pca.explained_variance_ratio_.sum()
        print(f"  PCA: 768 → {d} dims ({var_kept*100:.1f}% variance)")

        self.clfs["svm"] = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
            class_weight="balanced", random_state=42).fit(X_s,y)
        self.clfs["logreg"] = LogisticRegression(C=10, max_iter=3000,
            solver="lbfgs", class_weight="balanced", random_state=42).fit(X_s,y)
        self.clfs["knn"] = KNeighborsClassifier(n_neighbors=min(self.knn_k,X_raw.shape[0]-1),
            weights="distance", metric="cosine").fit(X_s,y)
        print(f"  Ensemble ready: SVM-RBF + LogReg + kNN")

        # Learn optimal temperature on held-out data
        self.temperature = self._learn_temperature(X_s, y)
        print(f"  Learned temperature: {self.temperature:.3f}")

    def _learn_temperature(self, X_scaled, y):
        """Find temperature T that minimizes NLL on held-out data (Guo et al. 2017)."""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from scipy.optimize import minimize_scalar

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        all_probs, all_labels = [], []
        for train_idx, val_idx in skf.split(X_scaled, y):
            lr = LogisticRegression(C=10, max_iter=3000, solver="lbfgs",
                class_weight="balanced", random_state=42).fit(X_scaled[train_idx], y[train_idx])
            all_probs.append(lr.predict_proba(X_scaled[val_idx]))
            all_labels.append(y[val_idx])
        all_probs = np.vstack(all_probs)
        all_labels = np.concatenate(all_labels)

        def nll(T):
            scaled = np.log(all_probs + 1e-12) / T
            scaled -= scaled.max(axis=1, keepdims=True)
            exp_scaled = np.exp(scaled)
            softmax = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
            correct_probs = softmax[np.arange(len(all_labels)), all_labels]
            return -np.log(correct_probs + 1e-12).mean()

        result = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
        return max(round(result.x, 3), 1.0)  # T >= 1.0: never sharpen

    def _apply_temperature(self, probs):
        """Apply learned temperature scaling to a probability vector."""
        log_p = np.log(probs + 1e-12) / self.temperature
        log_p -= log_p.max()
        exp_p = np.exp(log_p)
        return exp_p / exp_p.sum()

    def _transform(self, X):
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        return self.scaler.transform(self.pca.transform(X_n))

    def predict_proba(self, X):
        """Raw weighted ensemble probability (no temperature)."""
        X_f = self._transform(X)
        p = sum(clf.predict_proba(X_f) * self.weights[n] for n, clf in self.clfs.items())
        return p / (p.sum(axis=1, keepdims=True) + 1e-10)

    def predict_single(self, vec):
        """Predict with 3-layer calibration: temperature + disagreement + confusion-pair."""
        # Raw ensemble probabilities
        X_f = self._transform(vec.reshape(1,-1))
        raw_p = sum(
            clf.predict_proba(X_f) * self.weights[n]
            for n, clf in self.clfs.items()
        )[0]

        # Layer 1: Temperature scaling
        calibrated_p = self._apply_temperature(raw_p)
        idx = np.argsort(calibrated_p)[::-1]
        top5 = [(self.label_names[i], float(calibrated_p[i])) for i in idx[:5]]

        # Sub-classifier agreement
        indiv = {n: self.label_names[clf.predict(X_f)[0]] for n, clf in self.clfs.items()}
        agreement = len(set(indiv.values())) == 1

        raw_confidence = float(raw_p[idx[0]])
        confidence = float(calibrated_p[idx[0]])
        margin = float(calibrated_p[idx[0]] - calibrated_p[idx[1]]) if len(idx) > 1 else 1.0

        # Layer 2: Disagreement penalty
        if not agreement:
            n_unique = len(set(indiv.values()))
            penalty = 1.0 - (n_unique - 1) * 0.15
            confidence *= penalty
            margin *= penalty

        # Layer 3: Confusion-pair penalty
        top1 = self.label_names[idx[0]]
        top2 = self.label_names[idx[1]] if len(idx) > 1 else ""
        is_confusion_pair = (
            top1 in self.CONFUSION_PAIRS
            and top2 in self.CONFUSION_PAIRS.get(top1, set())
        )
        if is_confusion_pair:
            confidence *= 0.80
            margin *= 0.80

        return {
            "intent": top1,
            "confidence": confidence,
            "raw_confidence": raw_confidence,
            "margin": margin,
            "top_5": top5,
            "individual": indiv,
            "agreement": agreement,
            "is_confusion_pair": is_confusion_pair,
        }

    def cross_validate(self, X_raw, y):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for tr, va in cv.split(X_raw, y):
            Xtr, Xva, ytr, yva = X_raw[tr], X_raw[va], y[tr], y[va]
            fp = IntentPipeline(self.n_pca, self.knn_k, self.temperature)
            fp.label_names = self.label_names
            Xn = Xtr / (np.linalg.norm(Xtr,axis=1,keepdims=True)+1e-10)
            d = min(self.n_pca, Xtr.shape[0]-1, Xtr.shape[1])
            fp.pca = PCA(n_components=d, whiten=True, random_state=42)
            Xp = fp.pca.fit_transform(Xn)
            fp.scaler = StandardScaler()
            Xs = fp.scaler.fit_transform(Xp)
            fp.clfs["svm"] = SVC(kernel="rbf",C=10,gamma="scale",probability=True,
                class_weight="balanced",random_state=42).fit(Xs,ytr)
            fp.clfs["logreg"] = LogisticRegression(C=10,max_iter=3000,
                solver="lbfgs",class_weight="balanced",random_state=42).fit(Xs,ytr)
            fp.clfs["knn"] = KNeighborsClassifier(n_neighbors=min(self.knn_k,Xtr.shape[0]-1),
                weights="distance",metric="cosine").fit(Xs,ytr)
            preds = np.argmax(fp.predict_proba(Xva), axis=1)
            accs.append((preds == yva).mean())
        return np.mean(accs), np.std(accs), accs


# ═════════════════════════════════════════════════════════════════════════════
# PCA DIM SEARCH
# ═════════════════════════════════════════════════════════════════════════════

def search_pca(X, y, labels):
    print(f"\n{'='*60}")
    print(f"  PCA DIMENSION SEARCH (5-fold CV)")
    print(f"{'='*60}")
    print(f"  {'Dims':>8} {'CV Acc':>10} {'Std':>8}")
    print(f"  {'-'*28}")
    best_d, best_a = 50, 0
    for d in [20,30,40,50,75,100,150,200,250,300]:
        if d >= X.shape[0]: continue
        p = IntentPipeline(n_pca=d); p.label_names = labels
        a, s, _ = p.cross_validate(X, y)
        star = " <-- BEST" if a > best_a else ""
        print(f"  {d:>8} {a*100:>8.2f}% {s*100:>6.2f}%{star}")
        if a > best_a: best_a = a; best_d = d
    print(f"\n  Optimal: PCA-{best_d} ({best_a*100:.2f}%)")
    print(f"{'='*60}\n")
    return best_d


# ═════════════════════════════════════════════════════════════════════════════
# ABLATION: COMPARE EVERYTHING
# ═════════════════════════════════════════════════════════════════════════════

def run_ablation(X, y, labels, best_dim):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    Xn = X / (np.linalg.norm(X,axis=1,keepdims=True)+1e-10)
    pca = PCA(n_components=best_dim, whiten=True, random_state=42)
    Xp = pca.fit_transform(Xn)
    sc = StandardScaler()
    Xs = sc.fit_transform(Xp)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    algos = {
        "kNN-7 (raw 768d, cosine)": (KNeighborsClassifier(7, weights="distance", metric="cosine"), Xn),
        "kNN-7 (PCA, cosine)": (KNeighborsClassifier(7, weights="distance", metric="cosine"), Xs),
        "SVM-Linear (PCA)": (SVC(kernel="linear",C=1,probability=True,class_weight="balanced",random_state=42), Xs),
        "SVM-RBF (PCA)": (SVC(kernel="rbf",C=10,gamma="scale",probability=True,class_weight="balanced",random_state=42), Xs),
        "LogReg (PCA)": (LogisticRegression(C=10,max_iter=3000,solver="lbfgs",class_weight="balanced",random_state=42), Xs),
        "kNN-5 (PCA, cosine)": (KNeighborsClassifier(5, weights="distance", metric="cosine"), Xs),
    }

    print(f"\n{'='*60}")
    print(f"  ALGORITHM COMPARISON (5-fold CV, PCA-{best_dim})")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<30} {'CV Acc':>10} {'Std':>8}")
    print(f"  {'-'*50}")
    for name, (clf, Xuse) in algos.items():
        sc2 = cross_val_score(clf, Xuse, y, cv=cv, scoring="accuracy")
        print(f"  {name:<30} {sc2.mean()*100:>8.2f}% {sc2.std()*100:>6.2f}%")

    pipe = IntentPipeline(n_pca=best_dim); pipe.label_names = labels
    ea, es, _ = pipe.cross_validate(X, y)
    print(f"  {'ENSEMBLE (SVM+LR+kNN)':<30} {ea*100:>8.2f}% {es*100:>6.2f}%  ★")
    print(f"{'='*60}\n")


# ═════════════════════════════════════════════════════════════════════════════
# LLM FALLBACK (mirrors llm_judge_node from nodes/llm_judge.py)
# ═════════════════════════════════════════════════════════════════════════════

def llm_classify(query, candidates, ensemble_intent=None, ensemble_confidence=0.0, top5_with_probs=None):
    """Domain-aware LLM fallback classifier.
    
    Uses the domain-specific prompt modules from prompt_templates/domain_prompts/
    for highly accurate intent classification. Each domain has an expert-level
    prompt with exhaustive disambiguation rules, decision trees, and confusion
    pair resolution.
    
    Args:
        query: User's natural language query (original, with claim numbers)
        candidates: Top-5 intent names from ensemble (ordered by probability)
        ensemble_intent: The ensemble's top prediction (for context)
        ensemble_confidence: The ensemble's confidence (for context)
        top5_with_probs: List of (intent_name, probability) tuples from ensemble
        
    Returns:
        Predicted intent name
    """
    # Add parent directory to path so we can import prompt_templates
    parent_dir = os.path.dirname(BASE_DIR)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from prompt_templates.domain_prompts.llm_fallback import llm_fallback_classify
    
    # Build top5 with probabilities if not provided
    if top5_with_probs is None:
        # Fallback: assign decreasing dummy probabilities
        top5_with_probs = [(c, max(0.1, ensemble_confidence - i*0.1)) for i, c in enumerate(candidates)]
    
    result = llm_fallback_classify(
        query=query,
        top5_intents=top5_with_probs,
        ensemble_intent=ensemble_intent,
        ensemble_confidence=ensemble_confidence,
    )
    
    return result.get("intent", candidates[0] if candidates else "unknown")


def evaluate(test_data, pipeline, embedder, use_llm=True, conf_t=0.30, margin_t=0.05):
    results, llm_n = [], 0
    gate_stats = {"ensemble_pass": 0, "disagreement_to_llm": 0,
                  "low_conf_to_llm": 0, "confusion_pair_to_llm": 0}

    for idx, rec in enumerate(test_data):
        normalized_text = normalize_query(rec["text"])
        vec = np.array(embedder.embed(normalized_text))
        pred = pipeline.predict_single(vec)

        # ── Confusion-aware gating ───────────────────────────────────
        confident = (
            pred["confidence"] >= conf_t
            and pred["margin"] >= margin_t
            and pred["agreement"]
        )
        gate_reason = "pass"

        if not pred["agreement"]:
            confident = False
            gate_reason = "disagreement"
        elif confident and pred["intent"] in IntentPipeline.CONFUSION_PRONE_INTENTS:
            if not (pred["confidence"] >= 0.55 and pred["margin"] >= 0.20):
                confident = False
                gate_reason = "confusion_prone"
        if confident and pred.get("is_confusion_pair", False):
            if not (pred["confidence"] >= 0.60 and pred["margin"] >= 0.25):
                confident = False
                gate_reason = "confusion_pair"

        if confident:
            gate_stats["ensemble_pass"] += 1
        elif gate_reason == "disagreement":
            gate_stats["disagreement_to_llm"] += 1
        elif gate_reason == "confusion_pair":
            gate_stats["confusion_pair_to_llm"] += 1
        else:
            gate_stats["low_conf_to_llm"] += 1

        if confident or not use_llm:
            final, src = pred["intent"], "ensemble"
        else:
            final = llm_classify(
                rec["text"],
                [n for n, _ in pred["top_5"]],
                ensemble_intent=pred["intent"],
                ensemble_confidence=pred["confidence"],
                top5_with_probs=pred["top_5"],
            )
            src = "llm"; llm_n += 1

        results.append({
            "text": rec["text"],
            "actual_intent": rec["actual_intent"], "predicted_intent": final,
            "intent_match": rec["actual_intent"] == final,
            "actual_domain": rec["actual_domain"],
            "predicted_domain": INTENT_TO_DOMAIN.get(final,"unknown"),
            "domain_match": rec["actual_domain"] == INTENT_TO_DOMAIN.get(final,"unknown"),
            "confidence": pred["confidence"],
            "raw_confidence": pred.get("raw_confidence", pred["confidence"]),
            "margin": pred["margin"], "source": src,
            "agreement": pred["agreement"],
            "is_confusion_pair": pred.get("is_confusion_pair", False),
            "gate_reason": gate_reason,
        })
        if (idx+1) % 50 == 0: logger.info(f"  {idx+1}/{len(test_data)}")

    df = pd.DataFrame(results)
    mode = "hybrid" if use_llm else "ensemble_only"
    df.to_csv(os.path.join(OUTPUTS, f"results_{mode}.csv"), index=False)

    ia, da = df["intent_match"].mean()*100, df["domain_match"].mean()*100
    errors = df[~df["intent_match"]]
    high_conf_errors = errors[errors["raw_confidence"] >= 0.85]

    print(f"\n{'='*65}")
    print(f"  {'ENSEMBLE + LLM' if use_llm else 'ENSEMBLE ONLY'}")
    print(f"  Intent Accuracy    : {ia:.2f}%")
    print(f"  Domain Accuracy    : {da:.2f}%")
    print(f"  Total Errors       : {len(errors)}")
    print(f"  High-conf Errors   : {len(high_conf_errors)} (raw conf >= 0.85)")
    if use_llm:
        ep = (len(test_data)-llm_n)/len(test_data)*100
        print(f"  Ensemble resolved  : {ep:.1f}% ({len(test_data)-llm_n}/{len(test_data)})")
        print(f"  LLM calls          : {llm_n} ({llm_n/len(test_data)*100:.1f}%)")
    print(f"\n  Gate Statistics:")
    for k, v in gate_stats.items():
        print(f"    {k:25s}: {v}")
    print(f"{'='*65}")

    print(f"\n  {'Domain':<25} {'Intent':>8} {'Domain':>8} {'Count':>6}")
    print(f"  {'-'*50}")
    for dom in sorted(df["actual_domain"].unique()):
        s = df[df["actual_domain"]==dom]
        print(f"  {dom:<25} {s['intent_match'].mean()*100:>6.1f}% {s['domain_match'].mean()*100:>6.1f}% {len(s):>6}")

    wrong = df[~df["intent_match"]]
    if len(wrong):
        print(f"\n  Top Confusions:")
        for (a,p),c in wrong.groupby(["actual_intent","predicted_intent"]).size().sort_values(ascending=False).head(10).items():
            print(f"    {a} -> {p}: {c}")

    if use_llm and llm_n:
        lr = df[df["source"]=="llm"]
        print(f"\n  LLM accuracy: {lr['intent_match'].mean()*100:.1f}% ({int(lr['intent_match'].sum())}/{llm_n})")

    print()
    return {"intent_accuracy": ia, "domain_accuracy": da, "gate_stats": gate_stats}


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*65)
    print("  Intent Detection v3 — All Domains")
    print("  PCA + Ensemble (SVM-RBF / LogReg / kNN) + LLM Fallback")
    print("  Domains: cap_api, benefits_api, claim_history_search,")
    print("           member_domain, override_domain, general")
    print("="*65)

    print("\nStep 1 — Loading cached embeddings...")
    all_emb = load_embeddings()

    # Use corrected test data if available (fixes 25 mislabeled queries)
    TESTDATA_CORRECTED = os.path.join(BASE_DIR, "Testdata_corrected.csv")
    TESTDATA_ORIGINAL = os.path.join(BASE_DIR, "Testdata.csv")
    if os.path.exists(TESTDATA_CORRECTED):
        TESTDATA = TESTDATA_CORRECTED
        print(f"  Using CORRECTED test data: {TESTDATA_CORRECTED}")
    elif os.path.exists(TESTDATA_ORIGINAL):
        TESTDATA = TESTDATA_ORIGINAL
        print(f"  Using original test data: {TESTDATA_ORIGINAL}")
        print(f"  WARNING: Original data has ~25 mislabeled queries. Run _audit_labels.py to fix.")
    else:
        print("Testdata.csv not found"); sys.exit(1)
    tdf = pd.read_csv(TESTDATA)
    test_data = [{"text":r["Prompt"],"actual_intent":r["Intent"],"actual_domain":r["domain"]} for _,r in tdf.iterrows()]
    print(f"  Test: {len(test_data)} queries, {tdf['Intent'].nunique()} intents")

    train_intents = set(all_emb.keys()) & set(INTENT_TO_DOMAIN.keys())

    # Augment training data with real-world phrasing patterns
    print(f"\nStep 2 — Augmenting training data with real-world examples...")
    augmented_emb = augment_embeddings(all_emb)
    X, y, labels = build_Xy(augmented_emb, train_intents)
    print(f"  {X.shape[0]} samples x {X.shape[1]} dims, {len(labels)} classes")

    print("\nStep 3 — Finding optimal PCA dimensions...")
    best_dim = search_pca(X, y, labels)

    print("\nStep 4 — Algorithm comparison...")
    run_ablation(X, y, labels, best_dim)

    print(f"\nStep 5 — Training final ensemble (PCA-{best_dim})...")
    pipe = IntentPipeline(n_pca=best_dim)
    pipe.fit(X, y, labels)
    with open(MODEL_PKL, "wb") as f: pickle.dump(pipe, f)
    print(f"  Saved → {MODEL_PKL}")

    print("\nStep 6 — Evaluation (ensemble only, no API calls)...")
    embedder = get_embedder()
    m1 = evaluate(test_data, pipe, embedder, use_llm=False)

    if "--no-llm" not in sys.argv:
        print("\nStep 7 — Evaluation (ensemble + LLM fallback)...")
        m2 = evaluate(test_data, pipe, embedder, use_llm=True)
        print(f"\n{'='*60}")
        print(f"  FINAL: Ensemble {m1['intent_accuracy']:.1f}% → +LLM {m2['intent_accuracy']:.1f}%")
        print(f"{'='*60}")
