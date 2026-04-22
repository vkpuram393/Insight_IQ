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

import os, sys, json, logging, time, pickle
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
    "claim_status":"cap_api","multi_claim_summary":"cap_api","pharmacy_info":"cap_api",
    "prescriber_info":"cap_api","pricing_info":"cap_api","reimbursement_info":"cap_api",
    "rejection_reasons":"cap_api","settlement_info":"cap_api","rx_details":"cap_api",
    "reversal_info":"cap_api","cob_info":"cap_api","generic_availability":"cap_api",
    "daw_info":"cap_api","government_claim_type":"cap_api","mail_order_info":"cap_api",
    "medicare_part_d":"cap_api","network_info":"cap_api","prior_auth_info":"cap_api",
    "approval_info":"benefits_api","audit_info":"benefits_api","beneficiary_info":"benefits_api",
    "compound_info":"claim_history_search","date_range_claims":"claim_history_search",
    "drug_info":"claim_history_search","drug_interaction_info":"claim_history_search",
    "fill_date_info":"claim_history_search",
    "greeting":"general","help":"general","out_of_scope":"general",
}

INTENT_DESC = {
    "claim_status":"General claim status, adjudication outcome, paid/rejected/pending",
    "multi_claim_summary":"Summary of ALL/MULTIPLE claims for a member",
    "pharmacy_info":"Dispensing pharmacy name, location, address, NCPDP, store",
    "prescriber_info":"Prescribing physician/doctor name, NPI, credentials",
    "pricing_info":"Copay, ingredient cost, dispensing fee, patient pay, cost breakdown",
    "reimbursement_info":"Amount paid TO pharmacy, reimbursement rationale, payment",
    "rejection_reasons":"Rejection codes, failed edits, denial reasons, how to resolve",
    "settlement_info":"Settlement codes, pharmacy response/feedback codes",
    "rx_details":"RX number, fill number, quantity, days supply, strength",
    "reversal_info":"Claim reversal, R&R, manual adjustments, resubmission",
    "cob_info":"Coordination of Benefits, other insurance, secondary payer, dual coverage",
    "generic_availability":"Generic alternatives, therapeutic equivalents, formulary substitutes",
    "approval_info":"Claim approval, plan overrides, transition fill (TF), BPG, Smart PA",
    "audit_info":"Audit trail, change history, modification records, timestamps",
    "beneficiary_info":"Member benefit phase, coverage tier, eligibility, accumulations",
    "compound_info":"Compound medication, MIC breakdown, ingredient costs",
    "date_range_claims":"Claims within date range, deductible claims, accumulation history",
    "drug_info":"Drug name, NDC, GPI, therapeutic class, formulary status",
    "drug_interaction_info":"DUR edits, drug interaction alerts, clinical screening",
    "fill_date_info":"Date prescription was filled, dispensing date, service date",
    "greeting":"Hello, hi, welcome, good morning/afternoon/evening",
    "help":"How to submit claims, steps to avoid rejection, filing guidance",
    "out_of_scope":"Unrelated to pharmacy — weather, recipes, sports, gibberish",
    "daw_info":"DAW status, brand vs generic requirement, substitution",
    "government_claim_type":"Medicare/Medicaid claim type, government program",
    "mail_order_info":"Mail order/home delivery prescription, shipping",
    "medicare_part_d":"Medicare Part D summary, PDE, MEDD pricing, LICS",
    "network_info":"Pharmacy network details, which network processed claim",
    "prior_auth_info":"Prior authorization status, Smart PA, authorization requirements",
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
# CORE PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

class IntentPipeline:
    """PCA → Ensemble (SVM-RBF + LogReg + kNN) with calibrated probabilities."""

    def __init__(self, n_pca=50, knn_k=5):
        self.n_pca = n_pca
        self.knn_k = knn_k
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
        self.clfs["logreg"] = LogisticRegression(C=10, max_iter=3000, multi_class="multinomial",
            solver="lbfgs", class_weight="balanced", random_state=42).fit(X_s,y)
        self.clfs["knn"] = KNeighborsClassifier(n_neighbors=min(self.knn_k,X_raw.shape[0]-1),
            weights="distance", metric="cosine").fit(X_s,y)
        print(f"  Ensemble ready: SVM-RBF + LogReg + kNN")

    def _transform(self, X):
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        return self.scaler.transform(self.pca.transform(X_n))

    def predict_proba(self, X):
        X_f = self._transform(X)
        p = sum(clf.predict_proba(X_f) * self.weights[n] for n, clf in self.clfs.items())
        return p / (p.sum(axis=1, keepdims=True) + 1e-10)

    def predict_single(self, vec):
        p = self.predict_proba(vec.reshape(1,-1))[0]
        idx = np.argsort(p)[::-1]
        top5 = [(self.label_names[i], float(p[i])) for i in idx[:5]]
        X_f = self._transform(vec.reshape(1,-1))
        indiv = {n: self.label_names[clf.predict(X_f)[0]] for n, clf in self.clfs.items()}
        return {
            "intent": self.label_names[idx[0]],
            "confidence": float(p[idx[0]]),
            "margin": float(p[idx[0]] - p[idx[1]]) if len(idx) > 1 else 1.0,
            "top_5": top5,
            "individual": indiv,
            "agreement": len(set(indiv.values())) == 1,
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
            fp = IntentPipeline(self.n_pca, self.knn_k)
            fp.label_names = self.label_names
            Xn = Xtr / (np.linalg.norm(Xtr,axis=1,keepdims=True)+1e-10)
            d = min(self.n_pca, Xtr.shape[0]-1, Xtr.shape[1])
            fp.pca = PCA(n_components=d, whiten=True, random_state=42)
            Xp = fp.pca.fit_transform(Xn)
            fp.scaler = StandardScaler()
            Xs = fp.scaler.fit_transform(Xp)
            fp.clfs["svm"] = SVC(kernel="rbf",C=10,gamma="scale",probability=True,
                class_weight="balanced",random_state=42).fit(Xs,ytr)
            fp.clfs["logreg"] = LogisticRegression(C=10,max_iter=3000,multi_class="multinomial",
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
    for d in [20,30,40,50,75,100,150,200]:
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
        "LogReg (PCA)": (LogisticRegression(C=10,max_iter=3000,multi_class="multinomial",solver="lbfgs",class_weight="balanced",random_state=42), Xs),
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
# LLM FALLBACK
# ═════════════════════════════════════════════════════════════════════════════

def llm_classify(query, candidates):
    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True,
        project=os.getenv("PROJECT_ID","pbm-poc-coderev-genai-poc"),
        location=os.getenv("LOCATION","us-central1"))
    desc = "\n".join(f"- {n}: {INTENT_DESC.get(n,n)}" for n in candidates)
    prompt = f"Classify this pharmacy query into ONE intent.\n\nINTENTS:\n{desc}\n\nQUERY: {query}\n\nReply ONLY the intent name."
    try:
        r = client.models.generate_content(model="gemini-2.0-flash", contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=30))
        pred = r.text.strip().strip('"').strip("'")
        for c in candidates:
            if c.lower() == pred.lower(): return c
        return candidates[0]
    except: return candidates[0]


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(test_data, pipeline, embedder, use_llm=True, conf_t=0.45, margin_t=0.12):
    results, llm_n = [], 0
    for idx, rec in enumerate(test_data):
        vec = np.array(embedder.embed(rec["text"]))
        pred = pipeline.predict_single(vec)
        confident = pred["confidence"] >= conf_t and pred["margin"] >= margin_t

        if confident or not use_llm:
            final, src = pred["intent"], "ensemble"
        else:
            final = llm_classify(rec["text"], [n for n,_ in pred["top_5"]])
            src = "llm"; llm_n += 1

        results.append({
            "text": rec["text"],
            "actual_intent": rec["actual_intent"], "predicted_intent": final,
            "intent_match": rec["actual_intent"] == final,
            "actual_domain": rec["actual_domain"],
            "predicted_domain": INTENT_TO_DOMAIN.get(final,"unknown"),
            "domain_match": rec["actual_domain"] == INTENT_TO_DOMAIN.get(final,"unknown"),
            "confidence": pred["confidence"], "margin": pred["margin"], "source": src,
            "agreement": pred["agreement"],
        })
        if (idx+1) % 50 == 0: logger.info(f"  {idx+1}/{len(test_data)}")

    df = pd.DataFrame(results)
    mode = "hybrid" if use_llm else "ensemble_only"
    df.to_csv(os.path.join(OUTPUTS, f"results_{mode}.csv"), index=False)

    ia, da = df["intent_match"].mean()*100, df["domain_match"].mean()*100
    print(f"\n{'='*60}")
    print(f"  {'ENSEMBLE + LLM' if use_llm else 'ENSEMBLE ONLY'}")
    print(f"  Intent Accuracy : {ia:.2f}%")
    print(f"  Domain Accuracy : {da:.2f}%")
    if use_llm:
        ep = (len(test_data)-llm_n)/len(test_data)*100
        print(f"  Ensemble resolved : {ep:.1f}% ({len(test_data)-llm_n}/{len(test_data)})")
        print(f"  LLM calls         : {llm_n} ({llm_n/len(test_data)*100:.1f}%)")
    print(f"{'='*60}")

    print(f"\n  {'Domain':<25} {'Intent':>8} {'Domain':>8} {'Count':>6}")
    print(f"  {'-'*50}")
    for dom in sorted(df["actual_domain"].unique()):
        s = df[df["actual_domain"]==dom]
        print(f"  {dom:<25} {s['intent_match'].mean()*100:>6.1f}% {s['domain_match'].mean()*100:>6.1f}% {len(s):>6}")

    wrong = df[~df["intent_match"]]
    if len(wrong):
        print(f"\n  Top Confusions:")
        for (a,p),c in wrong.groupby(["actual_intent","predicted_intent"]).size().sort_values(ascending=False).head(10).items():
            print(f"    {a} → {p}: {c}")

    if use_llm and llm_n:
        lr = df[df["source"]=="llm"]
        print(f"\n  LLM accuracy: {lr['intent_match'].mean()*100:.1f}% ({int(lr['intent_match'].sum())}/{llm_n})")
    print()
    return {"intent_accuracy": ia, "domain_accuracy": da}


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*65)
    print("  Intent Detection v3")
    print("  PCA + Ensemble (SVM-RBF / LogReg / kNN) + LLM Fallback")
    print("="*65)

    print("\nStep 1 — Loading cached embeddings...")
    all_emb = load_embeddings()

    TESTDATA = os.path.join(BASE_DIR, "Testdata.csv")
    if not os.path.exists(TESTDATA): print("Testdata.csv not found"); sys.exit(1)
    tdf = pd.read_csv(TESTDATA)
    test_data = [{"text":r["Prompt"],"actual_intent":r["Intent"],"actual_domain":r["domain"]} for _,r in tdf.iterrows()]
    print(f"  Test: {len(test_data)} queries, {tdf['Intent'].nunique()} intents")

    train_intents = set(all_emb.keys()) & set(INTENT_TO_DOMAIN.keys())
    print(f"\nStep 2 — Building training data ({len(train_intents)} intents)...")
    X, y, labels = build_Xy(all_emb, train_intents)
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
