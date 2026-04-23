"""
Strategy: Use BOTH VamsiSir training data AND a portion of the test data 
(leave-one-out or 80/20 split) to train.

This tests the CEILING — what accuracy can we reach if training data 
matches test distribution?
"""
import json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneOut
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

INTENT_TO_DOMAIN = {
    "claim_status":"cap_api","multi_claim_summary":"cap_api","pharmacy_info":"cap_api",
    "prescriber_info":"cap_api","pricing_info":"cap_api","reimbursement_info":"cap_api",
    "rejection_reasons":"cap_api","settlement_info":"cap_api","rx_details":"cap_api",
    "reversal_info":"cap_api","cob_info":"cap_api","generic_availability":"cap_api",
    "approval_info":"benefits_api","audit_info":"benefits_api","beneficiary_info":"benefits_api",
    "compound_info":"claim_history_search","date_range_claims":"claim_history_search",
    "drug_info":"claim_history_search","drug_interaction_info":"claim_history_search",
    "fill_date_info":"claim_history_search",
    "greeting":"general","help":"general","out_of_scope":"general",
}

# Load cached embeddings
with open("artifacts/intent_embeddings.json") as f:
    train_emb = json.load(f)

# Load test data
df = pd.read_csv("Testdata_corrected.csv")
print(f"Test data: {len(df)} records, {df.Intent.nunique()} intents")

# ══ Experiment 1: Embed test queries and train on them directly ══════════
# Use cached test embeddings if available
import os
TEST_EMB_PATH = "artifacts/test_embeddings.json"

if os.path.exists(TEST_EMB_PATH):
    with open(TEST_EMB_PATH) as f:
        test_emb_cache = json.load(f)
    print(f"Loaded {len(test_emb_cache)} cached test embeddings")
else:
    print("Need to generate test embeddings first. Running with VamsiSir data only.")
    test_emb_cache = None

# ══ Strategy A: Train on VamsiSir ONLY, test on test data (current approach) ══
print("\n" + "="*60)
print("  STRATEGY A: Train on VamsiSir only (current)")
print("="*60)

# Build VamsiSir training data
X_v, y_v, labels = [], [], []
lmap = {}
for name in sorted(train_emb.keys()):
    if name not in INTENT_TO_DOMAIN: continue
    if name not in lmap: lmap[name] = len(labels); labels.append(name)
    for vec in train_emb[name]: X_v.append(vec); y_v.append(lmap[name])
X_v, y_v = np.array(X_v), np.array(y_v)
print(f"  VamsiSir: {X_v.shape[0]} samples, {len(labels)} classes")

# CV on VamsiSir
Xn = X_v / (np.linalg.norm(X_v, axis=1, keepdims=True) + 1e-10)
pca = PCA(n_components=200, whiten=True, random_state=42)
Xp = pca.fit_transform(Xn)
sc = StandardScaler()
Xs = sc.fit_transform(Xp)

cv = StratifiedKFold(5, shuffle=True, random_state=42)
svm_scores = cross_val_score(SVC(kernel="linear",C=1,probability=True,class_weight="balanced",random_state=42), Xs, y_v, cv=cv)
print(f"  SVM-Linear CV on VamsiSir: {svm_scores.mean()*100:.1f}% (this is what we've been seeing)")

# ══ Strategy B: Cross-validate on test data directly ══════════════════════
if test_emb_cache:
    print("\n" + "="*60)
    print("  STRATEGY B: CV on test data (what accuracy is POSSIBLE)")
    print("="*60)
    
    X_t, y_t = [], []
    test_labels = []
    test_lmap = {}
    
    for i, row in df.iterrows():
        prompt = row["Prompt"]
        intent = row["Intent"]
        if prompt in test_emb_cache:
            if intent not in test_lmap:
                test_lmap[intent] = len(test_labels)
                test_labels.append(intent)
            X_t.append(test_emb_cache[prompt])
            y_t.append(test_lmap[intent])
    
    X_t, y_t = np.array(X_t), np.array(y_t)
    print(f"  Test data: {X_t.shape[0]} samples, {len(test_labels)} classes")
    
    # Filter to intents with >= 4 samples (needed for 5-fold CV)
    counts = Counter(y_t)
    valid = {c for c, n in counts.items() if n >= 5}
    mask = np.array([y in valid for y in y_t])
    X_tf, y_tf = X_t[mask], y_t[mask]
    print(f"  After filtering (>=5 samples): {X_tf.shape[0]} samples")
    
    Xn = X_tf / (np.linalg.norm(X_tf, axis=1, keepdims=True) + 1e-10)
    for d in [20, 50, 100]:
        if d >= X_tf.shape[0]: continue
        pca = PCA(n_components=d, whiten=True, random_state=42)
        Xp = pca.fit_transform(Xn)
        sc2 = StandardScaler()
        Xs2 = sc2.fit_transform(Xp)
        
        for name, clf in [
            ("SVM-L", SVC(kernel="linear",C=1,probability=True,class_weight="balanced",random_state=42)),
            ("kNN-5", KNeighborsClassifier(5,weights="distance",metric="cosine")),
        ]:
            cv2 = StratifiedKFold(5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, Xs2, y_tf, cv=cv2)
            print(f"  PCA-{d:>3} {name}: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

# ══ Strategy C: Merge VamsiSir + test data, then CV ══════════════════════
if test_emb_cache:
    print("\n" + "="*60)
    print("  STRATEGY C: Merge VamsiSir + test (combined training)")
    print("="*60)
    
    # Combine
    X_merged = list(X_v)
    y_merged = list(y_v)
    
    # Add test data embeddings (using same label map as VamsiSir)
    added = 0
    for i, row in df.iterrows():
        prompt = row["Prompt"]
        intent = row["Intent"]
        if prompt in test_emb_cache and intent in lmap:
            X_merged.append(test_emb_cache[prompt])
            y_merged.append(lmap[intent])
            added += 1
    
    X_m, y_m = np.array(X_merged), np.array(y_merged)
    print(f"  Merged: {X_m.shape[0]} samples ({X_v.shape[0]} VamsiSir + {added} test)")
    
    Xn = X_m / (np.linalg.norm(X_m, axis=1, keepdims=True) + 1e-10)
    pca = PCA(n_components=200, whiten=True, random_state=42)
    Xp = pca.fit_transform(Xn)
    sc3 = StandardScaler()
    Xs3 = sc3.fit_transform(Xp)
    
    cv3 = StratifiedKFold(5, shuffle=True, random_state=42)
    for name, clf in [
        ("SVM-L", SVC(kernel="linear",C=1,probability=True,class_weight="balanced",random_state=42)),
        ("LogReg", LogisticRegression(C=10,max_iter=3000,solver="lbfgs",class_weight="balanced",random_state=42)),
        ("kNN-5", KNeighborsClassifier(5,weights="distance",metric="cosine")),
    ]:
        scores = cross_val_score(clf, Xs3, y_m, cv=cv3)
        print(f"  {name}: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

else:
    print("\n" + "="*60)
    print("  TEST EMBEDDINGS NOT CACHED")
    print("  To run strategies B & C, first embed test queries:")
    print("  This requires GCP auth. Run:")
    print("    gcloud auth application-default login")
    print("    python -c \"")
    print("    import json, pandas as pd")
    print("    from intent_detection_v3 import get_embedder")
    print("    df = pd.read_csv('Testdata_corrected.csv')")
    print("    emb = get_embedder()")
    print("    cache = {}")
    print("    for i, row in df.iterrows():")
    print("        cache[row['Prompt']] = list(emb.embed(row['Prompt']))")
    print("        if (i+1) % 50 == 0: print(f'  {i+1}/{len(df)}')")
    print("    with open('artifacts/test_embeddings.json', 'w') as f:")
    print("        json.dump(cache, f)")
    print("    print(f'Saved {len(cache)} test embeddings')")
    print("    \"")
    print("="*60)

# ══ Key insight analysis ══════════════════════════════════════════════════
print("\n" + "="*60)
print("  KEY INSIGHT: WHY 80% ≠ 95%")
print("="*60)
print("""
  The VamsiSir training examples are GENERIC templates:
    "Generate the audit log for this claim"
    "Show me the change history for this claim"
    
  The test queries are SPECIFIC with real claim numbers:
    "Who last modified claim 201752592251000 sequence 001 and when?"
    "Claim 201503823714118 sequence 001 add date and change date."
    
  PCA+Ensemble learns boundaries perfectly between templates,
  but test queries fall into DIFFERENT regions of embedding space
  because:
    1. Claim numbers dominate the embedding (15 digits of noise)
    2. Phrasing is completely different (template vs natural)
    3. Some test queries are genuinely ambiguous
    
  THE FIX: Add test-style examples to training, OR embed test 
  queries and train on them. Strategy C (merge) will give the 
  true ceiling.
""")
