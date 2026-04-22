"""
Rigorous validation of the PCA+Ensemble approach.
Tests:
  1. Per-intent CV accuracy (which intents drop below 90%?)
  2. Cross-validation confusion matrix (which pairs get confused?)
  3. Stability test (10 different random seeds)
  4. Leave-2-Out stress test (train on 18, test on 2 per intent)
  5. PCA sensitivity analysis
All offline — uses only cached embeddings, no API calls.
"""
import json, numpy as np
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from intent_detection_v3 import INTENT_TO_DOMAIN, build_Xy

# ── Load data ────────────────────────────────────────────────────────────────
with open("artifacts/intent_embeddings.json") as f:
    emb = json.load(f)

train_intents = set(emb.keys()) & set(INTENT_TO_DOMAIN.keys())
X, y, labels = build_Xy(emb, train_intents)
print(f"Data: {X.shape[0]} samples, {X.shape[1]} dims, {len(labels)} classes")
print(f"Samples per class: {Counter(y).most_common()[:5]}... (all ~20)")


def build_ensemble(Xtr, ytr, n_pca=200):
    """Build PCA+Ensemble pipeline on training data."""
    Xn = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-10)
    d = min(n_pca, Xtr.shape[0]-1, Xtr.shape[1])
    pca = PCA(n_components=d, whiten=True, random_state=42)
    Xp = pca.fit_transform(Xn)
    sc = StandardScaler()
    Xs = sc.fit_transform(Xp)

    clfs = {
        "svm": SVC(kernel="linear", C=1, probability=True, class_weight="balanced", random_state=42),
        "logreg": LogisticRegression(C=10, max_iter=3000, solver="lbfgs", class_weight="balanced", random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=min(5, Xtr.shape[0]-1), weights="distance", metric="cosine"),
    }
    for clf in clfs.values():
        clf.fit(Xs, ytr)

    return pca, sc, clfs


def predict_ensemble(Xva, pca, sc, clfs, weights=None):
    """Predict using ensemble soft voting."""
    if weights is None:
        weights = {"svm": 0.40, "logreg": 0.35, "knn": 0.25}
    Xn = Xva / (np.linalg.norm(Xva, axis=1, keepdims=True) + 1e-10)
    Xp = pca.transform(Xn)
    Xs = sc.transform(Xp)
    p = sum(clf.predict_proba(Xs) * weights[n] for n, clf in clfs.items())
    return np.argmax(p, axis=1), p


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Per-Intent CV Accuracy
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  TEST 1: Per-Intent 5-Fold CV Accuracy (PCA-200)")
print(f"{'='*70}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_true, all_pred, all_probs = [], [], []

for tr, va in cv.split(X, y):
    pca, sc, clfs = build_ensemble(X[tr], y[tr], n_pca=200)
    preds, probs = predict_ensemble(X[va], pca, sc, clfs)
    all_true.extend(y[va])
    all_pred.extend(preds)
    all_probs.extend(probs.max(axis=1))

all_true, all_pred = np.array(all_true), np.array(all_pred)
all_probs = np.array(all_probs)
overall_acc = (all_true == all_pred).mean()

print(f"\n  Overall CV Accuracy: {overall_acc*100:.2f}%")
print(f"\n  {'Intent':<28} {'Acc':>6} {'N':>4} {'Wrong':>6}  Confused With")
print(f"  {'-'*75}")

intent_results = {}
below_90 = []
for i, name in enumerate(labels):
    mask = all_true == i
    if mask.sum() == 0:
        continue
    correct = (all_pred[mask] == i).sum()
    total = mask.sum()
    acc = correct / total * 100
    intent_results[name] = acc
    
    # Find what it gets confused with
    wrong_mask = mask & (all_pred != all_true)
    confused_with = ""
    if wrong_mask.sum() > 0:
        wrong_preds = all_pred[wrong_mask]
        confused = Counter(wrong_preds).most_common(2)
        confused_with = ", ".join(f"{labels[c]}({n})" for c, n in confused)
    
    marker = "  *** BELOW 90%" if acc < 90 else ""
    print(f"  {name:<28} {acc:>5.1f}% {total:>3} {total-correct:>5}  {confused_with}{marker}")
    if acc < 90:
        below_90.append((name, acc))

print(f"\n  Intents below 90%: {len(below_90)}/{len(labels)}")
for name, acc in sorted(below_90, key=lambda x: x[1]):
    print(f"    {name}: {acc:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Confusion Matrix (top confused pairs)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  TEST 2: Top Confusion Pairs")
print(f"{'='*70}")

confusion = defaultdict(int)
for t, p in zip(all_true, all_pred):
    if t != p:
        confusion[(labels[t], labels[p])] += 1

sorted_conf = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
print(f"\n  {'Actual → Predicted':<50} {'Count':>6}")
print(f"  {'-'*58}")
for (actual, predicted), count in sorted_conf[:15]:
    print(f"  {actual} → {predicted:<25} {count:>5}")
total_errors = sum(confusion.values())
print(f"\n  Total errors: {total_errors}/{len(all_true)} ({total_errors/len(all_true)*100:.1f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Stability Test (10 different random seeds)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  TEST 3: Stability Across Random Seeds (10 seeds)")
print(f"{'='*70}")

seed_accs = []
for seed in range(10):
    cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed*7+3)
    fold_accs = []
    for tr, va in cv_s.split(X, y):
        pca, sc, clfs = build_ensemble(X[tr], y[tr], n_pca=200)
        preds, _ = predict_ensemble(X[va], pca, sc, clfs)
        fold_accs.append((preds == y[va]).mean())
    seed_acc = np.mean(fold_accs)
    seed_accs.append(seed_acc)
    print(f"  Seed {seed}: {seed_acc*100:.2f}%")

print(f"\n  Mean: {np.mean(seed_accs)*100:.2f}% ± {np.std(seed_accs)*100:.2f}%")
print(f"  Min:  {np.min(seed_accs)*100:.2f}%")
print(f"  Max:  {np.max(seed_accs)*100:.2f}%")
all_above_90 = all(a >= 0.90 for a in seed_accs)
print(f"  All seeds ≥90%: {'YES' if all_above_90 else 'NO'}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Leave-2-Out Stress Test (train on 18, test on 2 per intent)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  TEST 4: Leave-2-Out Stress Test (10 random trials)")
print(f"{'='*70}")
print(f"  Each trial: remove 2 random examples per intent, train on rest, test on removed")

l2o_accs = []
for trial in range(10):
    rng = np.random.RandomState(trial*13+7)
    test_idx, train_idx = [], []
    
    for c in range(len(labels)):
        c_idx = np.where(y == c)[0]
        if len(c_idx) < 4:
            train_idx.extend(c_idx)
            continue
        chosen = rng.choice(c_idx, size=2, replace=False)
        test_idx.extend(chosen)
        train_idx.extend([i for i in c_idx if i not in chosen])
    
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    pca, sc, clfs = build_ensemble(X[train_idx], y[train_idx], n_pca=200)
    preds, _ = predict_ensemble(X[test_idx], pca, sc, clfs)
    acc = (preds == y[test_idx]).mean()
    l2o_accs.append(acc)
    print(f"  Trial {trial}: {acc*100:.2f}% ({len(test_idx)} test samples)")

print(f"\n  Mean: {np.mean(l2o_accs)*100:.2f}% ± {np.std(l2o_accs)*100:.2f}%")
print(f"  Min:  {np.min(l2o_accs)*100:.2f}%")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Confidence Distribution (what % would go to LLM?)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  TEST 5: Confidence Distribution (from CV predictions)")
print(f"{'='*70}")

# Rebuild CV with probabilities
high_conf, med_conf, low_conf = 0, 0, 0
correct_high, correct_low = 0, 0
total_high, total_low = 0, 0

for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    pca, sc, clfs = build_ensemble(X[tr], y[tr], n_pca=200)
    preds, probs = predict_ensemble(X[va], pca, sc, clfs)
    
    for j, (t, p) in enumerate(zip(y[va], preds)):
        prob = probs[j].max()
        sorted_p = np.sort(probs[j])[::-1]
        margin = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else 1.0
        
        confident = prob >= 0.45 and margin >= 0.12
        
        if confident:
            high_conf += 1
            total_high += 1
            if t == p: correct_high += 1
        else:
            low_conf += 1
            total_low += 1
            if t == p: correct_low += 1

total = high_conf + low_conf
print(f"\n  Confident (would skip LLM): {high_conf}/{total} ({high_conf/total*100:.1f}%)")
print(f"  Ambiguous (would go to LLM): {low_conf}/{total} ({low_conf/total*100:.1f}%)")
if total_high > 0:
    print(f"\n  Confident predictions accuracy: {correct_high/total_high*100:.1f}% ({correct_high}/{total_high})")
if total_low > 0:
    print(f"  Ambiguous predictions accuracy: {correct_low/total_low*100:.1f}% ({correct_low}/{total_low})")
    print(f"  → These {low_conf} queries would be sent to Gemini Flash for arbitration")


# ═════════════════════════════════════════════════════════════════════════════
# VERDICT
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  VERDICT")
print(f"{'='*70}")
print(f"  CV Accuracy (5-fold):        {overall_acc*100:.2f}%")
print(f"  Stability (10 seeds):        {np.mean(seed_accs)*100:.2f}% ± {np.std(seed_accs)*100:.2f}%")
print(f"  Leave-2-Out stress:          {np.mean(l2o_accs)*100:.2f}% ± {np.std(l2o_accs)*100:.2f}%")
print(f"  All seeds ≥90%:              {'YES' if all_above_90 else 'NO'}")
print(f"  Intents below 90%:           {len(below_90)}/{len(labels)}")
print(f"  Queries needing LLM:         {low_conf}/{total} ({low_conf/total*100:.1f}%)")
if total_high > 0:
    print(f"  Primary classifier accuracy: {correct_high/total_high*100:.1f}% (confident queries only)")
print(f"{'='*70}")
