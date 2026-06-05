# Normalization Analysis: PCA → Ensemble Pipeline
**ReAct Loop with Self-Criticism**  
Date: 2026-05-15  
File scope: `pipeline.py`, `training.py`, `classifier.py`

---

## The Core Question

Two specific questions to answer:
1. Are we L2/mathematical normalizing embeddings **after PCA** and **before training the ensemble**?
2. Are we normalizing the **input vector** before feeding it into the ensemble at inference time?

---

## Actual Pipeline (Reading the Code)

The full normalization chain lives in two methods in `pipeline.py`:

### Training path — `fit()` (lines 128–134)
```python
# L2-normalize → PCA → scale
X_n = X_raw / (np.linalg.norm(X_raw, axis=1, keepdims=True) + 1e-10)   # Step 1
self.pca = PCA(n_components=d, whiten=True, random_state=42)             # Step 2
X_p = self.pca.fit_transform(X_n)                                         # Step 2 applied
self.scaler = StandardScaler()                                            # Step 3
X_s = self.scaler.fit_transform(X_p)                                      # Step 3 applied
# classifiers are trained on X_s
```

### Inference path — `_transform()` (lines 356–358)
```python
def _transform(self, X: np.ndarray) -> np.ndarray:
    X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)        # Step 1
    return self.scaler.transform(self.pca.transform(X_n))                 # Steps 2+3
```

**Direct answers before the analysis:**

| Question | Answer |
|---|---|
| Normalized after PCA and before ensemble training? | **YES** — `StandardScaler` is fitted on PCA output; classifiers see `X_s` |
| Input normalized before feeding to ensemble at inference? | **YES** — `_transform()` mirrors the training chain exactly |

---

## ReAct Analysis

ReAct = **Re**asoning + **Act**ing in a loop with an Observation after each step.  
Self-criticism is applied at the end of each round before moving to the next.

---

### Round 1 — Reason about Step 1 (L2-norm before PCA)

**Reasoning:**  
Raw Vertex AI `text-embedding-005` vectors are 768-dimensional floats. Their magnitude is not guaranteed to be uniform across queries. A query like "show me claims" might have a different L2 norm than "What are the prior authorization requirements for Humira 40mg prefilled pen injection for member 00987654321 under plan XYZ?". PCA is a variance-maximizing decomposition — if some vectors are "louder" (larger norm) than others, those samples will dominate the principal components. That would make the PCA representation more sensitive to sentence length and embedding amplitude than to semantic content.

**Action (code read):**  
`X_n = X_raw / (np.linalg.norm(X_raw, axis=1, keepdims=True) + 1e-10)` — this is row-wise L2 normalization. All 768-d vectors are projected onto the unit sphere before PCA.

**Observation:**  
L2 normalization destroys magnitude information but preserves direction (semantics). For intent classification where we care about *what* the query means, not how much it says, this is correct. PCA on the unit sphere will find directions of maximum semantic variation.

**Self-Criticism:**  
Wait — Vertex AI text embeddings are sometimes already near-unit-norm by design. If the input vectors are already unit-norm, the division is a no-op and the `+ 1e-10` stabilizer still protects against the zero-vector edge case. However, this assumption must not be baked in — the L2-normalize step is the safe choice regardless. No issue here.

**Verdict:** ✅ Correct and intentional.

---

### Round 2 — Reason about Step 2 (PCA with `whiten=True`)

**Reasoning:**  
Standard `PCA(whiten=False)` rotates the data to align axes with directions of maximum variance and reduces dimensions. The output coordinates have *unequal* variances: the first PC has high variance, the last has low variance. Most distance-based classifiers (SVM-RBF, kNN) are sensitive to feature scale — a PC with variance 100 will dominate one with variance 0.1.

`PCA(whiten=True)` adds one more step: it divides each PC coordinate by `sqrt(eigenvalue)`. Mathematically:

```
X_whitened[i, j] = X_pca[i, j] / sqrt(λ_j)
```

where `λ_j` is the j-th eigenvalue. This makes each PC have **unit variance across the training set**.

**Action (implications):**  
After PCA whitening on training data:
- Mean per PC = 0 (PCA always centers)
- Variance per PC = 1 (whitening enforces this)
- Training data is on a "normalized" manifold in PC space

**Observation:**  
PCA with whitening is mathematically equivalent to doing PCA then applying a per-feature StandardScaler fitted on training data (because the per-feature std of whitened output equals 1 by construction, and mean equals 0).

**Self-Criticism:**  
If `PCA(whiten=True)` already produces zero-mean unit-variance per PC, what does the subsequent `StandardScaler` in Step 3 actually do? This is the critical contradiction I need to resolve in Round 3.

**Verdict:** ✅ Correct choice — whitening benefits SVM-RBF and LogReg equally.

---

### Round 3 — The Redundancy Problem: PCA whiten + StandardScaler

**Reasoning:**  
`StandardScaler.fit(X_p)` computes the mean and std of each feature across the training PCA output.  
After `PCA(whiten=True)`, the training data has:
- `mean(X_p[:, j]) ≈ 0` for all j (PCA centers data)
- `std(X_p[:, j]) ≈ 1` for all j (whitening enforces unit variance)

So `StandardScaler` fitted on `X_p` learns parameters `mean_ ≈ 0` and `scale_ ≈ 1` for every feature.

For **training data**: `StandardScaler.transform(X_p) ≈ (X_p - 0) / 1 ≈ X_p`. Essentially a no-op.

For **test data at inference**: `StandardScaler.transform(X_p_test) = (X_p_test - mean_train) / std_train`. The stored `mean_train ≈ 0` and `std_train ≈ 1`, so again approximately `X_p_test`. Still near no-op.

**The only non-trivial benefit** of StandardScaler here is:
- It stores the exact training statistics, providing a deterministic and numerically stable transformation that matches what the classifiers saw during fit, even if floating-point precision causes tiny differences.
- If a future code change removes PCA whitening, StandardScaler provides a safety net.

**Action (verify the math):**  
Let `X_raw` have shape `(N, 768)`.  
After L2-norm: all rows have `||X_n[i]|| = 1`.  
After PCA whiten with `d` components: `X_p` has shape `(N, d)`, with training stats `mean ≈ 0, std ≈ 1` per column.  
After StandardScaler: `X_s = (X_p - 0) / 1 ≈ X_p`. Shape unchanged.  
Classifiers are trained on `X_s ≈ X_p`.

**Observation:**  
The combination is *redundant* for normalization purpose on training data. It is *weakly useful* for ensuring exact train/test consistency and numerical stability. It is *not harmful*.

**Self-Criticism:**  
I initially called this "semi-redundant." Let me be more precise: it's redundant for variance normalization, but it's NOT redundant as a stateful transform that ensures test vectors go through identical statistics as training. Removing StandardScaler would require ensuring PCA whitening is always active and consistent. Keeping it is the defensive correct choice, even if mathematically redundant.

**Verdict:** ⚠️ Redundant but harmless. The real issue lies elsewhere.

---

### Round 4 — The kNN Cosine Problem (Most Important Finding)

**Reasoning:**  
The kNN classifier is configured with `metric="cosine"` on line 162 of `pipeline.py`:
```python
self.clfs["knn"] = KNeighborsClassifier(
    n_neighbors=min(self.knn_k, X_raw.shape[0] - 1),
    weights="distance",
    metric="cosine",
).fit(X_s, y)
```

`X_s` is the output of `StandardScaler.transform(PCA.transform(L2_norm(X_raw)))`.

**Cosine distance** is defined as:
```
cosine_distance(a, b) = 1 - (a · b) / (||a|| * ||b||)
```

For cosine to measure pure *angle* (semantic direction), the vectors must lie on the unit sphere (`||a|| = 1`). If they have different norms, cosine distance measures a mix of direction AND relative magnitudes.

**Are vectors in X_s unit-norm?**  
After PCA whitening, each PC coordinate has std=1 across training samples. But individual sample vectors `X_s[i]` do NOT have unit L2 norm. Their norms are approximately `sqrt(d)` on average (if each feature is i.i.d. N(0,1)), but with significant variance. Some queries will have vectors with norm 1.5, others with norm 8.2.

**Consequence:**  
Cosine distance in `X_s` space is **not a pure angle measure**. A query with a large-norm vector will be pulled toward neighbors with large-norm vectors regardless of semantic similarity. This is the opposite of what cosine metric is supposed to do.

**What should happen:**  
For kNN with cosine metric, vectors should be L2-normalized after the final transform:
```python
# After StandardScaler
X_for_knn = X_s / (np.linalg.norm(X_s, axis=1, keepdims=True) + 1e-10)
```

Or alternatively, switch kNN to `metric="euclidean"`, because PCA-whitened space is already isotropic (all directions have equal variance), so Euclidean distance already captures semantic similarity without the cosine unit-norm requirement.

**Self-Criticism:**  
Am I sure the vectors are not unit-norm after whitening+StandardScaler? Let me think again.

If `X_p = PCA_whiten(L2_norm(X_raw))`, and `X_s = StandardScaler(X_p)`:
- Each feature `X_s[:, j]` has `std ≈ 1` and `mean ≈ 0`
- Each sample `X_s[i, :]` has `||X_s[i]||^2 = sum(X_s[i, j]^2)`
- If features were truly i.i.d. N(0,1), then `||X_s[i]||^2 ~ chi-squared(d)`, so `||X_s[i]|| ≈ sqrt(d) ± sqrt(2d)/2`
- For d=50 (typical PCA dim): norm ≈ 7.07 ± 5.0 → NOT unit norm, high variance

So yes: after PCA whiten + StandardScaler, vectors have norms around `sqrt(d)` with significant spread. Cosine is NOT a pure angle measure here.

**Counterpoint:** In practice, sklearn's `KNeighborsClassifier(metric='cosine')` normalizes internally before computing distances. Let me verify...

Actually, sklearn's `KNeighborsClassifier` does NOT automatically L2-normalize vectors when `metric='cosine'` — it computes `1 - (a · b) / (||a|| * ||b||)` directly. The denominator handles the magnitude difference, so cosine distance IS mathematically correct as an angle measure. The magnitudes cancel in the denominator.

**Revised Self-Criticism:**  
I was wrong in my initial analysis. Cosine distance formula `1 - (a · b) / (||a|| * ||b||)` DOES divide by magnitudes, making it scale-invariant. So `||X_s[i]||` varying does not affect cosine distance. The cosine metric computes the correct angle regardless of vector magnitude.

However, there is STILL a concern: if you use `weights="distance"` with cosine metric in kNN, the *weights* assigned to neighbors are `1 / cosine_distance`. These are in units of angle (radians). This is mathematically sound. No issue.

**Revised Verdict:** The kNN cosine usage is mathematically correct because cosine distance is scale-invariant by definition. My initial concern was wrong, but the reasoning process was valuable to check.

**Final Verdict on Round 4:** ✅ kNN cosine metric is correct — cosine distance is scale-invariant.

---

### Round 5 — Inference Path Consistency Check

**Reasoning:**  
Training and inference must apply the exact same transformation chain. A common bug is applying L2-norm in training but forgetting it at inference, or fitting the scaler on different data.

**Action (read `_transform`):**
```python
def _transform(self, X: np.ndarray) -> np.ndarray:
    X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)   # same L2-norm
    return self.scaler.transform(self.pca.transform(X_n))            # same stored PCA+scaler
```

In `predict_single()` (line 390):
```python
X_f = self._transform(vec.reshape(1, -1))
```

In `classifier.py` (line 249–252):
```python
vec = np.array(embedder.embed(normalized))
pred = self._pipeline.predict_single(vec)
```

**Observation:**  
- Training: `L2_norm → PCA.fit_transform → StandardScaler.fit_transform` (stores PCA and scaler)
- Inference: `L2_norm → PCA.transform → StandardScaler.transform` (uses stored PCA and scaler)

These are identical in the applied operations. The stored PCA and StandardScaler use training data statistics only.

**Self-Criticism:**  
Is the cross-validation path also consistent? In `cross_validate()` (lines 461–510):
```python
Xn = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-10)   # ✓ L2-norm
fold_pipe.pca = PCA(n_components=d, whiten=True, random_state=42)
Xp = fold_pipe.pca.fit_transform(Xn)                                 # ✓ PCA on train fold
fold_pipe.scaler = StandardScaler()
Xs = fold_pipe.scaler.fit_transform(Xp)                              # ✓ Scaler on train fold
# classifiers trained on Xs

# Validation:
preds = np.argmax(fold_pipe.predict_proba(Xva), axis=1)             # ✓ uses _transform()
```

Correct. The CV path fits PCA and StandardScaler on training fold, and inference on validation fold uses `_transform()` which applies `.transform()` (not `.fit_transform()`). No leakage.

**Verdict:** ✅ Perfectly consistent across training, cross-validation, and inference.

---

### Round 6 — PCA Whitening Interaction with SVM-RBF

**Reasoning:**  
SVM-RBF uses Gaussian kernel: `K(a, b) = exp(-gamma * ||a - b||^2)`.  
`gamma='scale'` sets `gamma = 1 / (n_features * X.var())`.  

After PCA whiten + StandardScaler: each feature has `var ≈ 1`, total variance ≈ `n_features`.  
So `gamma_scale = 1 / (n_features * n_features) = 1 / n_features^2`.

**Self-Criticism:**  
Wait — `sklearn.svm.SVC(gamma='scale')` computes var from the **training data passed to `.fit()`**, which is `X_s`. Since each feature of `X_s` has var≈1, the total variance across all features is ≈ `n_features`. So `gamma = 1 / (n_features * n_features_variance_per_feature)`. Actually sklearn's 'scale' computes `1 / (n_features * X.var())` where `X.var()` is the global variance across all elements (not per-feature). Let me think...

For `X_s` with each of `d` features having var=1, the global variance is:
`X_s.var() = (1/N) * sum_all_elements((x - mean)^2) ≈ 1`

So `gamma_scale = 1 / (d * 1) = 1/d`.

This is a well-calibrated gamma for whitened features. SVM-RBF is operating in a well-conditioned space.

**Verdict:** ✅ SVM-RBF benefits correctly from the whitened feature space.

---

### Round 7 — What Would Happen Without Post-PCA Normalization?

This round answers the counterfactual: what if we removed StandardScaler?

**Scenario A: Remove StandardScaler, keep `PCA(whiten=True)`**
- Training features: zero mean, unit variance per PC (whitening guarantees this)
- Inference features: same, because whiten applies stored eigenvalues
- **Impact:** Minimal — StandardScaler is near-identity after whitening. Classifiers see almost identical features.

**Scenario B: Remove StandardScaler, use `PCA(whiten=False)`**
- Training features: zero mean, but UNEQUAL variances per PC (PC1 >> PC2 >> ... >> PCd)
- SVM-RBF kernel: dominated by high-variance PCs
- LogReg: regularization treats all features equally despite unequal importance
- kNN cosine: correct (scale-invariant), but kNN euclidean would be wrong
- **Impact:** Significant degradation for SVM-RBF and LogReg. kNN cosine unaffected.

**Scenario C: Remove L2-norm before PCA (keep whiten=True + StandardScaler)**
- PCA is now sensitive to raw embedding magnitude
- Longer/more emphatic queries dominate the PCA directions
- Semantic signal may be diluted by magnitude noise
- **Impact:** Moderate to significant degradation depending on magnitude variance in the corpus.

**Verdict:** The current chain (L2-norm → PCA whiten → StandardScaler) is the **most robust combination** for a mixed ensemble of SVM-RBF, LogReg, kNN(cosine), and ExtraTrees.

---

## Summary Table

| Step | What Happens | Necessary? | Benefit |
|---|---|---|---|
| L2-norm (before PCA) | All 768-d vectors → unit sphere | YES | Prevents magnitude-dominated PCA; focuses on semantic direction |
| PCA(whiten=True) | Rotates + compresses to d dims; divides by √eigenvalue | YES | Reduces dimensionality; equalizes PC variances for SVM/LogReg |
| StandardScaler (after PCA) | Subtracts training mean, divides by training std | PARTIALLY | ~No-op after whitening; provides stored statistics for train/test consistency |
| Ensemble training on X_s | SVM, LogReg, kNN, ExtraTrees fit on StandardScaler output | YES | Classifiers see same normalized feature space |
| _transform() at inference | Applies stored L2-norm + PCA.transform + StandardScaler.transform | YES | Exactly mirrors training transformation chain |

---

## Issues Found

### Issue 1 (INFORMATIONAL): StandardScaler redundancy after PCA(whiten=True)
- **Severity:** Low — not harmful, barely useful
- **What it is:** PCA whitening already produces zero-mean unit-variance features. StandardScaler on top is nearly a mathematical identity.
- **Should we fix it?** No. It acts as insurance: if `whiten=True` is ever set to `False` by mistake, StandardScaler saves correctness. Defense-in-depth is appropriate.
- **Code:** `pipeline.py:133-134`

### Issue 2 (CONFIRMED CORRECT): kNN cosine metric on non-unit-norm vectors
- **Severity:** Non-issue — initially appeared concerning, confirmed correct
- **What it is:** Cosine distance is mathematically scale-invariant (`(a·b)/(||a||·||b||)` cancels magnitudes). sklearn's cosine metric does this correctly.
- **Should we fix it?** No.
- **Code:** `pipeline.py:161-165`

### Issue 3 (ACTIONABLE): No L2-norm after StandardScaler for kNN
- **Severity:** Low — kNN cosine is correct without it, but adding it would make distances *slightly* more numerically stable by avoiding the magnitude division in the denominator.
- **Recommendation (optional):** Consider L2-normalizing after StandardScaler before kNN fitting. Not required.
- **Code would be:** `X_for_knn = X_s / (np.linalg.norm(X_s, axis=1, keepdims=True) + 1e-10)`

---

## Should We Add Normalization Anywhere?

| Location | Add L2-norm? | Rationale |
|---|---|---|
| After PCA, before ensemble | Already done via StandardScaler (scale equalization) | No change needed |
| After StandardScaler, before all classifiers | Not needed for SVM/LogReg/ET; kNN cosine is scale-invariant | No change needed |
| At inference input (before _transform) | Already done in `_transform()` | No change needed |
| Between ensemble and temperature scaling | Temperature acts on probabilities [0,1] — normalization irrelevant | No |

**Conclusion: The current normalization chain is correct and complete.**  
Both questions in the original query are confirmed YES. No normalization steps are missing. The one "redundancy" (StandardScaler after PCA whiten) is a useful safety net, not a bug.

---

## ReAct Self-Criticism Final Pass

Over 7 rounds I caught one major self-correction: in Round 4, I initially claimed kNN cosine was problematic because vectors after StandardScaler are not unit-norm. After applying the math, cosine distance is defined as `(a·b)/(||a||·||b||)` — it IS scale-invariant by construction. My initial instinct was wrong, and the self-criticism loop caught it before producing a bad recommendation.

This is the value of the ReAct loop: **acting on the first instinct without checking the math would have produced a wrong recommendation to add a normalization step that provides zero benefit.**

---

## Code Locations (Quick Reference)

| Thing | File | Lines |
|---|---|---|
| L2-norm before PCA (training) | `pipeline.py` | 129 |
| PCA with whitening (training) | `pipeline.py` | 131–132 |
| StandardScaler (training) | `pipeline.py` | 133–134 |
| Full inference transform | `pipeline.py` | 356–358 |
| kNN cosine metric | `pipeline.py` | 161–165 |
| CV fold normalization | `pipeline.py` | 468–473 |
| Inference embedding → predict | `classifier.py` | 249–252 |
