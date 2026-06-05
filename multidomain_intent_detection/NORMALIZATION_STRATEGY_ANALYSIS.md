# Normalization Strategy — ReACT Analysis & Recommendation

> **Question:** Should training embeddings include BOTH normalized AND
> raw (un-normalized) versions of each example to boost embedding accuracy?
> Or is the current "normalize everywhere" approach correct?

---

## 0. Premise Correction

The doubt was framed as:

| Stage | Claim | Reality |
|---|---|---|
| Training (augmented) — `training.py:145` | normalized ✅ | **normalized** ✅ |
| Training (base) — `training.py:88-90` | NOT normalized ❌ | **normalized** ✅ (line 90 + line 101 both call `normalize_query`) |

So the current pipeline is already **symmetric**: base, augmented, and inference
queries (`classifier.py:244`) all pass through `normalize_query()` before
embedding. The real question becomes:

> *Is symmetric normalization optimal, or should we ALSO add raw versions
> (a "dual-view" training set)?*

---

## 1. REASON — Theoretical Framing

Embedding-based intent classification operates on this principle:

> Train and inference inputs must live in the **same distribution** in the
> embedding space. Distance/similarity is only meaningful between vectors
> produced from inputs that were preprocessed identically.

There are three possible strategies:

| Strategy | Train | Inference | Comment |
|---|---|---|---|
| **A. Symmetric Raw** | raw | raw | Baseline; numeric IDs leak into vectors |
| **B. Symmetric Normalized** *(current)* | normalized | normalized | Removes ID noise; standard practice |
| **C. Dual-view (both)** | raw + normalized | normalized | More data, but asymmetric; risks drift |

---

## 2. ACT — What the Code Actually Does

```
load_embeddings()      → normalize_query(s)  ✅ (training.py:90, 101)
augment_embeddings()   → normalize_query(ex) ✅ (training.py:156)
classifier.classify()  → normalize_query()   ✅ (classifier.py:244)
evaluate()             → normalize_query()   ✅ (training.py:284)
```

**Verdict on current state:** Strategy **B** is implemented consistently.
This is the textbook-correct setup.

---

## 3. OBSERVE — What Normalization Actually Strips

From `normalizer.py`:

| Pattern | Example Input | After Normalization |
|---|---|---|
| `_CLAIM_NUM_PATTERN` (12–18 digits) | `claim 132435151040074` | `claim claim_id` |
| `_SEQ_PATTERN` / `_SEQ_NUM` | `sequence 001` / `seq 5` | *(removed)* |
| `_PA_NUM_PATTERN` | `PA JW012726LC` | `pa` |
| `_NDC_NUM_PATTERN` | `NDC 33342-0395-44` | `ndc` |

What survives: **the intent-bearing words** ("prescriber details for claim",
"check status of pa", "lookup ndc"). Exactly what we want the embedder to
encode.

---

## 4. THINK — Pros and Cons of Adding Raw Versions (Strategy C)

### Arguments FOR adding raw alongside normalized
1. **More training points per intent** → potentially better cluster density.
2. **Captures lexical signal of digit-rich phrasing** in case some users send
   queries with ID-laden text and inference path skips normalization.
3. **Defense in depth** — if `normalize_query` ever has a bug or misses an
   ID format, raw versions still have a chance to match.

### Arguments AGAINST (much stronger)

1. **Distribution mismatch.** Inference always normalizes
   (`classifier.py:244`). Adding raw vectors places half the training cloud
   in a region the inference query will *never* visit. Those vectors become
   dead weight, or worse, distort PCA components and centroid calculations.

2. **Numeric IDs are pure noise for classification.**
   `text-embedding-005` will allocate embedding capacity to encode
   `"132435151040074"` — a token sequence that has no generalizable signal
   for intent (it's a unique ID). This dilutes the intent signal in the
   averaged/PCA-projected representation.

3. **Class-specific ID memorization.** If `ClaimNum` examples all contain
   15-digit numbers and `PrescriberDetails` examples also contain 15-digit
   numbers, the embedder still encodes "this looks like an ID" — pulling
   distinct intents *closer* in space, hurting separability.

4. **PCA degradation.** PCA picks the directions of maximum variance.
   Numeric tokens add high-variance, low-information dimensions. Including
   raw versions means the search in `search_pca()` may pick a higher
   `n_pca` than truly needed, increasing overfitting risk and inference
   latency.

5. **Doubles embedding cost & cache size.** ~2× Vertex API calls during
   first-time generation; ~2× JSON cache footprint; ~2× memory for
   `IntentPipeline`.

6. **Breaks the contract documented in `augment_embeddings`:**
   > *"Normalizes examples (strips claim numbers) before embedding so they
   > match the normalized test query space."*

   Adding raw versions silently violates this invariant.

7. **Confusion-pair gating relies on clean clusters.**
   `CONFUSION_PRONE_INTENTS` thresholds (lines 268–273) are tuned against
   the current cluster geometry. Injecting noisy raw vectors will shift
   confidence/margin distributions and break those thresholds.

---

## 5. SELF-CRITICISM — Steel-manning the "Add Both" Position

Let me push back on my own conclusion:

- *"But text-embedding-005 is huge — surely it can ignore the noise?"*
  Empirically, yes, it largely does. But the downstream pipeline is
  PCA + SVM/LogReg/kNN, which is **not** noise-robust. It explicitly
  preserves variance, including the unhelpful kind.

- *"What if a real user sends raw text with no normalization at the
  classifier?"*
  The classifier already normalizes (`classifier.py:244`). The only way
  this matters is if someone bypasses the public API. Out of scope.

- *"What about the case where normalization removes too much?
  E.g. `'sequence 001'` becomes empty residue."*
  This is a real concern — but the fix is **better normalization**
  (preserving an `<SEQ>` token), not duplicating training data.

- *"Could a small amount of raw data act as a regularizer?"*
  Possibly, but the disciplined way to regularize is via PCA dim choice,
  ensemble weighting, and label smoothing — not by polluting the input
  distribution.

The steel-man arguments survive only as edge cases that have superior,
targeted fixes. The "add both" approach addresses none of them cleanly.

---

## 6. CONCLUSION

**Keep the current symmetric-normalized approach. Do NOT add raw versions.**

The current code is correct in principle and consistent in implementation.
Adding un-normalized duplicates would:

- Violate train/inference distribution symmetry,
- Inject ID-shaped noise into the PCA/ensemble,
- Inflate cost and cache size,
- Destabilize confusion-pair gating thresholds.

---

## 7. RECOMMENDED IMPROVEMENTS (Higher-Leverage than Dual-View)

If the goal is to **boost accuracy**, the following changes will pay off
much more than adding raw versions:

### 7.1 Replace ID strips with explicit placeholder tokens (semantic anchors)

Currently `_SEQ_PATTERN` and `_SEQ_NUM` substitute with `''` (empty string).
This loses the *signal* that a sequence number was mentioned. Replace with
a token instead — same idea as `claim_id` does for claim numbers:

```python
# normalizer.py
t = _SEQ_PATTERN.sub('seq_id', t)
t = _SEQ_NUM.sub('seq_id', t)
```

This way `"prescriber details for claim 12345 sequence 001"` becomes
`"prescriber details for claim claim_id seq_id"` — the model learns that
*the presence of* a sequence reference is intent-relevant, without
memorizing the digits.

### 7.2 Add inverse-augmentation (drop-the-ID variants)

For each base example that contains an ID, also produce a version with
the ID textually removed (not just regex-stripped). This generates
realistic short-form user queries without inflating embedding noise:

- Original: `"Show details for claim 132435151040074"`
- Variant : `"Show claim details"` *(added once, normalized)*

This is augmentation in the *intent* dimension, not the *noise* dimension.

### 7.3 Verify normalization coverage against `intents_mapping.py`

Run a one-off audit script: for every example in
`embeddingVars.CVS_INTENT_EXAMPLES`, print `(original, normalized)` pairs
and grep for any leftover digit runs. If any slip through, extend the
regex set rather than duplicating data.

### 7.4 Consider a small LLM-paraphrase augmentation pass

Use Gemini Flash (already used by `llm_entity_extractor`) to generate 2–3
paraphrases per base example. These add **lexical diversity** while
preserving intent — exactly what the embedder benefits from. Cache them
in `augmented_examples.py` and let the existing normalization pipeline
process them.

### 7.5 Centralize the normalization contract

Add a unit test that asserts: `normalize_query(x) == normalize_query(normalize_query(x))`
(idempotence) and that all training/inference call sites flow through
the same function. Today this is enforced by convention only.

---

## 8. TL;DR

| Question | Answer |
|---|---|
| Is current code correct? | **Yes** — base & augmented are both normalized; inference is too. |
| Add raw versions alongside normalized? | **No** — breaks distribution symmetry, hurts PCA, no real upside. |
| What to do for accuracy gains? | Improve normalization (placeholder tokens), paraphrase-augment, audit coverage, add idempotence test. |
