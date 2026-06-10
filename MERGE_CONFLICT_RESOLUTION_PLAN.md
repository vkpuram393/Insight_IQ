# Merge Conflict Resolution Plan
## `poc/intent-detection` → `release/path14`
**Authority: `poc/intent-detection` wins all conflicts**

---

## 0. Critical Pre-Merge Finding: No Common Ancestor

`git merge-base poc/intent-detection origin/release/path14` returns no result — these two
branches have **entirely disjoint git histories**. Git cannot perform a standard 3-way merge.

**Consequence**: every file that differs between the two trees will be treated as a
conflict by git. The only safe strategy is:

```bash
git checkout -b release/path14 --track origin/release/path14
git merge --allow-unrelated-histories -X theirs poc/intent-detection
```

`-X theirs` resolves every conflict by taking the POC side, which is exactly the
priority rule specified. The sections below document *what that means file-by-file*
so nothing is silently lost.

---

## 1. Scope Summary

| Category | Count |
|---|---|
| Files added only in POC (new code) | ~128 |
| Files deleted in POC (removed from release/path14) | ~8 |
| Files modified in both branches (conflict candidates) | **40** |
| Files with only whitespace / mode changes | ~6 |

---

## 2. File-by-File Classification

### Legend
- ✅ **Non-conflicting** — safe to include, additive only
- ⚠️ **Potential conflict** — diverged in non-overlapping areas, needs care
- ❌ **Direct conflict** — mutually exclusive logic; POC version wins

---

### 2.1 Core Graph (`langgraph_agent.py`) — ❌ DIRECT CONFLICT

**POC adds:**
- Import of `Claims_search_api.claims_search_node_v2` and `Claims_search_api.intent_router`
- `call_claims_search` node registration
- `_route_after_build_context()` conditional edge: dispatches to either
  `call_claims_tool` (single claim) or `call_claims_search` (member history)
- `call_claims_search` → `response_safety_pii_precheck` edge

**release/path14 adds (superseded):**
- `_seed_redis_from_history()` — pre-populates Redis from MongoDB when chatbot reopens
- `user_session` parameter threading through `run_graph()` and `run_graph_stream()`
- `user_session` and `response_id` in the streaming `complete` event payload

**Resolution**: **POC wins entirely.**
`_seed_redis_from_history` is tightly coupled to the `user_session`/persistent-history
system that POC deliberately removed. Including it without its supporting infrastructure
would cause silent failures. The removal of `user_session` and `response_id` from the
streaming payload is intentional — these are tracked via `state.schema` fields elsewhere.

**Risk check**: The new `_route_after_build_context` router calls
`is_claims_search_query(state)` from `Claims_search_api.intent_router`. That function
must be present and correct, or the entire graph build fails at startup. Confirm
`Claims_search_api/intent_router.py` is included in the merge (it is — it's an added
file in the POC, not modified, so it comes in cleanly).

---

### 2.2 State Schema (`state/schema.py`) — ❌ DIRECT CONFLICT

**POC adds to `AgentState`:**
- `ensemble_intent: Optional[str]`
- `ensemble_confidence: Optional[float]`
- `llm_fallback_intent: Optional[str]`
- `llm_fallback_confidence: Optional[float]`
- Updated comment for `confidence`: "always the ensemble post-calibration score"

**POC removes:**
- `user_session: Optional[str]` — stable per-login session ID
- `user_session` parameter from `create_initial_state()`

**release/path14 adds (superseded):**
- `user_session` field and `session_uuid` plumbing

**Resolution**: **POC wins entirely.**
The four new fields are critical — they are written by the multidomain classifier,
read by `nodes/confidence.py` for the LLM-fallback early-exit decision, and surfaced
in the API response via `IntentResolutionTrail`. Losing any one of them breaks the
intent routing transparency.

`user_session` removal is safe: the persistent-history feature it supports
(`api/history_routes.py`, `_seed_redis_from_history`) is also removed in POC.

---

### 2.3 Intent Agent (`agents/extended_intent_agent_node.py`) — ❌ DIRECT CONFLICT

**POC adds (all critical):**
- `_looks_like_claim_history_query()` — regex safety net using `_CLAIMS_SEARCH_PATTERNS`
- `_build_intent_classification_metadata()` — builds `intent_classification_metadata`
  block including top-N intents and LLM-fallback thinking fields
- `domain = intent_result.get('domain')` extraction
- Multidomain entity merging: maps `claim_number`/`claimId` etc. from
  `intent_result['entities_from_query']` into the standard `entities` dict
- Claim-history domain normalization block (the large `is_claim_history` section):
  overrides `api_endpoint`, `required_entities_list`, `requires_llm`, and boosts
  `confidence` to 0.85 when domain is `claim_history_search`

**release/path14 version:** pre-domain-normalization code — no awareness of multidomain
classifier output.

**Resolution**: **POC wins entirely.**
The normalization block is the core of the routing correctness guarantee. Without it,
queries like "show all rejected claims" would hit the single-claim endpoint and ask for
a 3-digit sequence the member-history API never needs.

---

### 2.4 Confidence Router (`nodes/confidence.py`) — ❌ DIRECT CONFLICT

**POC adds:**
- LLM-fallback early-exit: if `llm_fallback_confidence` is set AND below the
  threshold AND ensemble confidence is also below threshold AND domain is NOT
  `claim_history_search`, route directly to `clarification` (skips a redundant
  second LLM judge call)
- Vague-query CHS safety net: deeply uncertain CHS queries (below 0.4 with no
  entities) also go to clarification

**release/path14:** older routing logic without these two guards.

**Resolution**: **POC wins entirely.**
The `claim_history_search` exemption in the early-exit is particularly important —
it was added specifically because the previous behavior incorrectly short-circuited
CHS queries to clarification. Removing this guard reintroduces that regression.

---

### 2.5 Clarification Node (`nodes/clarification.py`) — ❌ DIRECT CONFLICT

**POC adds:**
- CHS safety net: removes `sequence` from `missing_slots` for `claim_history_search`
  domain
- Vague-query detection: uses `original_confidence` (distinguishing between
  `intent_reclassified=True`/`False` paths) to set `reason = "ambiguous_intent"`
  and clears `missing_slots` so the response agent doesn't jump to asking for
  claim identifiers before understanding what the user needs
- `intent_reclassified` and `original_confidence` diagnostic fields in context

**Resolution**: **POC wins entirely.**
The `sequence` removal is a correctness fix: asking users for a sequence number they
will never need is a UX bug. The vague-query path is new capability that must not be
regressed.

---

### 2.6 Response Agent (`agents/response_agent.py`) — ❌ DIRECT CONFLICT

**POC adds:**
- Import of `build_claim_history_prompt` from `Claims_search_api.llm_query_responder`
- `RecommendationChip` and `LLMStructuredResponse` Pydantic models for Gemini native
  JSON-mode output
- Reason-based clarification guide (`REASON = "missing_entity"` / `"low_confidence"` /
  `"ambiguous_intent"` sections)

**POC removes:**
- `"ABSOLUTE RULE — INSTRUCTION CONFIDENTIALITY"` block from the clarification system
  prompt

**release/path14:** no claim-history prompt support; has the confidentiality block.

**Resolution**: **POC wins entirely.**
The removal of the confidentiality block from the *clarification* prompt (not the main
response prompt) is intentional — it adds noise to the follow-up question generation
without adding security. The main response prompt's confidentiality instructions remain
in place elsewhere.

---

### 2.7 API Routes (`api/routes.py`) — ❌ DIRECT CONFLICT

**POC adds:**
- `IntentStageResult` and `IntentResolutionTrail` Pydantic models
- `x_api_key` and `x_clientrefid` header capture into `user_info` (required by
  `Claims_search_api` downstream)
- `intent_resolution_trail` field in `ChatResponse`
- `_ensemble_stage`, `_llm_fallback_stage`, `_clarification_stage` construction logic

**POC removes:**
- `session_uuid: Optional[str]` from `ChatRequest`
- `session_uuid` echo-back from `ChatResponse`
- `user_session` parameter in `run_graph()` calls

**release/path14 adds (superseded):**
- `session_uuid` / `user_session` plumbing

**Resolution**: **POC wins entirely.**
`x_api_key` and `x_clientrefid` capture is required for the Claims_search_api to
authenticate against the internal CVS gateway. Without these headers, the member-history
pipeline will fail with 401/403 errors on the Step-2 search call.

---

### 2.8 Config Settings (`config/config.py`) — ⚠️ POTENTIAL CONFLICT with SECURITY FLAG

**POC adds (legitimate):**
- `use_multidomain_classifier: bool = True`
- `LLM_FALLBACK_CONFIDENCE_THRESHOLD: float = 0.85`
- `TRAINING_LLM_FALLBACK_CSV_PATH: str = ...`
- `is_normalised: bool = True`
- `claim_list_api`, `claims_internal_base_url`, `claims_history_search_url` endpoints
- `claims_history_search_x_api_key`

**POC has hardcoded credentials — MUST BE FIXED BEFORE MERGE:**
```python
# THESE MUST BE REMOVED / REVERTED TO EMPTY DEFAULTS:
redis_password: str = "124d8306-f5b6-40f0-82e7-9f7f464d0ca6"
mongodb_connection_string: str = "mongodb+srv://myclaims_dev:1sw2tSZ89tjpn1sm@..."
mongodb_host: str = "mdb-use4-myclaims-dev01-pl-0.knpouh.mongodb.net"
mongodb_database_name: str = "myclaims-DEV"
claims_history_search_x_api_key: str = "fbbae75e-cd91-47a5-bb65-b68f525a66e3"
redis_host: str = "10.236.128.156"
project_id: str = "pbm-poc-coderev-genai-poc"  # reverts prod project ID
```
These are dev/POC test credentials committed to code. Before merging into
`release/path14` they must be reverted to the empty-string defaults that Vault/env vars
will fill at runtime. **This is the single highest-priority pre-merge fix.**

**Resolution**: POC logical additions (the new config flags) win. Credentials must be
scrubbed back to empty defaults before merge.

---

### 2.9 Domain Config (`config/domain_config.json`) — ❌ DIRECT CONFLICT

**POC changes:**
- `confidence_threshold`: `0.6` → `0.85`
- Adds `llm_fallback_confidence_threshold: 0.85`
- Adds `vague_query_confidence_threshold: 0.4`

**release/path14:** threshold is 0.6.

**Resolution**: **POC wins entirely.**
The 0.85 threshold is calibrated for the multidomain classifier's output range.
With the embedding-only classifier (release/path14), 0.6 was appropriate; with the
PCA+Ensemble stack, scores are distributed differently and 0.85 is the correct gate.
Reverting to 0.6 would route the majority of queries through `llm_judge` unnecessarily.

---

### 2.10 API Routing Config (`config/api_routing_config.py`) — ✅ NON-CONFLICTING

**POC adds:**
- `CLAIM_HISTORY_SEARCH_ENDPOINT` constant
- `_CLAIM_HISTORY_SEARCH_BASE` template
- `_CLAIM_HISTORY_INTENTS` list + loop that registers 18 intents
  (NDC, Manufacturer, Generic, Brand, etc.) in `INTENT_API_ROUTING`
- `get_domain_for_intent()` helper
- `CLAIM_HISTORY_SEARCH_ENDPOINT` export

**release/path14:** this section does not exist. No overlap.

**Resolution**: POC additions come in cleanly. No conflict.

---

### 2.11 Main Entry (`main.py`) — ❌ DIRECT CONFLICT

**POC changes:**
- Startup: adds multidomain classifier eager-load block with GCS path logging,
  `FileNotFoundError` fallback to embedding classifier, fatal `raise` on other errors
- Removes `history_routes` registration (and its `try/except` import block)
- Changes default port from `8001` → `5000`

**release/path14:** has `history_routes` import+registration; port is 8001.

**Resolution**: **POC wins entirely.**
Port 5000 is the project standard per CLAUDE.md. The removal of `history_routes`
is intentional — that API depends on `user_session` infrastructure that POC removes.
The multidomain startup block is required for the POC's primary feature.

---

### 2.12 Dockerfile — ❌ DIRECT CONFLICT

**POC uses:** `FROM langchain/langgraph-api:3.11` as base; installs via `uv`; adds
`v3_pipeline.pkl` existence check at build time.

**release/path14 uses:** Multi-stage `python:3.12-slim` builder → `distroless/python-3.12`
runtime (the CVS JFrog distroless image). More secure, smaller attack surface.

**Resolution**: **POC wins** (required for the multidomain classifier's runtime
dependencies). However, **note the security regression**: the distroless approach in
release/path14 is a deliberate security hardening. Post-merge, the security team should
evaluate whether the langchain base image meets CVS security standards or whether the
distroless approach should be restored.

---

### 2.13 `requirements.txt` — ❌ DIRECT CONFLICT

**POC adds:**
- `numpy>=1.26.0,<2.0.0`, `pandas>=2.0.0`, `scikit-learn>=1.4.0`, `scipy>=1.13.0`
  (required for multidomain PCA+Ensemble)
- `google-cloud-storage>=2.14.0` (GCS model download)
- `openai>=1.30.0`, `azure-identity>=1.16.0` (optional, present but not actively used)
- `langchain-core==1.3.3` (upgraded from 1.2.23)
- `langgraph>=1.0.25` (upgraded from 1.0.10)
- `typing_extensions>=4.9.0`, `nest-asyncio>=1.5.8`

**POC removes:**
- `langfuse==1.14.0` (telemetry SDK)

**release/path14 pins (Snyk SLA compliance):**
- `langchain-core==1.2.23` (Snyk-compliant, addresses directory traversal CVE)
- `langgraph==1.0.10`, `langgraph-checkpoint==4.0.1`, `langgraph-checkpoint-sqlite==3.0.3`

**Resolution**: **POC wins** (ML dependencies are non-negotiable for the multidomain
classifier). **Flag for security review**: `langchain-core` is upgraded from 1.2.23 to
1.3.3. The Snyk SLA required 1.2.22+ for a specific CVE — confirm 1.3.3 also covers it.
`langfuse` removal is acceptable if CVS has other telemetry in place.

---

### 2.14 `nodes/safety.py` — ❌ DIRECT CONFLICT

**POC removes:**
- `SYSTEM_PROMPT_FINGERPRINTS` list and `_check_system_prompt_leakage()` function
- "Step 0.5: Check for System Prompt Leakage" block in `response_safety_pii_postcheck_node`
- `check_for_prompt_injection()` method (Gemini-based semantic injection detection)
- Corresponding `SafetyCheck` docstring references (updated from 4-method to 3-method)

**release/path14:** has all of these security features.

**Resolution**: **POC wins** per the priority rule. **This is the most significant
security regression in the merge.** Three security layers are removed:
1. System prompt leakage fingerprint detection (post-response check)
2. Gemini-based prompt injection semantic detection (pre-query check)
3. The `PROMPT_INJECTION` enum value in `SafetyViolationType` (also removed in `core/node_models.py`)

Post-merge action: the security team should assess whether this removal is intentional
(e.g., these checks were found to produce false positives) or accidental. If accidental,
these blocks should be re-introduced in a follow-up PR.

---

### 2.15 `core/node_models.py` — ✅ NON-CONFLICTING (trivial)

**POC removes:** `SafetyViolationType.PROMPT_INJECTION` enum value.

This is consistent with the removal of prompt injection detection in `nodes/safety.py`.
Both changes must go together or neither. POC wins.

---

### 2.16 `core/logger.py` — ✅ NON-CONFLICTING

**POC changes:** log level `ERROR` → `INFO`.

More verbose logging. Safe, no conflict with any logic.

---

### 2.17 `services/llm_connection.py` — ✅ NON-CONFLICTING

**POC adds:** `response_mime_type` and `response_schema` to `GenerateRequest` +
passes them into the `_generate_core` Gemini call.

This enables native JSON-mode structured output, used by the response agent's
`LLMStructuredResponse` model. Pure addition, no conflict with release/path14's version.

---

### 2.18 `services/pii_protection.py` — ⚠️ POTENTIAL CONFLICT

**POC changes:** narrows SSN regex patterns (removes the more permissive separator
variants and "Social Security Number" full-phrase matching); removes
`check_for_prompt_injection()` from `SafetyCheck`; updates docstring.

**release/path14:** broader SSN detection.

**Resolution**: **POC wins.** The narrowed SSN patterns reduce false positives.
The `check_for_prompt_injection()` removal is consistent with `nodes/safety.py`.

---

### 2.19 `nodes/context.py` — ❌ DIRECT CONFLICT

**POC removes from `update_memory_node`:**
- `user_session = state.get("user_session")` extraction
- `response_id = state.get("response_id")` extraction
- Their logging lines and their passing to `persistence_store.save_conversation()`

**Resolution**: **POC wins.** The `user_session` and `response_id` parameters in the
persistence interface are also removed in POC (see §2.20). The two changes must be kept
in sync.

---

### 2.20 `persistence/__init__.py` — ❌ DIRECT CONFLICT

**POC reverts:**
- `save_conversation()` signature: removes `user_session`, `response_id` params
- `get_conversation_history()`: removes `user_session` param; changes return type
  from `Optional[Dict]` back to `List[Dict]`
- Removes abstract method `get_feedback_for_responses()`
- `delete_session_conversations()` param renamed back to `session_id`

**release/path14:** all of these `user_session`-keyed persistence methods exist.

**Resolution**: **POC wins.** This is part of the deliberate removal of the
persistent-history-across-sessions feature.

---

### 2.21 `persistence/mongodb_store.py` — ❌ DIRECT CONFLICT

**POC reverts:** MongoDB indexes back to `session_id`-based (removes
`user_session`-based primary index); removes `user_session`-keyed document schema from
`save_conversation()`.

**Resolution**: **POC wins**, consistent with §2.20.

---

### 2.22 `persistence/sqlite_store.py` — ⚠️ POTENTIAL CONFLICT

Likely mirrors the interface changes in §2.20. Confirm the POC version does not
have stub implementations of the removed methods that would cause an
`AbstractMethodNotImplemented` error at runtime.

---

### 2.23 `classifiers/embedded_classifier.py` — ✅ NON-CONFLICTING

**POC adds:** 18 new intent example banks (NDC, Manufacturer, Generic, Brand, Refills,
DaysSupply, PriorAuth, Diagnosis, Settlement, PharmType, Plan, Pharmacy, Prescriber,
Pricing, Status, RejectCode, DrugLast, Month, ClaimNum) to `CVS_INTENT_EXAMPLES`.

Pure addition at the bottom of the dict. No conflict with release/path14's entries.

---

### 2.24 `classifiers/intent_classifier_wrapper.py` — ⚠️ POTENTIAL CONFLICT

Likely adds multidomain classifier call path. Needs final verification that the POC
version does not break the embedding-only fallback path (for when
`use_multidomain_classifier=False`).

---

### 2.25 `agents/__init__.py` — ✅ NON-CONFLICTING

**POC adds:** a one-line comment clarifying the import. Trivial.

---

### 2.26 `.github/workflows/cd.yaml` — ⚠️ POTENTIAL CONFLICT

**POC triggers on:** `IntentDetectionPOC` and `MVP-2-April23-release`
**release/path14 triggers on:** `MVP-2-May28-release` and `kunwar-history-api`

**Resolution**: After merging into `release/path14`, the CD trigger should include
`release/path14` in its branch list, not `IntentDetectionPOC`. Update post-merge.

---

### 2.27 `.github/workflows/ci.yaml` — ⚠️ POTENTIAL CONFLICT

**POC uses:** `contents: read` (restrictive)
**release/path14 uses:** `contents: write`, `actions: read`

**Resolution**: **POC wins** (least-privilege is correct for a test job).

---

### 2.28 `deploy-configs/*/v1.yaml` — ⚠️ POTENTIAL CONFLICT (production impact)

| Setting | release/path14 | POC |
|---|---|---|
| prod `maxReplicas` | 4 | 3 |
| prod CPU request/limit | `8` | `2000m` |
| prod memory request/limit | `24Gi` | `2Gi` |
| prod image tag | SHA `537b700...` | SHA `e0c8f47...` |
| dev image tag | SHA `1735c69...` | commented out |
| qa image tag | SHA `537b700...` | SHA `079a21b...` |

**Resolution**: **POC wins per priority rule.** However, the 12× CPU and memory
reduction (`8`→`2000m`, `24Gi`→`2Gi`) is significant and almost certainly reflects
dev/test sizing, not production intent. **Before deploying to prod**, the infra/DevOps
team must confirm correct resource sizing for the multidomain classifier (which loads
a ~50–200 MB `.pkl` file into RAM at startup).

---

### 2.29 Files DELETED in POC (existed in `release/path14`) — ❌ DIRECT CONFLICT

These are entirely removed by the POC. With `-X theirs`, git will take the deletion.

| File | What it provided | Risk of loss |
|---|---|---|
| `api/history_routes.py` | GET `/session/{session_uuid}` endpoint — paginated history, feedback enrichment | Medium — if UI depends on this endpoint, requests will 404 |
| `agents/intent_agent_not_used.py` | Unused legacy intent agent | None |
| `.idea/` files | JetBrains IDE config | None |

**Action required**: Confirm with the UI/frontend team whether `api/history_routes.py`
is actively called by the myClaims web UI before the merge is deployed.

---

### 2.30 Files ADDED only in POC (safe includes) — ✅ NON-CONFLICTING

All files in the following directories/groups come in cleanly with the merge:

- `multidomain_intent_detection/` — the entire PCA+Ensemble+LLM fallback package
- `Claims_search_api/` — the member-history search pipeline
- `.claude/` — Claude Code settings (agents, commands, settings.json)
- `docs/` — new architecture documentation
- `suggested_changes/` — analysis documents
- `CLAUDE.md`, `ANALYSIS_LLM_CLARIFICATION_ROUTING.md`, etc.
- `tests/test_gcs_model_loading.py`, `test_multidomain_intent_detection.py`
- `PersonalExplaination/`, `ImportantData/` — dev analysis artifacts

**Note on `bearer.txt` and `Claims_search_api/.bearer.txt`**: These files appear to
contain raw API bearer tokens. They must be excluded from the merge or added to
`.gitignore`. **Never commit bearer tokens to source control.**

---

## 3. Pre-Merge Checklist (Ordered by Priority)

### P0 — Must Fix Before Merge (Security)

- [ ] **Remove hardcoded credentials from `config/config.py`:**
  - `redis_password` → revert to `""`
  - `redis_host` → revert to `""`
  - `mongodb_password` remains `""` (already empty)
  - `mongodb_host` → revert to `""`
  - `mongodb_database_name` → revert to `""`
  - `mongodb_connection_string` → revert to `"mongodb://localhost:27017"`
  - `claims_history_search_x_api_key` → revert to `""`
  - `project_id` → revert to `"pbm-nonprod-myclaims"` (or confirm correct prod value)

- [ ] **Remove or gitignore bearer token files:**
  - `bearer.txt`
  - `Claims_search_api/.bearer.txt`

- [ ] **Rotate any credentials already committed:** Since these credentials have been
  on a remote branch (`poc/intent-detection`), they must be treated as compromised
  and rotated regardless of whether the history is cleaned up.

### P1 — Must Confirm Before Deployment

- [ ] Confirm `api/history_routes.py` removal does not break myClaims UI
  (GET `/pss/pbmassist/v1/conversation/history/session/{session_uuid}` will 404)
- [ ] Confirm production pod resource sizing (`2000m` CPU, `2Gi` RAM) is sufficient
  for the multidomain classifier with expected traffic
- [ ] Confirm `v3_pipeline.pkl` is available in GCS and CI has the `gsutil cp` step
  to download it before `docker build` (Dockerfile now requires it)

### P2 — Should Verify Post-Merge

- [ ] Verify `langchain-core==1.3.3` covers the CVE addressed by Snyk SLA (was 1.2.22+)
- [ ] Confirm removal of prompt injection Gemini detection (`check_for_prompt_injection`)
  and system prompt leakage fingerprints is intentional, not accidental
- [ ] Evaluate `langchain/langgraph-api:3.11` base image against CVS container security
  policy (was distroless in release/path14)
- [ ] Update CD trigger branches in `.github/workflows/cd.yaml` to include `release/path14`

### P3 — Housekeeping

- [ ] Remove `suggested_changes/`, `PersonalExplaination/`, `ImportantData/` from the
  merge if they are dev-analysis artifacts not intended for the release branch
- [ ] Add `*.pkl` to `.gitignore` if not already present (binary model artifacts
  should not be committed)
- [ ] Remove `accuracy_results.csv`, `combinedtestset.csv`, `expected_intent_Final.jsonl`,
  `incorrect_llm_predictions.csv` from the release branch (training artifacts)

---

## 4. Recommended Merge Commands

```bash
# 1. Create a local tracking branch for release/path14
git fetch origin release/path14
git checkout -b release/path14 --track origin/release/path14

# 2. Fix P0 credential issues in config/config.py on poc/intent-detection FIRST
#    (or cherry-pick a fix commit onto poc/intent-detection)

# 3. Merge with POC as the authoritative side
git merge --allow-unrelated-histories -X theirs poc/intent-detection \
    -m "merge: integrate poc/intent-detection multidomain intent detection into release/path14"

# 4. Verify no conflicts remain
git status

# 5. Run startup check
python -c "import main; print('Import OK')"

# 6. Run tests
pytest tests/ -v --tb=short
```

If the `-X theirs` strategy produces any unexpected results on specific files, resolve
them individually using:

```bash
git checkout poc/intent-detection -- <file>
git add <file>
```

---

## 5. Self-Criticism Pass

> *Did I miss any subtle dependency?*

- **Yes**: The `classifiers/intent_classifier_wrapper.py` diff was not fully reviewed.
  It likely wraps the multidomain classifier call. The fallback path (embedding-only)
  must still work when `use_multidomain_classifier=False`. Final review required.

- **Yes**: `persistence/sqlite_store.py` diff was not fully analyzed. The abstract
  interface changes in `persistence/__init__.py` (§2.20) require all concrete
  implementations to be updated. Confirm `sqlite_store.py` does NOT still have stub
  implementations of `get_feedback_for_responses()` or the old `user_session`
  signatures — that would cause `AbstractMethodNotImplemented` at runtime when SQLite
  persistence is used.

> *Am I accidentally prioritizing `release/path14` anywhere?*

- The security concerns flagged (credential cleanup, safety feature regressions) are
  documented as *required actions*, not as *reasons to prefer release/path14*. The
  POC version wins in all cases; the flags are for the team to consciously address,
  not for git to auto-resolve.

> *Could this integration introduce a silent regression in intent detection?*

- **Potential silent regression**: The `confidence_threshold` is raised from 0.6 to
  0.85. Any query the old embedding classifier classified with confidence 0.6–0.84
  would previously go to `build_context` but will now route through `llm_judge` or
  `clarification`. This is intentional for the multidomain classifier. If for any
  reason the multidomain classifier is unavailable and the system falls back to the
  embedding classifier, the 0.85 threshold will cause significantly more queries to
  hit `llm_judge`. Consider whether `domain_config.json` should have
  separate thresholds per classifier type, or whether the fallback path should
  temporarily lower the threshold.

- **Potential silent regression**: The `.gitignore` shows as a binary diff between the
  two branches. The POC version adds `CLAUDE.local.md` and `.claude/settings.local.json`
  exclusions. These are benign additions but the binary diff flag should be
  investigated (likely Windows CRLF vs LF encoding difference — use `-X theirs` and
  normalize line endings post-merge).

---

*Generated: 2026-06-05 | Branch: poc/intent-detection → release/path14 | Authority: POC wins all conflicts*
