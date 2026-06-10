# Merge Guide: `poc/intent-detection` → `release/path14`

> **Analysis method:** REACT loop (5 iterations) + `git merge-tree --allow-unrelated-histories` + per-file deep diffs  
> **Date:** 2026-06-04  
> **Repositories:** `pss-myclaims-ai-agent_IntentDetectionPOC` (source) → `pss-myclaims-ai-agent` (target)

---

## ⚠️ CRITICAL: UNRELATED HISTORIES

These two branches share the same remote (`pss-myclaims-ai-agent.git`) but have **no common ancestor commit**. Every overlapping file is flagged as `CONFLICT (add/add)` by git.

**What this means:**
- A plain `git merge poc/intent-detection` will fail with *"refusing to merge unrelated histories"*
- `git merge --allow-unrelated-histories` will conflict on all 39 shared files simultaneously — not practical to resolve inline
- **Recommended approach: Targeted file-copy strategy** (§3), not a direct merge

---

## 1. Situation Summary

| | Source | Target |
|---|---|---|
| **Branch** | `poc/intent-detection` | `origin/release/path14` |
| **Repo** | `pss-myclaims-ai-agent_IntentDetectionPOC` | `pss-myclaims-ai-agent` |
| **Port** | 5000 | 8001 |
| **Primary purpose** | Multidomain intent detection (PCA + Ensemble + LLM fallback) | Production (embedding classifier + user_session persistence) |
| **Tip commit** | `e3f6abb` | `296e79c` |
| **Common ancestor** | **None** | — |

### What each branch uniquely contains

**Only in `poc/intent-detection`** (needs to be evaluated for carrying forward):
- `multidomain_intent_detection/` — entire PCA+Ensemble+LLM fallback module
- `Claims_search_api/` — multi-claim history search module with v1/v2 nodes, filter extractor, intent router, query responder
- `classifiers/nli_domain_router.py` — NLI domain routing
- `prompt_templates/domain_prompts/` — domain-specific prompt templates (all LLM fallback prompts)
- Test datasets (`ImportantData/`, `accuracy_results.csv`, `combinedtestset.csv`)
- Analysis docs (`ANALYSIS_LLM_CLARIFICATION_ROUTING.md`, `INTENT_DETECTION_DEEP_DIVE.md`, etc.)
- `.claude/agents/` and `.claude/commands/` (Claude Code tooling config)
- **DO NOT carry over:** `bearer.txt`, `Claims_search_api/.bearer.txt` (raw bearer tokens)

**Only in `origin/release/path14`** (production additions to keep):
- `api/history_routes.py` — new conversation history API
- `agents/intent_agent_not_used.py` — archived agent (harmless)
- `.idea/` — IDE files (**do not copy** — already in `.gitignore`)

---

## 2. Conflict Inventory — All 39 Files

All conflicts are `add/add` type (no common base). The table below gives the resolution for each.

**Legend:** 🟢 Take PROD wholesale · 🔵 Manual merge required · 🔴 Critical — read §4 carefully

| # | File | Severity | Resolution |
|---|------|----------|------------|
| 1 | `.github/workflows/cd.yaml` | Low | 🟢 Take PROD — CD trigger branches are production-controlled |
| 2 | `.github/workflows/ci.yaml` | Low | 🟢 Take PROD — CI pipeline is unchanged between branches |
| 3 | `.gitignore` | Low | 🔵 Manual merge — binary diff (line endings differ); add POC-specific ignores to PROD |
| 4 | `Dockerfile` | High | 🟢 Take PROD — distroless multi-stage Python 3.12 build is production-correct |
| 5 | `agents/__init__.py` | Low | 🟢 Take PROD |
| 6 | `agents/extended_intent_agent_node.py` | High | 🟢 Take PROD — 196 POC-only lines removed (domain routing, CHS normalization, LLM fallback scoring) |
| 7 | `agents/response_agent.py` | Medium | 🟢 Take PROD |
| 8 | `api/routes.py` | High | 🔵 Manual merge (§4.1) — PROD adds `session_uuid`; both sides have structural changes |
| 9 | `build.sh` | Low | 🟢 Take PROD |
| 10 | `classifiers/embedded_classifier.py` | High | 🔴 Decision required (§4.2) — POC has 19 extra intent categories for Claims_search_api |
| 11 | `classifiers/intent_classifier_wrapper.py` | Medium | 🟢 Take PROD |
| 12 | `config/api_routing_config.py` | Medium | 🟢 Take PROD |
| 13 | `config/config.py` | High | 🟢 Take PROD — drops hardcoded dev credentials, retains correct project_id |
| 14 | `config/domain_config.json` | Medium | 🟢 Take PROD |
| 15 | `core/logger.py` | Low | 🟢 Take PROD |
| 16 | `core/node_models.py` | Medium | 🟢 Take PROD |
| 17 | `deploy-configs/pbmassist-dev/values/v1.yaml` | Low | 🟢 Take PROD — CD pipeline manages image tag |
| 18 | `deploy-configs/pbmassist-prod/values/v1.yaml` | Low | 🟢 Take PROD |
| 19 | `deploy-configs/pbmassist-qa/values/v1.yaml` | Low | 🟢 Take PROD |
| 20 | `langgraph_agent.py` | High | 🟢 Take PROD — PROD adds `_seed_redis_from_history()` + `user_session` threading |
| 21 | `main.py` | High | 🟢 Take PROD — PROD drops multidomain startup, adds history router, correct port 8001 |
| 22 | `nodes/clarification.py` | High | 🟢 Take PROD — drops vague-query/CHS-sequence POC logic |
| 23 | `nodes/confidence.py` | High | 🟢 Take PROD — drops all LLM-fallback/CHS routing special-cases |
| 24 | `nodes/context.py` | Low | 🟢 Take PROD — additive only: adds `user_session`/`response_id` wiring |
| 25 | `nodes/safety.py` | Medium | 🟢 Take PROD — adds system-prompt leakage detection (new security feature) |
| 26 | `persistence/__init__.py` | Low | 🟢 Take PROD |
| 27 | `persistence/mongodb_store.py` | Critical | 🔴 Take PROD (§4.3) — breaking schema change; migration script required |
| 28 | `persistence/sqlite_store.py` | Critical | 🔴 Take PROD (§4.3) — breaking schema change for dev store |
| 29 | `requirements.txt` | High | 🟢 Take PROD — Snyk-pinned versions; drops ML stack (scikit-learn, numpy, etc.) |
| 30 | `scripts/check_gcp_config.py` | Low | 🟢 Take PROD |
| 31 | `scripts/create_logs_collection.py` | Low | 🟢 Take PROD |
| 32 | `scripts/create_mongodb_collections.py` | Low | 🟢 Take PROD |
| 33 | `scripts/setup_mongodb_collections.py` | Low | 🟢 Take PROD |
| 34 | `scripts/test_mongodb_connection.py` | Low | 🟢 Take PROD |
| 35 | `services/llm_connection.py` | Medium | 🔴 Check first (§4.4) — PROD removed `response_mime_type`/`response_schema` fields |
| 36 | `services/pii_protection.py` | Medium | 🟢 Take PROD |
| 37 | `setup_cert.sh` | Low | 🟢 Take PROD |
| 38 | `start_server.sh` | Low | 🟢 Take PROD |
| 39 | `state/schema.py` | Critical | 🔴 Manual merge (§4.5) — POC has 4 multidomain fields; PROD adds `user_session` |
| 40 | `tests/test_safety_nodes.py` | Low | 🟢 Take PROD |

**Summary:** 35 files → take PROD wholesale · 2 files → manual merge · 3 files → read decision notes first

---

## 3. Recommended Merge Strategy

Because of unrelated histories, **do not use `git merge --allow-unrelated-histories`** — it would create 39 simultaneous conflict blocks. Instead:

### Step-by-step approach

```bash
# In pss-myclaims-ai-agent (the PRODUCTION repo)

# 1. Make sure release/path14 is up to date
git checkout release/path14
git pull origin release/path14

# 2. Create an integration branch
git checkout -b feature/multidomain-intent-integration

# 3. Add the POC repo as a remote so its files are accessible
git remote add poc ../pss-myclaims-ai-agent_IntentDetectionPOC
git fetch poc

# 4. Copy POC-UNIQUE directories wholesale (no conflicts — these don't exist in PROD)
git checkout poc/poc/intent-detection -- multidomain_intent_detection/
git checkout poc/poc/intent-detection -- prompt_templates/domain_prompts/

# 5. For Claims_search_api: decide first (see §4.2)
#    If including it:
git checkout poc/poc/intent-detection -- Claims_search_api/

# 6. Handle the 5 non-trivial files (§4)
#    - api/routes.py     → manual merge (take PROD, add specific POC lines)
#    - classifiers/embedded_classifier.py → decision required
#    - persistence/      → take PROD, run migration script
#    - services/llm_connection.py → check if LLM fallback needs it
#    - state/schema.py   → manual merge

# 7. For the 35 "take PROD wholesale" files: no action needed
#    (PROD version is already the current file on release/path14)

# 8. Update requirements.txt for new ML dependencies
#    (see §5 — add scikit-learn, numpy, pandas, scipy, google-cloud-storage)

# 9. Commit and test
git add -A
git commit -m "feat: integrate multidomain intent detection from poc/intent-detection"
```

---

## 4. Files Requiring Manual Action

### 4.1 `api/routes.py` — Manual Merge

**What PROD added (keep these):**
```python
# On ChatRequest
session_uuid: Optional[str] = None  # stable per-login session ID from UI

# On ChatResponse
session_uuid: Optional[str] = None  # echoed back to the client

# In chat() and chat_stream() handlers:
user_session = request.session_uuid
# ...
run_graph(..., user_session=user_session)
# Streaming complete event:
"user_session": final_state.get("user_session"),
"response_id": final_state.get("response_id"),
```

**What POC added (discard these):**
- `IntentStageResult` and `IntentResolutionTrail` Pydantic models
- `intent_resolution_trail` field on `ChatResponse`
- `x_api_key` and `x_clientrefid` header capture and forwarding

**Action:** Use the PROD version of `api/routes.py` as-is. No changes needed — the PROD file already has `session_uuid` and does not need the POC additions.

---

### 4.2 `classifiers/embedded_classifier.py` — Decision Required

The POC added **19 intent categories** to the training example dictionary for the `Claims_search_api` pipeline:
`NDC`, `Manufacturer`, `Generic`, `Brand`, `Refills`, `DaysSupply`, `PriorAuth`, `Diagnosis`, `Settlement`, `PharmType`, `Plan`, `Pharmacy`, `Prescriber`, `Pricing`, `Status`, `RejectCode`, `DrugLast`, `Month`, `ClaimNum`

**Option A — Include all POC intent categories (if Claims_search_api is being merged):**
```bash
# Take the POC version which has all intent categories
git checkout poc/poc/intent-detection -- classifiers/embedded_classifier.py
```

**Option B — Take PROD as-is (if Claims_search_api is NOT being merged):**
```bash
# No action — PROD version is already on release/path14
# The 19 intent categories are not needed if Claims_search_api is excluded
```

**Decision guideline:** If you are bringing `Claims_search_api/` into production (the multi-claim history search feature), choose Option A. If only bringing `multidomain_intent_detection/` (the PCA+Ensemble classifier), choose Option B.

---

### 4.3 `persistence/mongodb_store.py` and `persistence/sqlite_store.py` — Breaking Schema Change

**What changed:** Production redesigned the conversation history storage model:

| Aspect | POC (old) | Production (new) |
|---|---|---|
| Schema | One document per conversation **turn** | One document per **`user_session`** (accumulating array) |
| Document key | `uuid4()` | `user_session` (stable per-login ID) |
| Storage shape | Flat: `user_message`, `agent_response`, `intent`, `duration_ms` | `conversation_history: [{role, content, timestamp, response_id}]` |
| New method | — | `get_feedback_for_responses()` — batch fetch feedback |
| `delete_session_conversations` param | `session_id` | `user_session` |

**Action:** Take PROD versions wholesale for both files. Then create a migration script for any existing MongoDB data:

```python
# migration_conv_history.py — run ONCE against any existing MongoDB data
# Converts old per-turn documents to new per-session array format
from pymongo import MongoClient

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
old = db["conversation_history"]
new = db["conversation_history_v2"]

by_session = {}
for doc in old.find():
    key = doc.get("session_id", doc.get("user_session", "unknown"))
    by_session.setdefault(key, []).append({
        "role": "user",   "content": doc.get("user_message", ""), "timestamp": doc.get("timestamp")
    })
    by_session.setdefault(key, []).append({
        "role": "assistant", "content": doc.get("agent_response", ""), "timestamp": doc.get("timestamp")
    })

for session_id, history in by_session.items():
    new.update_one(
        {"_id": session_id},
        {"$setOnInsert": {"session_id": session_id, "created_at": history[0]["timestamp"]},
         "$set": {"conversation_history": history, "updated_at": history[-1]["timestamp"]}},
        upsert=True
    )
print(f"Migrated {len(by_session)} sessions")
```

---

### 4.4 `services/llm_connection.py` — Verify First

**What PROD removed:**
```python
# REMOVED from GenerateRequest:
response_mime_type: Optional[str] = None
response_schema: Optional[Any] = None

# REMOVED from _generate_core call:
response_mime_type=req.response_mime_type or None,
response_schema=req.response_schema or None,
```

**Check before deciding:**
```bash
# Does multidomain_intent_detection use these fields?
grep -r "response_mime_type\|response_schema\|GenerateRequest" multidomain_intent_detection/ prompt_templates/
```

- If **no matches**: take PROD wholesale — the fields are unused.
- If **matches found**: restore the two fields to PROD's `GenerateRequest` and the `_generate_core` call, then take PROD as base.

The LLM fallback in `multidomain_intent_detection/llm_fallback.py` calls the Vertex AI SDK directly (not `GenerateRequest`), so these fields are likely **not needed**. Verify to be sure.

---

### 4.5 `state/schema.py` — Manual Merge (Critical)

**POC has these fields (needed if multidomain classifier is integrated):**
```python
llm_fallback_confidence: Optional[float]  # Score from the LLM fallback step
ensemble_intent: Optional[str]            # Top intent from ensemble vote
ensemble_confidence: Optional[float]      # Ensemble's pre-calibration score
llm_fallback_intent: Optional[str]        # Final intent after LLM fallback
```

**PROD has this field (required — do not remove):**
```python
user_session: Optional[str]               # Stable per-login session ID for MongoDB
```
And in `create_initial_state`:
```python
def create_initial_state(session_id: str, user_session: Optional[str] = None, ...) -> AgentState:
    ...
    "user_session": user_session,
```

**Resolution — take PROD as base and ADD the four multidomain fields:**

```python
# In AgentState TypedDict, add these four fields (from POC) alongside the existing PROD fields:
llm_fallback_confidence: Optional[float]
ensemble_intent: Optional[str]
ensemble_confidence: Optional[float]
llm_fallback_intent: Optional[str]
```

Ensure `create_initial_state` initialises them to `None`:
```python
"llm_fallback_confidence": None,
"ensemble_intent": None,
"ensemble_confidence": None,
"llm_fallback_intent": None,
```

**Keep PROD's `user_session` field and `create_initial_state` signature exactly as-is.**

---

### 4.6 `.gitignore` — Line-ending Fix

Both files differ at the binary level (CRLF vs LF). Extract both text versions and merge:

```bash
# From the integration branch in pss-myclaims-ai-agent:
git show HEAD:.gitignore > /tmp/prod_gitignore.txt
git show poc/poc/intent-detection:.gitignore > /tmp/poc_gitignore.txt

# Manually add POC-specific ignores to the PROD file:
# multidomain_intent_detection/artifacts/*.pkl
# accuracy_results.csv
# ImportantData/
# bearer.txt
# *.bearer.txt
```

Ensure final `.gitignore` uses LF line endings (`dos2unix .gitignore` or editor setting).

---

## 5. `requirements.txt` — Additional Dependencies for Multidomain Classifier

The PROD `requirements.txt` **does not include** the ML stack required by `multidomain_intent_detection/`. After taking PROD wholesale, add these lines:

```
# Multidomain intent detection — PCA + Ensemble classifier
numpy>=1.26.0,<2.0.0
pandas>=2.0.0
scikit-learn>=1.4.0
scipy>=1.13.0

# GCS model artifact download (v3_pipeline.pkl)
google-cloud-storage>=2.14.0
```

> **Do NOT add** `azure-identity` or `openai` — those are POC-only dependencies not used by the multidomain module.  
> **Verify** scikit-learn/numpy compatibility with Python 3.12 (prod uses 3.12, POC uses 3.11).

---

## 6. POC-Unique Modules to Carry Forward

These directories exist **only in the POC** and must be copied to the production branch:

### Always carry over:
| Directory/File | Why |
|---|---|
| `multidomain_intent_detection/` | The PCA+Ensemble+LLM fallback classifier itself |
| `prompt_templates/domain_prompts/` | LLM fallback prompt templates (hundreds of lines of curated disambiguation rules) |

### Carry over IF Claims_search_api is included:
| Directory/File | Why |
|---|---|
| `Claims_search_api/` | Multi-claim history search v1+v2 nodes, filter extractor, intent router |
| `classifiers/nli_domain_router.py` | NLI domain routing (optional, used by some CHS paths) |

### Explicitly DO NOT carry over:
| File | Reason |
|---|---|
| `bearer.txt`, `Claims_search_api/.bearer.txt` | Raw tokens — gitignored for a reason |
| `PersonalExplaination/` | Personal analysis notes, not production code |
| `SESSION_FIXES.md`, `ANALYSIS_LLM_CLARIFICATION_ROUTING.md` | Local analysis docs |
| `accuracy_results.csv`, `ImportantData/` | Test datasets, not production |
| `.claude/`, `.mcp.json` | Dev-environment tooling, not application code |
| `LocalHostChatEndpointRepititiveHitting.py` | Load-test script |

---

## 7. `config/config.py` — Security Warning

The POC `config.py` contains **hardcoded dev credentials**:
```python
# POC version (DO NOT merge these values):
redis_host: str = "redis-..."           # real hostname
redis_password: str = "..."             # real password
mongodb_host: str = "..."               # real hostname
mongodb_connection_string: str = "..."  # real connection string with credentials
```

The PROD version correctly blanks these to empty strings or localhost defaults. **Always take PROD wholesale** — never carry over any hardcoded credential values.

---

## 8. Complete Merge Walkthrough

```bash
# === In pss-myclaims-ai-agent (PRODUCTION repo) ===

git checkout release/path14
git pull origin release/path14
git checkout -b feature/multidomain-intent-integration

# Add POC repo as a remote
git remote add poc-repo ../pss-myclaims-ai-agent_IntentDetectionPOC
git fetch poc-repo

# --- Copy POC-unique modules (no conflicts) ---
git checkout poc-repo/poc/intent-detection -- multidomain_intent_detection/
git checkout poc-repo/poc/intent-detection -- prompt_templates/domain_prompts/
# Decision: include Claims_search_api? If yes:
git checkout poc-repo/poc/intent-detection -- Claims_search_api/

# --- state/schema.py: manual merge (§4.5) ---
# Edit state/schema.py: add the 4 multidomain fields to AgentState
# Keep user_session from PROD (already present)
git add state/schema.py

# --- classifiers/embedded_classifier.py: decision (§4.2) ---
# Option A (with Claims_search_api):
git checkout poc-repo/poc/intent-detection -- classifiers/embedded_classifier.py
# Option B (without): no action needed

# --- services/llm_connection.py: verify first (§4.4) ---
grep -r "response_mime_type\|response_schema" multidomain_intent_detection/ prompt_templates/
# If no matches: no action. If matches: add fields back to PROD file.

# --- requirements.txt: add ML dependencies (§5) ---
# Manually add numpy, pandas, scikit-learn, scipy, google-cloud-storage to PROD requirements.txt

# --- .gitignore: fix line endings + add POC ignores (§4.6) ---
# Manually add: multidomain_intent_detection/artifacts/*.pkl, bearer.txt, etc.

# --- All other 35 files: PROD version already correct, no action ---

# --- Run migration script for existing MongoDB data (§4.3) ---
python migration_conv_history.py

# --- Verify no credentials leaked from POC ---
grep -r "redis_password\|mongodb_connection_string" config/config.py
# Should show only empty string "" values

# --- Stage and commit ---
git add -A
git status   # Review staged files
git commit -m "feat: integrate multidomain intent detection from poc/intent-detection

- Brings in multidomain PCA+Ensemble+LLM fallback classifier
- Brings in domain-specific prompt templates
- state/schema.py: added 4 multidomain diagnostic fields alongside PROD user_session
- requirements.txt: added ML stack (scikit-learn, numpy, pandas, scipy, google-cloud-storage)
- All 35 overlapping files taken from release/path14 (Snyk-compliant, production-correct)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## 9. Post-Merge Verification Checklist

### Startup verification
- [ ] `python main.py` starts on port **8001** (not 5000)
- [ ] `[STARTUP] ✅ Multidomain classifier ready` log line appears
- [ ] `[STARTUP] ✅ Startup complete` appears with no errors
- [ ] Health endpoint `GET /health` returns `200 {"status": "healthy"}`

### Feature verification
- [ ] Intent classification returns `domain` field for claims-search queries
- [ ] LLM fallback triggers on low-confidence queries
- [ ] `user_session` round-trips through `POST /chat` (in request → in response)
- [ ] Conversation history seeds from MongoDB on session open (check logs for `_seed_redis_from_history`)
- [ ] Safety: prompt leakage check does not false-positive on normal responses

### Security verification
- [ ] `config/config.py` has no hardcoded dev credentials (`grep -r "redis_password" config/`)
- [ ] `bearer.txt` is NOT committed (check `git status` before pushing)
- [ ] All ML dependencies (`scikit-learn`, `numpy`, etc.) pass Snyk scan

### Regression verification
- [ ] Run existing test suite: `pytest tests/`
- [ ] `nodes/safety.py` prompt-leakage detection works (new test if needed)
- [ ] `api/history_routes.py` history endpoint still responds correctly
- [ ] `persistence/mongodb_store.py` upsert-style writes work (test with a dev MongoDB instance)

---

## 10. Abort / Rollback

```bash
# If anything goes wrong during the integration branch work:
git merge --abort        # If a merge was started
git checkout release/path14
git branch -D feature/multidomain-intent-integration
git remote remove poc-repo
```

---

## Appendix: REACT Analysis — Self-Criticism Loop

| Iteration | Observation | Self-criticism |
|---|---|---|
| 1 | `git merge-base` returns exit code 1 | Verified by checking earliest commits in both branches — confirmed truly unrelated histories, not a git config issue |
| 2 | All 39 overlapping files are `CONFLICT (add/add)` | Means git has zero context to auto-resolve any of them — each file needs an explicit decision |
| 3 | Most POC additions are multidomain-classifier scaffolding | Risk: might overlook a production-critical feature in a POC file. Mitigated by doing full diffs on all 39 files before deciding |
| 4 | `config/config.py` contains hardcoded dev credentials in POC | Flagged explicitly — a missed credential carry-over would be a security incident |
| 5 | `persistence/mongodb_store.py` is a breaking schema change | Any existing data in POC schema format would be silently dropped if migration is skipped — migration script required |
