# Rendering Agent Kill Switch — Design & Implementation Plan

**Date:** 2026-06-03  
**Status:** Pending implementation  
**Scope:** Backend-only change — zero UI modifications required  
**Repo:** `pss-myclaims-ai-agent`

---

## 1. Problem Statement

The MyClaims rendering agent (`MyclaimsRenderingAgent`) enriches chatbot responses with structured HTML tables and a JSON layout DSL (`render_dsl`). The UI code changes that consume these outputs were frozen on 2026-06-02 — **no further UI changes can be deployed.**

This creates a risk gap: if the rendering agent misbehaves after a backend deploy (bad HTML, malformed DSL, layout regressions, latency spikes), there is currently no mechanism to disable rendering from the backend side without:

- A full code rollback, or
- A UI code change (blocked by freeze)

**Goal:** Implement a backend-only, operator-controllable kill switch that forces `render_format: "text"` for every response, causing the UI to display plain text always — identical to pre-rendering behaviour.

---

## 2. Current Architecture — How Rendering Works End to End

### 2.1 Full Request Lifecycle (with Rendering)

```
HTTP POST /api/v1/chat  (or /api/v1/chat/stream)
  │
  ├─ run_graph()  ──────────────────────────────────────── LangGraph execution
  │     │
  │     ├─ orchestrator_node
  │     ├─ intent_agent_node
  │     ├─ call_claims_tool_node          ← fetches real claim data
  │     ├─ response_agent_node
  │     │     ├─ Gemini LLM produces:
  │     │     │     {"response": "...", "recommendations": [...]}
  │     │     │     ===RENDER_START=== {...layout JSON...} ===RENDER_END===
  │     │     ├─ _extract_render_dsl()   ← strips DSL block, stores in AgentState.render_dsl
  │     │     └─ _parse_response_with_recommendations()
  │     └─ safety nodes
  │
  ├─ process_rendering(final_state)  ──────────────────── POST-GRAPH, in routes.py
  │     │
  │     ├─ GUARD: intent in NO_RENDER_INTENTS?  → return render_format="text"
  │     ├─ GUARD: needs_clarification?           → return render_format="text"
  │     ├─ GUARD: no tool_results?               → return render_format="text"
  │     ├─ GUARD: tool_results.status != success?→ return render_format="text"
  │     │
  │     ├─ Unmask PII tokens in tool_results (token_mapping from state)
  │     ├─ validate_render_dsl(raw_dsl)           ← validates layout + field paths
  │     └─ MyclaimsRenderingAgent.execute(...)    ← produces html_content + css_content
  │
  └─ ChatResponse returned to UI
        render_format: "table" | "text"
        html_content: "<table ...>" | None
        css_content:  "<style ...>" | None
        answer_header: "Claim Details" | None
        render_dsl:   {...layout dict...} | None
        response:     "plain text answer" (always present)
```

### 2.2 Key Files Involved

| File | Role |
|---|---|
| `api/routes.py` | `process_rendering()` — post-graph rendering orchestrator; `ChatResponse` model |
| `config/config.py` | `Settings` — all feature flags and env-var overrides |
| `agents/post_processing/myclaims_rendering_agent.py` | `MyclaimsRenderingAgent.execute()` — HTML table builder |
| `agents/post_processing/render_dsl.py` | DSL dataclasses + `validate_render_dsl()` |
| `agents/post_processing/rendering_themes.py` | `NO_RENDER_INTENTS` — intents that always get text |
| `agents/response_agent.py` | `_extract_render_dsl()` — strips DSL block from LLM output |
| `state/schema.py` | `AgentState.render_dsl` — DSL carried through graph |

### 2.3 ChatResponse Fields the UI Consumes

The UI (Angular MFE `pss-myclaims-claiminquiry-mfe`) reads the following fields from the backend JSON response:

| Field | Type | Purpose |
|---|---|---|
| `response` | `string` | Primary answer text — **always shown** |
| `render_format` | `string` | `"text"` or `"table"` — **rendering gate** |
| `html_content` | `string \| null` | HTML table markup — used when `render_format="table"` |
| `css_content` | `string \| null` | Scoped CSS for the table |
| `render_dsl` | `object \| null` | JSON layout DSL for Angular-native renderer |
| `render_mode` | `string \| null` | Layout type hint (e.g. `"table"`) |
| `answer_header` | `string \| null` | Section heading above the table |
| `recommendations` | `array` | Follow-up chips |
| `response_id` | `string` | Feedback tracking |
| `session_uuid` | `string` | Stable session identifier |

**Critical:** The UI's rendering decision is gated entirely on `render_format`. When `render_format === "text"`, the UI renders `response` as plain text and ignores all other rendering fields. **Sending `render_format: "text"` from the backend is sufficient to disable all rendering, regardless of what else is in the payload.**

---

## 3. The Kill Switch Design

### 3.1 Mechanism

Add a single boolean flag `enable_rendering_agent` to `Settings` in `config/config.py`. Wire a one-line early-exit guard at the top of `process_rendering()` in `api/routes.py`. No other changes required.

**When `enable_rendering_agent=False`:**
- `process_rendering()` exits immediately, returning `{"render_format": "text", "render_dsl": None}`
- `MyclaimsRenderingAgent.execute()` is never called — zero extra latency
- `ChatResponse.render_format` is always `"text"`
- UI falls back to `{{ response }}` plain text — identical to pre-rendering behaviour
- `response` field is unaffected — the answer text is always returned as normal

**When `enable_rendering_agent=True` (default):**
- Full rendering pipeline runs as today
- No behaviour change

### 3.2 Change 1 — `config/config.py`

Location: after `max_recommendations` (currently line 190), before the inner `class Config:`.

```python
    # =========================================================================
    # RENDERING AGENT KILL SWITCH
    # =========================================================================
    # Set ENABLE_RENDERING_AGENT=false to force render_format="text" for every
    # response, bypassing the rendering agent entirely.  No UI or code deploy
    # needed — flip the env var and restart the pod.
    # When false: process_rendering() exits immediately with render_format="text"
    # and the UI falls back to showing the plain response text.
    enable_rendering_agent: bool = True  # ⚠️ Overridden by ENABLE_RENDERING_AGENT env var
```

**Why here:** Every feature-level on/off switch in this codebase lives in `Settings` (see `enable_recommendations`, `enable_safety_precheck`, `enable_streaming`). This follows the exact same pattern. Pydantic Settings automatically reads `ENABLE_RENDERING_AGENT` from the OS environment or `.env` file.

### 3.3 Change 2 — `api/routes.py`

Location: first two lines of the `try:` block inside `process_rendering()`, currently at line 70. Insert **before** the `from agents...` imports.

```python
def process_rendering(state_or_event_data: dict) -> dict:
    """Post-graph HTML rendering. NEVER raises — returns render_format='text' on any failure."""
    try:
        # Kill switch: operator can set ENABLE_RENDERING_AGENT=false to force text mode.
        if not settings.enable_rendering_agent:
            return {"render_format": "text", "render_dsl": None}

        from agents.post_processing.myclaims_rendering_agent import MyclaimsRenderingAgent
        ...
```

**Why at this exact position:**
- `settings` is already imported at the top of `routes.py` (line 18: `from config.config import settings`)
- Placing the guard before the `from agents...` lazy imports means the rendering agent module is never even loaded on the hot path — no import overhead, no LLM calls, no PII unmasking
- The existing `except Exception` at the bottom already returns `render_format="text"` — this guard short-circuits before any of that even runs

### 3.4 What Does NOT Change

| Component | Change needed | Reason |
|---|---|---|
| `pss-myclaims-claiminquiry-mfe` (Angular UI) | None | `render_format: "text"` already causes text fallback |
| `agents/response_agent.py` | None | LLM still generates the DSL block; it's just discarded in `process_rendering()` |
| `state/schema.py` | None | `AgentState.render_dsl` still populated; ignored downstream |
| `agents/post_processing/render_dsl.py` | None | `validate_render_dsl()` simply never called |
| `agents/post_processing/myclaims_rendering_agent.py` | None | `execute()` simply never called |
| Streaming endpoint (`/chat/stream`) | None | Both endpoints call `process_rendering()` — the guard fires in both |
| Kubernetes manifests | None | Env var injected at runtime, no manifest change needed |

---

## 4. Activation — How to Use the Kill Switch

### 4.1 Local Development (`.env` file)

```dotenv
# Disable rendering agent — force text-only mode
ENABLE_RENDERING_AGENT=false
```

Restart the server. Effective immediately.

### 4.2 Kubernetes / GKE (ConfigMap or deployment env)

**Option A — ConfigMap patch (no manifest commit needed):**

```bash
kubectl set env deployment/myclaims-ai-agent \
  ENABLE_RENDERING_AGENT=false \
  -n <namespace>
```

This triggers a rolling restart. No code change, no image rebuild.

**Option B — Edit the deployment manifest directly:**

In `myclaims-np-use4-gke_manifests/` (or the relevant deploy-config), add to the `env:` section of the container spec:

```yaml
- name: ENABLE_RENDERING_AGENT
  value: "false"
```

Then apply: `kubectl apply -f <manifest-path>`

**To re-enable:**

```bash
kubectl set env deployment/myclaims-ai-agent \
  ENABLE_RENDERING_AGENT=true \
  -n <namespace>
```

### 4.3 Verification After Flipping

Send any claim details query to the API. Confirm:

```json
{
  "response": "Your claim ... was paid on ...",
  "render_format": "text",
  "html_content": null,
  "css_content": null,
  "render_dsl": null
}
```

`render_format` must be `"text"` and `html_content` must be `null`.  
The UI will display the `response` string as plain text — no table, no DSL rendering.

---

## 5. Why `render_format` is the Right Field to Control

The UI decides rendering mode by checking `render_format` first. The decision tree in the Angular code is:

```
if (render_format === "table" && html_content) {
    // inject html_content via [innerHTML] — show rendered table
} else {
    // show response as plain text — {{ message.text }}
}
```

Forcing `render_format: "text"` collapses the entire rendering path in the UI to the plain-text branch, regardless of whether `html_content`, `css_content`, or `render_dsl` also have values. **The UI is already written to respect this contract — no UI change is needed.**

---

## 6. Existing Safety Nets (Already in Place)

`process_rendering()` already has four built-in guards that return `render_format="text"` without invoking the rendering agent:

| Guard condition | Line in routes.py | Effect |
|---|---|---|
| `intent in NO_RENDER_INTENTS` | ~105 | Text-only for unsupported intents |
| `needs_clarification == True` | ~106 | Text-only when agent needs more info |
| `not tool_results` | ~107 | Text-only when no data was fetched |
| `tool_results.status != "success"` | ~108 | Text-only on API errors |
| `except Exception` catch-all | ~150 | Text-only on any rendering crash |

The new kill switch sits **above** all of these — it is the first check, and it exits before any rendering logic runs. It is the operator-level override on top of the existing code-level guards.

---

## 7. Risk Assessment

### 7.1 Risk of the Kill Switch Itself

| Risk | Likelihood | Mitigation |
|---|---|---|
| Accidentally left `false` in production | Low | Clearly named env var; default is `true` |
| `settings` object not loaded at process_rendering call time | None | `settings` is a module-level singleton, always available |
| Kills rendering on both `/chat` and `/chat/stream` | Intended | Both call `process_rendering()` — consistent behaviour |
| `response` field still present and correct | Always | The guard only affects rendering output, not the text answer |

### 7.2 Risk of NOT Having the Kill Switch

| Scenario | Impact without kill switch | Impact with kill switch |
|---|---|---|
| Rendering agent produces malformed HTML | UI injects bad markup — visual corruption | Flip one env var, UI shows text |
| DSL validation fails for a new intent | Blank table / partial render | Flip one env var, UI shows text |
| Rendering agent adds significant latency | Every response delayed | Flip one env var, latency drops immediately |
| LLM DSL generation starts hallucinating layouts | Confusing UI tables | Flip one env var, UI shows text |
| Post-UI-freeze rendering bug discovered | Blocked — UI can't be changed | Flip one env var from backend |

---

## 8. Implementation Checklist

```
[ ] 1. Add `enable_rendering_agent: bool = True` to Settings in config/config.py
        — After max_recommendations, before class Config
        — Include the ⚠️ env var override comment (matches project convention)

[ ] 2. Add kill switch guard in process_rendering() in api/routes.py
        — First two lines of the try: block (lines 70-71)
        — Before the lazy from agents... imports
        — if not settings.enable_rendering_agent: return {"render_format": "text", "render_dsl": None}

[ ] 3. Verify in local dev
        — Add ENABLE_RENDERING_AGENT=false to .env
        — POST to /api/v1/chat with a claim details query
        — Confirm render_format="text" and html_content=null in response

[ ] 4. Verify default still works
        — Remove/set ENABLE_RENDERING_AGENT=true in .env
        — POST same query
        — Confirm render_format="table" and html_content has table markup

[ ] 5. Document env var in deploy-configs
        — Add ENABLE_RENDERING_AGENT=true to deploy-configs/<env>/env-vars or ConfigMap
        — Ensures the value is explicit in every environment
```

---

## 9. Relationship to Other Feature Flags

This follows the identical pattern used by all other feature flags in the codebase:

| Flag | Default | Env var |
|---|---|---|
| `enable_safety_precheck` | `True` | `ENABLE_SAFETY_PRECHECK` |
| `enable_safety_postcheck` | `True` | `ENABLE_SAFETY_POSTCHECK` |
| `enable_recommendations` | `True` | `ENABLE_RECOMMENDATIONS` |
| `enable_streaming` | `True` | `ENABLE_STREAMING` |
| `enable_claims_api_cache` | `True` | `ENABLE_CLAIMS_API_CACHE` |
| `enable_rendering_agent` | `True` | `ENABLE_RENDERING_AGENT` ← **new** |

All are `bool` fields in `Settings`, all default to `True` (feature on), all overridden by an env var of the same name in `SCREAMING_SNAKE_CASE`.

---

## 10. Summary

**Two file changes, zero UI changes, one env var flip to activate.**

```
config/config.py    → +1 flag:  enable_rendering_agent: bool = True
api/routes.py       → +2 lines: guard at top of process_rendering()
```

To disable rendering in any environment:

```bash
ENABLE_RENDERING_AGENT=false  # + pod restart
```

To re-enable:

```bash
ENABLE_RENDERING_AGENT=true   # + pod restart
```

The `response` field — plain text answer — is never affected. Users always get a readable answer.
