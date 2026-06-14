# Overrides Domain Onboarding — Prior Authorization Lookup

**Document Type:** Engineering Approach / Design Specification  
**Domain:** `override_domain` — Prior Authorization (PA) Lookup  
**Author:** Architecture Review (ReAct Analysis)  
**Status:** Pre-Implementation Review

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [New Files to Create](#2-new-files-to-create)
3. [Existing Files to Modify](#3-existing-files-to-modify)
4. [State Schema Additions](#4-state-schema-additions)
5. [Intent Definitions and Example Utterances](#5-intent-definitions-and-example-utterances)
6. [Redis Caching Design](#6-redis-caching-design)
7. [Prompt Template Strategy](#7-prompt-template-strategy)
8. [Error and Fallback Handling](#8-error-and-fallback-handling)
9. [Open Questions and Risks](#9-open-questions-and-risks)

---

## 1. High-Level Architecture

### Conceptual Flow

```
User Query (PA-related)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                   LangGraph Agent                       │
│                                                         │
│  orchestrator_node                                      │
│        │                                                │
│        ▼                                                │
│  safety_precheck_node                                   │
│        │                                                │
│        ▼                                                │
│  check_cache_node  ──── HIT ──────────────────────────► response_agent_node
│        │ MISS                                           │
│        ▼                                                │
│  extended_intent_agent_node                             │
│    • classify_intent_unified() → intent, domain         │
│    • extract_entities_unified() → {claim_id: "..."}     │
│    • get_api_config(intent) → required_entities=["claim_id"]
│    • sets state["domain_mapping"] = "override_domain"   │
│        │                                                │
│        ▼                                                │
│  confidence_check_router                                │
│    ├─ missing_slots → clarification_node                │
│    │       │                                            │
│    │       └─ user provides claim_id → back to intent   │
│    └─ slots satisfied                                   │
│        │                                                │
│        ▼                                                │
│  llm_judge_node                                         │
│        │                                                │
│        ▼                                                │
│  build_context_node                                     │
│        │                                                │
│        ▼                                                │
│  _route_after_build_context()                           │
│    ├─ is_claims_search_query()  → call_claims_search    │
│    ├─ is_override_domain_query()→ call_overrides_tool ◄─┤ NEW
│    └─ default                   → call_claims_tool      │
│        │                                                │
│        ▼                                                │
│  call_overrides_tool_node  ◄────────────────────────────┘
│    Step 1: fetch_claim_list(claim_id) → CAGM            │
│    Step 2: POST /pss/myclaims/override/.../search       │
│    Cache:  set session:{sid}:api_cache:overrides:{uid}_{cid}
│        │                                                │
│        ▼                                                │
│  response_safety_pii_precheck_node                      │
│        │                                                │
│        ▼                                                │
│  response_agent_node                                    │
│    • domain_mapping == "override_domain"                │
│    • calls _get_overrides_domain_prompt()               │
│    • Gemini LLM → JSON envelope + optional HTML table   │
│        │                                                │
│        ▼                                                │
│  response_safety_pii_postcheck_node                     │
│        │                                                │
│        ▼                                                │
│  update_memory_node → cache_response_node               │
│        │                                                │
│        ▼                                                │
    Final Response to User
```

### Domain Placement in Multi-Domain Classifier

```
User Query
    │
    ▼
PCA (512-dim → 128-dim)
    │
    ▼
4-Classifier Ensemble (SVM-RBF, LogReg, kNN, ExtraTrees)
    │
    ├── benefits_api        (existing)
    ├── claim_history_search (existing)
    ├── cap_api             (existing)
    ├── general             (existing)
    ├── member_domain       (existing)
    └── override_domain     (ALREADY IN intents_mapping.py — 16 intents)
```

### Two-Step API Flow (Override-Specific)

```
claim_id ("12345")
    │
    ▼
Step 1: Claims List API (REUSED — Claims_search_api.api_utils.fetch_claim_list)
    │   POST claim_list_api_url  {claimId: "12345"}
    │   Response: claims[0].member.{carrierId, accountId, groupId, memberId}
    ▼
_extract_cagm() → {carrierId, accountId, groupId, memberId, personCode}
    │
    ▼
Step 2: Overrides API (NEW)
    │   POST /pss/myclaims/override/exp/v1/priorauth/search
    │   Body: {idSource:"6003", carrier, account, group, memberId, orderBy:4, enableFollowMeLogic:true}
    ▼
PA Records → response_trimmer → LLM → HTML Table / Text
```

---

## 2. New Files to Create

### 2.1 `Overrides_api/__init__.py`

**Purpose:** Marks the directory as a Python package; exports the public API surface.

```python
# Overrides_api/__init__.py
from .api_client import get_member_cagm_from_claim, get_overrides_by_member
from .overrides_node import call_overrides_tool_node

__all__ = [
    "get_member_cagm_from_claim",
    "get_overrides_by_member",
    "call_overrides_tool_node",
]
```

---

### 2.2 `Overrides_api/api_client.py`

**Purpose:** Adapted copy of the reference `override_api_client.py` (root workspace), placed in the correct package path. No functional changes — only the import path for `Claims_search_api.api_utils.fetch_claim_list` and the fallback file reference change.

**Key functions (unchanged from reference):**
- `_build_headers(bearer_token, x_api_key, x_clientrefid)` — mirrors `Claims_search_api/api_utils._build_headers`
- `_extract_cagm(list_response)` — extracts `carrierId/accountId/groupId/memberId/personCode` from multiple known nesting paths
- `get_member_cagm_from_claim(claim_number, ...)` — Step 1 wrapper around `fetch_claim_list`
- `get_overrides_by_member(cagm, ...)` — Step 2: POST to Overrides API with 5xx→fallback logic
- `_build_overrides_payload(cagm)` — builds POST body with `idSource`, `orderBy`, `enableFollowMeLogic`
- `_load_fallback()` — reads `config/overriders.json` on server error when `settings.enable_api_fallback=True`

**Constants to add to `config/config.py` Settings:**
```python
overrides_api_base_url: str = "https://internal-sit1-apix.cvshealth.com"
overrides_api_path: str = "/pss/myclaims/override/exp/v1/priorauth/search"
overrides_id_source: str = "6003"
overrides_order_by: int = 4
overrides_enable_follow_me_logic: bool = True
overrides_api_cache_ttl_seconds: int = 300
```

**Fallback file reference:** `Path(__file__).parent.parent / "config" / "overriders.json"`

---

### 2.3 `Overrides_api/response_trimmer.py`

**Purpose:** Mirrors `Claims_search_api/llm_query_responder.py`'s `_CLAIM_FIELD_WHITELIST` pattern. Trims raw PA API response to only business-relevant fields before passing to the LLM, reducing token usage and preventing PII bleed.

**Field whitelist (PA-specific):**

```python
_PA_FIELD_WHITELIST = {
    "priorAuthorization": [
        "paReferenceNumber",
        "paStatusCode",
        "paStatusDescription",
        "drugName",
        "ndc",
        "effectiveDate",
        "terminationDate",
        "quantityAllowed",
        "daysSupplyAllowed",
        "approvedRefillCount",
        "rejectReasonCode",
        "rejectReasonDescription",
        "overrideCode",
        "overrideDescription",
        "agentCode",
        "ignoreStatus",
        "drugTypeIndicator",
        "transformCarePlanIndicator",
        "followMeIndicator",
        "clinicalAdminCode",
        "modificationHistory",         # list of {modDate, modByAgent, modReasonCode}
        "specialtyRxOverrideIndicator",
        "copayAmount",
        "pricingInfo",                 # sub-object
    ]
}

def trim_pa_response(raw_response: dict, max_records: int = 20) -> list[dict]:
    """
    Extract and whitelist PA records from the raw Overrides API response.
    Caps at max_records to prevent context overflow.
    """
    records = (raw_response or {}).get("priorAuthorizations") or []
    trimmed = []
    for rec in records[:max_records]:
        slim = {}
        for field in _PA_FIELD_WHITELIST["priorAuthorization"]:
            val = rec.get(field)
            if val is not None:
                slim[field] = val
        trimmed.append(slim)
    return trimmed
```

**Critical design note:** The `modificationHistory` and `pricingInfo` sub-objects should be included as-is (not recursively whitelisted) in v1. If LLM context becomes a concern in production, add recursive whitelisting in v2.

---

### 2.4 `Overrides_api/llm_query_responder.py`

**Purpose:** PA-specific LLM prompt builder + answer generator. Mirrors the structure of `Claims_search_api/llm_query_responder.py` exactly — same function signatures, same Render DSL contract, PA-specific system instructions and field knowledge.

**Functions:**
- `prepare_pa_data(api_response, max_records)` — calls `trim_pa_response()`, sorts by `effectiveDate` desc
- `build_pa_prompt(user_query, slim_records, member_summary, rendering_disabled)` → str
- `answer_pa_query(api_response, user_query, member_summary, rendering_disabled, session_id)` → str

**System Instructions excerpt (to be placed in `_SYSTEM_INSTRUCTIONS`):**

```
You are a pharmacy benefits assistant specializing in Prior Authorization (PA) records.
PA records represent approved, rejected, or pending overrides for prescription drug coverage.

Key PA field knowledge:
- paStatusCode / paStatusDescription: current PA status (Approved, Rejected, Pending, Cancelled)
- effectiveDate / terminationDate: validity window for the PA
- rejectReasonCode / rejectReasonDescription: why the PA was denied (if rejected)
- overrideCode / overrideDescription: what override was applied (if approved)
- agentCode: pharmacist/agent who processed the PA
- ignoreStatus: whether the PA is being ignored by the adjudication system
- followMeIndicator: whether this PA follows the member across carrier/group changes
- specialtyRxOverrideIndicator: PA applies to a specialty drug
- modificationHistory: audit trail of changes to this PA record
- drugTypeIndicator: formulary drug type classification
- transformCarePlanIndicator: associated Transform Care program

Always clarify: a rejected PA does not mean the drug is unavailable — alternatives may exist.
Never recommend specific medications or clinical pathways. Direct clinical questions to pharmacist.
```

**Render DSL Contract:** Identical to `Claims_search_api/llm_query_responder.py`. LLM outputs:
```json
{"render_mode": "table", "summary": "Found 3 PA records for member."}
===RENDER_START===
<table>...</table>
===RENDER_END===
```

When `rendering_disabled=True`, append `_DISABLED_RENDERING_OVERRIDE` (imported from `response_agent.py`) and set `render_mode: "text_only"`.

---

### 2.5 `Overrides_api/overrides_node.py`

**Purpose:** LangGraph node that orchestrates the full two-step Overrides API call, caches the result, and writes `tool_results` + `render_dsl` to state.

**Signature:** `async def call_overrides_tool_node(state: AgentState) -> dict`

**Logic:**

```python
async def call_overrides_tool_node(state: AgentState) -> dict:
    claim_id = (state.get("entities") or {}).get("claim_id", "").strip()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    bearer_token = state.get("bearer_token", "")
    x_api_key = state.get("x_api_key", "")
    x_clientrefid = state.get("x_clientrefid", "")

    # 1. Check API-level Redis cache
    cache_key = _build_overrides_cache_key(session_id, user_id, claim_id)
    cached = get_cached_response(cache_key)
    if cached:
        logger.info(f"[OverridesNode] Cache HIT: {cache_key}")
        return {"tool_results": cached, "domain_mapping": "override_domain"}

    # 2. Step 1 — resolve CAGM
    cagm = await get_member_cagm_from_claim(claim_id, bearer_token, x_api_key, x_clientrefid)
    if not cagm:
        return {
            "tool_results": {"error": "Could not resolve member from claim_id", "claim_id": claim_id},
            "domain_mapping": "override_domain",
        }

    # 3. Step 2 — fetch PA records
    pa_response = await get_overrides_by_member(cagm, bearer_token, x_api_key, x_clientrefid)

    # 4. Cache the raw API response
    ttl = getattr(settings, "overrides_api_cache_ttl_seconds", 300)
    set_cached_response(cache_key, pa_response, ttl_seconds=ttl)

    return {
        "tool_results": {
            "data": pa_response,
            "cagm": cagm,
            "claim_id": claim_id,
            "domain": "override_domain",
        },
        "domain_mapping": "override_domain",
    }
```

**Cache key helper (private to this module):**
```python
def _build_overrides_cache_key(session_id: str, user_id: str, claim_id: str) -> str:
    return f"session:{session_id}:api_cache:overrides:{user_id}_{claim_id}"
```

---

### 2.6 `agents/post_processing/overrides_rendering_config.py`

**Purpose:** PA-specific rendering configuration for `myclaims_rendering_agent.py`. Registers field aliases, status code label maps, and null-handling rules for PA records.

The `domain_configs.py` file already has a `# overrides_rendering_config.py -> FUTURE` comment — this file fulfills that placeholder.

**Contents:**

```python
# Field aliases: API field name → human-readable label
FIELD_ALIASES = {
    "paReferenceNumber":          "PA Reference #",
    "paStatusCode":               "Status Code",
    "paStatusDescription":        "Status",
    "drugName":                   "Drug",
    "ndc":                        "NDC",
    "effectiveDate":              "Effective Date",
    "terminationDate":            "End Date",
    "quantityAllowed":            "Qty Allowed",
    "daysSupplyAllowed":          "Days Supply",
    "approvedRefillCount":        "Refills Approved",
    "rejectReasonCode":           "Reject Code",
    "rejectReasonDescription":    "Reject Reason",
    "overrideCode":               "Override Code",
    "overrideDescription":        "Override Type",
    "agentCode":                  "Agent",
    "ignoreStatus":               "Ignore Status",
    "followMeIndicator":          "Follow-Me",
    "specialtyRxOverrideIndicator": "Specialty Rx",
    "drugTypeIndicator":          "Drug Type",
    "clinicalAdminCode":          "Clinical Admin Code",
    "transformCarePlanIndicator": "Transform Care",
    "copayAmount":                "Copay",
}

# Status codes that render as "approved" (green indicator in UI)
APPROVED_STATUS_CODES = {"A", "AP", "APPROVED"}
REJECTED_STATUS_CODES = {"R", "RJ", "REJECTED", "DENY", "DENIED"}

# Fields that should render as boolean Yes/No
BOOLEAN_FIELDS = {
    "ignoreStatus",
    "followMeIndicator",
    "specialtyRxOverrideIndicator",
    "transformCarePlanIndicator",
}

# Fields excluded from table rendering (too verbose or redundant)
EXCLUDED_FROM_TABLE = {"modificationHistory", "pricingInfo"}

# Default table column order
TABLE_COLUMN_ORDER = [
    "paReferenceNumber",
    "drugName",
    "ndc",
    "paStatusDescription",
    "effectiveDate",
    "terminationDate",
    "quantityAllowed",
    "daysSupplyAllowed",
    "rejectReasonDescription",
    "overrideDescription",
    "agentCode",
]
```

---

### 2.7 `config/overriders.json`

**Purpose:** Static fallback PA data used when Overrides API returns a 5xx or times out and `settings.enable_api_fallback=True`. Mirrors the pattern of any existing fallback JSON in `config/`.

**Minimum structure required:**

```json
{
  "priorAuthorizations": [
    {
      "paReferenceNumber": "FALLBACK-001",
      "paStatusCode": "A",
      "paStatusDescription": "Approved (Fallback Data)",
      "drugName": "Sample Drug",
      "ndc": "00000-0000-00",
      "effectiveDate": "2025-01-01",
      "terminationDate": "2025-12-31",
      "quantityAllowed": 30,
      "daysSupplyAllowed": 30,
      "agentCode": "SYSTEM",
      "ignoreStatus": false,
      "followMeIndicator": true
    }
  ],
  "_fallback_note": "Static fallback — Overrides API unavailable"
}
```

---

## 3. Existing Files to Modify

### 3.1 `state/schema.py`

**Change:** Add `domain_mapping` field to `AgentState` TypedDict.

**Why a separate field from `domain`:** The existing `domain` field is set by the multidomain classifier and represents the classifier's predicted domain. `domain_mapping` is an explicit routing directive written by `extended_intent_agent_node` (via `api_routing_config`) and by `call_overrides_tool_node`. Downstream nodes (`response_agent`, `_route_after_build_context`) consume `domain_mapping` for routing decisions without overloading the classifier's output.

**Diff:**

```python
# Before (in AgentState TypedDict):
domain: Optional[str]

# After (add immediately after `domain`):
domain: Optional[str]
domain_mapping: Optional[str]   # explicit routing directive; set by intent_agent and tool nodes
```

**Also update `create_initial_state()`:**

```python
# Add to the returned dict:
"domain_mapping": None,
```

---

### 3.2 `config/api_routing_config.py`

**Change:** Register all 16 `override_domain` intents in `INTENT_API_ROUTING`.

**Pattern to follow:** Mirror how `_CLAIM_HISTORY_INTENTS` is registered in a loop. Create a `_OVERRIDE_INTENTS` list and loop over it.

**New block to add (after the CHS loop):**

```python
_OVERRIDE_INTENTS = [
    "pa_summary",
    "pa_override_reject",
    "pa_field_help",
    "pa_copay_pricing",
    "pa_drug_coverage",
    "pa_claim_usage",
    "pa_reason_code",
    "pa_effective_dates",
    "pa_agent_code",
    "pa_ignore_status",
    "pa_specialty_rx_override",
    "pa_clinical_admin_code",
    "pa_transform_care",
    "pa_follow_me_logic",
    "pa_drug_type_indicator",
    "pa_modification_history",
]

for _intent in _OVERRIDE_INTENTS:
    INTENT_API_ROUTING[_intent] = {
        "api_endpoint":          "/pss/myclaims/override/exp/v1/priorauth/search",
        "required_entities":     ["claim_id"],
        "domain":                "override_domain",
        "requires_llm":          True,
        "tool_name":             "overrides_search",
    }
```

**Also add helper function (mirrors `is_claim_history_search_intent`):**

```python
_OVERRIDE_INTENT_SET = frozenset(_OVERRIDE_INTENTS)

def is_override_domain_intent(intent: str) -> bool:
    return intent in _OVERRIDE_INTENT_SET

def get_domain_for_intent(intent: str) -> Optional[str]:
    cfg = INTENT_API_ROUTING.get(intent)
    return cfg.get("domain") if cfg else None
```

---

### 3.3 `langgraph_agent.py`

**Changes:**

**A. Import the new node:**
```python
from Overrides_api.overrides_node import call_overrides_tool_node
```

**B. Register the node:**
```python
builder.add_node("call_overrides_tool", call_overrides_tool_node)
```

**C. Update `_route_after_build_context` routing function:**

The existing routing checks `is_claims_search_query(state)`. Add the override domain branch before the default fallthrough:

```python
def _route_after_build_context(state: AgentState) -> str:
    if is_claims_search_query(state):
        return "call_claims_search"
    if is_override_domain_query(state):          # NEW
        return "call_overrides_tool"             # NEW
    return "call_claims_tool"
```

**D. Add conditional edge from `build_context` to `call_overrides_tool`:**
```python
builder.add_conditional_edges(
    "build_context",
    _route_after_build_context,
    {
        "call_claims_search":  "call_claims_search",
        "call_overrides_tool": "call_overrides_tool",   # NEW
        "call_claims_tool":    "call_claims_tool",
    }
)
```

**E. Add edges FROM `call_overrides_tool` (mirrors `call_claims_search` edges):**
```python
builder.add_edge("call_overrides_tool", "response_safety_pii_precheck")
```

The rest of the path (`response_safety_pii_precheck` → `response_agent` → `response_safety_pii_postcheck` → `update_memory` → `cache_response`) is already wired and requires no changes.

---

### 3.4 `Claims_search_api/intent_router.py`

**Change:** Add `is_override_domain_query(state)` function. This function is what `langgraph_agent.py`'s `_route_after_build_context` calls.

```python
from config.api_routing_config import is_override_domain_intent

def is_override_domain_query(state: AgentState) -> bool:
    """Return True if the current state belongs to the override domain."""
    # Primary signal: explicit domain_mapping set by intent_agent
    if state.get("domain_mapping") == "override_domain":
        return True
    # Secondary signal: multidomain classifier domain field
    if state.get("domain") == "override_domain":
        return True
    # Tertiary signal: intent is in override routing config
    intent = state.get("intent", "")
    if intent and is_override_domain_intent(intent):
        return True
    return False
```

**Rationale for triple check:** The `domain_mapping` signal is the most reliable (set programmatically from routing config). `domain` from the classifier may differ in edge cases. The intent check is a tertiary safety net.

---

### 3.5 `agents/post_processing/domain_configs.py`

**Change:** Register `overrides_rendering_config` in `_REGISTRY`.

```python
# Add import (line 24 area, after existing imports):
from agents.post_processing import overrides_rendering_config as _overrides_cfg

# Add to _REGISTRY dict:
_REGISTRY = {
    "claims":               _claims_cfg,
    "claim_history_search": _chs_cfg,
    "override_domain":      _overrides_cfg,    # NEW
}
```

**No changes needed to `get_config()` or `resolve_domain()`** — `resolve_domain()` already uses `state_domain` as priority-1, and `get_config()` already does `_REGISTRY.get(domain, _claims_cfg)` as fallback.

---

### 3.6 `agents/response_agent.py`

**Changes:**

**A. Import LLM responder:**
```python
from Overrides_api.llm_query_responder import answer_pa_query
```

**B. Add overrides domain prompt method to `ResponseAgent`:**

```python
def _get_overrides_domain_prompt(self) -> str:
    """System prompt addition for PA / override domain queries."""
    return """
You are answering questions about Prior Authorization (PA) records.
PA records control whether specific prescription drugs are covered for a member.

Key concepts:
- A PA may be Approved, Rejected, Pending, or Cancelled.
- A rejected PA does not mean the drug is unavailable — alternatives may exist.
- overrideCode / overrideDescription explain what exception was granted.
- rejectReasonCode / rejectReasonDescription explain why coverage was denied.
- effectiveDate and terminationDate define the validity window.
- agentCode identifies the pharmacist or agent who processed the PA.
- followMeIndicator means this PA moves with the member across benefit changes.

Response rules:
- Do NOT recommend specific drugs or clinical pathways.
- Direct clinical or formulary questions to a licensed pharmacist.
- When multiple PA records exist, present them as a table (if rendering is enabled).
- Always state whether the PA status is active (within effective/termination dates).
"""
```

**C. Update the domain routing logic inside the main response generation method:**

The exact line depends on how `response_agent.py` currently selects its prompt. Based on the CHS pattern (which calls `build_claim_history_prompt` when domain is `claim_history_search`), add an analogous branch:

```python
# In the response generation method, add before the CHS branch or as an elif:
if domain_mapping == "override_domain" or domain == "override_domain":
    # Use PA-specific LLM responder
    pa_api_response = (tool_results or {}).get("data", {})
    member_summary = self._build_member_summary(state)
    rendering_disabled = state.get("rendering_disabled", False)
    return await answer_pa_query(
        api_response=pa_api_response,
        user_query=user_query,
        member_summary=member_summary,
        rendering_disabled=rendering_disabled,
        session_id=state.get("session_id", ""),
    )
```

**D. Add `"override_domain"` to blocked recommendation actions if needed:**

The existing `BLOCKED_RECOMMENDATION_ACTIONS` frozenset blocks certain tool names from generating next-step recommendations. Evaluate whether `"overrides_search"` should be added — likely yes, since PA data should not generate drug recommendations.

```python
BLOCKED_RECOMMENDATION_ACTIONS = frozenset({"claim_list", "overrides_search"})
```

---

### 3.7 `agents/extended_intent_agent_node.py`

**Change:** After `get_api_config(intent)` resolves the routing config, explicitly set `domain_mapping` in the returned state update.

```python
# After: api_config = get_api_config(intent)
domain_from_config = api_config.get("domain") if api_config else None

return {
    ...existing fields...,
    "domain_mapping": domain_from_config,   # NEW — explicit routing directive
}
```

This ensures `domain_mapping` is set at intent detection time (not only after the tool node runs), allowing `_route_after_build_context` to use it reliably.

---

## 4. State Schema Additions

### New Field

| Field | Type | Set By | Consumed By | Purpose |
|-------|------|--------|-------------|---------|
| `domain_mapping` | `Optional[str]` | `extended_intent_agent_node`, `call_overrides_tool_node` | `_route_after_build_context`, `response_agent`, `is_override_domain_query` | Explicit routing directive. Decouples classifier output (`domain`) from downstream routing decisions. |

### Schema Diff (`state/schema.py`)

```python
class AgentState(TypedDict):
    # ... existing fields ...
    domain: Optional[str]           # set by multidomain classifier
    domain_mapping: Optional[str]   # NEW: explicit routing directive from api_routing_config
    # ... rest of fields ...
```

### `create_initial_state()` addition

```python
def create_initial_state() -> AgentState:
    return {
        # ... existing fields ...
        "domain_mapping": None,   # NEW
    }
```

### Why Not Reuse `domain`?

The `domain` field is populated by the multidomain ensemble classifier and represents a probabilistic prediction. It can be `None` if confidence is too low, or can be overridden by classifier confusion. `domain_mapping` is a deterministic value derived from the explicit `INTENT_API_ROUTING` config. Having two separate fields preserves the classifier output for logging/analytics while giving routing logic a clean signal to act on.

---

## 5. Intent Definitions and Example Utterances

### Status: Already in Codebase

All 16 `override_domain` intents and their training examples **already exist** in `pss-myclaims-ai-agent_Saksham/multidomain_intent_detection/intents_mapping.py`. The `override_domain` entry in `DOMAIN_REGISTRY` is also already present.

**No changes required to `intents_mapping.py` or the classifier pipeline.**

### Existing Intents (from `intents_mapping.py`)

| Intent | Description |
|--------|-------------|
| `pa_summary` | General PA record summary / list of all PAs for member |
| `pa_override_reject` | Why was a PA rejected; override codes |
| `pa_field_help` | Explain a specific PA field (e.g., "what is followMe?") |
| `pa_copay_pricing` | Copay amounts or pricing info tied to a PA |
| `pa_drug_coverage` | Is a specific drug covered under this PA |
| `pa_claim_usage` | How many refills/quantity has been used under this PA |
| `pa_reason_code` | Reject reason codes and their meaning |
| `pa_effective_dates` | Effective and termination dates of a PA |
| `pa_agent_code` | Who processed this PA (agent code) |
| `pa_ignore_status` | Is this PA being ignored by the adjudication system |
| `pa_specialty_rx_override` | PA for specialty drugs |
| `pa_clinical_admin_code` | Clinical administration codes on the PA |
| `pa_transform_care` | Transform Care program PA indicators |
| `pa_follow_me_logic` | Follow-me logic: does the PA move with the member |
| `pa_drug_type_indicator` | Drug type indicator (formulary classification) |
| `pa_modification_history` | Audit trail of changes to the PA record |

### Confusion Pairs to Add (`multidomain_intent_detection/pipeline.py`)

The `CONFUSION_PAIRS` dict in `pipeline.py` defines known hard-to-distinguish intent pairs. The following pairs should be added for override domain:

```python
CONFUSION_PAIRS.update({
    # PA intents that are semantically close
    "pa_summary":         ["pa_drug_coverage", "pa_effective_dates"],
    "pa_override_reject": ["pa_reason_code", "pa_field_help"],
    "pa_copay_pricing":   ["pa_drug_coverage", "pa_claim_usage"],
    "pa_effective_dates": ["pa_summary", "pa_claim_usage"],
    "pa_follow_me_logic": ["pa_ignore_status", "pa_field_help"],
    "pa_modification_history": ["pa_summary", "pa_agent_code"],
})
```

### Required Entity: `claim_id`

Every override domain intent requires `claim_id` as a mandatory entity. The clarification flow is handled automatically by the existing `confidence_check_router`:

1. `extended_intent_agent_node` sets `required_entities_list = ["claim_id"]` (from `api_routing_config`)
2. `extract_entities_unified()` fills `entities["claim_id"]` if present in query
3. `confidence_check_router` computes `missing_slots = required_entities_list - entities.keys()`
4. If `missing_slots` is non-empty → routes to `clarification_node`
5. `clarification_node` generates: *"To look up Prior Authorization records, I need your claim number. Could you provide it?"*
6. User provides claim_id → re-enters intent detection → slots satisfied → proceeds to tool node

**No custom clarification logic is needed.** The generic slot-filling mechanism already handles this.

---

## 6. Redis Caching Design

### Cache Key Pattern

```
session:{session_id}:api_cache:overrides:{user_id}_{claim_id}
```

**Example:**
```
session:abc123:api_cache:overrides:user456_12345678
```

**Rationale for this pattern:**
- Matches the existing `session:{sid}:api_cache:{uid}_{claimNumber}_{sequenceNumber}` pattern in `tools/api_cache.py`
- `overrides:` namespace prefix prevents collision with claim history cache entries
- `claim_id` as the key discriminator: PA lookups are per-member (resolved via claim_id), so the claim_id uniquely identifies the CAGM lookup
- Not including `intent` in the key: all 16 PA intents fetch the same PA records set; the LLM differentiates the answer, not the API data

### Cache Scope

**API-level cache** (`tools/api_cache.py` pattern): Caches the raw Overrides API response. Located at the tool node level (`call_overrides_tool_node`). Separate from the semantic cache (`nodes/cache.py`) which caches full LLM answers.

Both caches are active:
- Semantic cache (check_cache_node) → checked first → returns full cached LLM response
- API cache (in overrides node) → checked second → returns raw PA data to feed the LLM

### TTL

```python
ttl_seconds = settings.overrides_api_cache_ttl_seconds  # default: 300 (5 minutes)
```

PA records change infrequently (PA approvals/rejections are batch-processed), so 5 minutes is appropriate. Can be increased to 600–900 seconds in production.

### Cache Helper

Reuse existing `tools/api_cache.py` functions:

```python
from tools.api_cache import get_cached_response, set_cached_response

# In call_overrides_tool_node:
cache_key = f"session:{session_id}:api_cache:overrides:{user_id}_{claim_id}"
cached = get_cached_response(cache_key)
if cached:
    return {"tool_results": cached, "domain_mapping": "override_domain"}

# ... fetch from API ...

set_cached_response(cache_key, pa_response, ttl_seconds=settings.overrides_api_cache_ttl_seconds)
```

**No new cache infrastructure is needed.** The existing Redis client and graceful degradation logic in `tools/api_cache.py` handles Redis failures transparently.

### Cache Invalidation

No explicit invalidation is needed. TTL-based expiry is sufficient. If a user reports stale PA data, support can clear the key pattern `session:*:api_cache:overrides:*` via Redis CLI.

---

## 7. Prompt Template Strategy

### Template Hierarchy

The overrides domain uses the same three-layer prompt architecture as `Claims_search_api/llm_query_responder.py`:

```
Layer 1: _SYSTEM_INSTRUCTIONS    — domain knowledge, behavioral rules, PII rules
Layer 2: _RENDER_DSL_CONTRACT    — how to produce render_mode + HTML table DSL
Layer 3: _USER_TEMPLATE          — runtime values: query, member summary, PA records JSON
```

### Layer 1: System Instructions (`_SYSTEM_INSTRUCTIONS`)

Full content for `Overrides_api/llm_query_responder.py`:

```
You are a pharmacy benefits assistant for CVS Health. You answer member questions
about Prior Authorization (PA) records retrieved from the CVS override system.

DOMAIN KNOWLEDGE:
- PA records represent approved, rejected, pending, or cancelled drug coverage exceptions.
- A member may have multiple PAs for different drugs or different time periods.
- paReferenceNumber uniquely identifies a PA record.
- effectiveDate and terminationDate define when the PA is valid.
- An "Approved" PA means the drug is covered under the override terms.
- A "Rejected" PA means coverage was denied; rejectReasonDescription explains why.
- overrideCode / overrideDescription describe what exception was granted for approved PAs.
- agentCode is a pharmacist or benefit agent identifier — do not expose as a name.
- ignoreStatus=true means adjudication is currently bypassing this PA.
- followMeIndicator=true means the PA applies even after plan/group transfers.
- specialtyRxOverrideIndicator=true means the PA covers a specialty drug.
- modificationHistory contains the audit trail of PA changes.

BEHAVIORAL RULES:
1. Never recommend specific drugs or clinical alternatives.
2. Never interpret clinical admin codes (clinicalAdminCode) without stating they
   require pharmacist review.
3. Direct questions about drug alternatives to a licensed pharmacist.
4. If multiple PA records are present, list them all; do not filter arbitrarily.
5. Always include the PA reference number when citing a specific record.
6. State clearly if a PA is expired (current date outside effective/termination window).
7. Treat copayAmount and pricingInfo as estimates — actual cost may differ.

PII / MASKED TOKEN RULES:
- If member data contains masked tokens (e.g. [MASKED]), reproduce them exactly.
- Do not attempt to reconstruct or guess masked values.
- Do not include member names or DOB in the response unless directly asked.
```

### Layer 2: Render DSL Contract (`_RENDER_DSL_CONTRACT`)

**Identical to `Claims_search_api/llm_query_responder.py`** — no changes needed:

```
RESPONSE FORMAT CONTRACT:
Your response MUST start with a valid JSON object on the FIRST line:
{"render_mode": "<text_only|table>", "summary": "<one-sentence summary>"}

If render_mode is "table", immediately follow with:
===RENDER_START===
<table>
  <thead>...</thead>
  <tbody>...</tbody>
</table>
===RENDER_END===

Rules:
- Use "table" only when there are multiple PA records and rendering is enabled.
- Use "text_only" for single-record answers, field explanations, or when rendering is disabled.
- The JSON line must be valid — no trailing text before the newline.
- Do not include markdown fences inside the RENDER block.
```

### Layer 3: User Template (`_USER_TEMPLATE`)

```python
_USER_TEMPLATE = """
Current date: {current_date}
Member: {member_summary}

User question: {user_query}

Prior Authorization records (JSON):
{pa_records_json}

Answer the user's question based on the PA records above.
"""
```

### Rendering Disabled Path

When `rendering_disabled=True` (mobile client or explicit override), append:

```python
_DISABLED_RENDERING_OVERRIDE = """
IMPORTANT: Rendering is DISABLED for this session.
- Set render_mode to "text_only" in your JSON envelope.
- Do NOT include ===RENDER_START=== blocks.
- Present all data as plain prose or a simple text list.
"""
```

Import this constant from `response_agent.py` to avoid duplication:

```python
from agents.response_agent import _DISABLED_RENDERING_OVERRIDE
```

### Table Column Strategy

For multi-PA responses, the LLM is instructed to produce an HTML `<table>` with these columns (in order), using the aliases from `overrides_rendering_config.py`:

```
PA Reference # | Drug | Status | Effective Date | End Date | Qty Allowed | Days Supply | Override Type / Reject Reason
```

The `rejectReasonDescription` and `overrideDescription` are mutually exclusive in practice — render whichever is non-null under a combined "Decision" column to reduce table width.

### Member Summary Construction

Reuse the existing `_build_member_summary` pattern from `response_agent.py`. The CAGM dict returned by `_extract_cagm()` in `call_overrides_tool_node` is stored in `tool_results["cagm"]` and can be used to construct:

```
Member: [masked if PII], Plan: {carrierId}/{accountId}/{groupId}
```

---

## 8. Error and Fallback Handling

### Error Taxonomy and Responses

| Error Condition | Detection | Response Strategy |
|----------------|-----------|-------------------|
| `claim_id` missing from query | `confidence_check_router` detects `missing_slots` | Clarification node asks for claim number |
| Step 1 failure (claim list API) | `get_member_cagm_from_claim` returns `None` | Return error tool_result → LLM generates "Could not resolve member from that claim number" |
| Step 2 5xx / timeout | `httpx.TimeoutException` or `status >= 500` | If `enable_api_fallback=True` → load `config/overriders.json`; else re-raise |
| Step 2 4xx (not found / unauthorized) | `httpx.HTTPStatusError` with 4xx status | Re-raise → propagates as tool error → LLM generates generic API error message |
| CAGM extraction failure (all paths miss) | `_extract_cagm()` returns `None` | Same as Step 1 failure |
| Redis cache failure | Caught in `get_cached_response` | Graceful degradation: proceed without cache (existing behavior in `tools/api_cache.py`) |
| LLM response malformed (bad JSON envelope) | `response_agent.py` JSON parse error | Existing fallback: return raw LLM text as `text_only` response |
| LLM timeout / Gemini error | Exception in `answer_pa_query` | Return user-facing: "Unable to process PA query. Please try again." |

### Fallback Data Design

The `config/overriders.json` fallback file is used only for Step 2 failures. It returns a clearly marked sample PA record to allow the UI to remain functional during Overrides API outages. The LLM will detect the `_fallback_note` field and must acknowledge uncertainty:

Add to `_SYSTEM_INSTRUCTIONS`:
```
If the PA data contains a "_fallback_note" field, acknowledge that the data may
not reflect the member's actual PA records and advise the user to call member services.
```

### LLM Fallback vs. API Fallback Distinction

- **API fallback** (`config/overriders.json`): activates on Step 2 5xx/timeout; preserves the LLM answer flow with mock data
- **LLM fallback** (generic error message): activates when the LLM itself fails; bypasses the LLM entirely; returns a hardcoded error string

Both paths are handled within `answer_pa_query()`:

```python
async def answer_pa_query(api_response, user_query, ...):
    try:
        slim = prepare_pa_data(api_response)
        prompt = build_pa_prompt(user_query, slim, ...)
        result = await generate(prompt)           # Gemini call
        return result
    except LLMError as e:
        logger.error(f"[OverridesLLM] LLM error: {e}")
        return '{"render_mode": "text_only", "summary": "Unable to process PA query."}\nI was unable to retrieve your Prior Authorization information at this time. Please try again or contact member services.'
    except Exception as e:
        logger.error(f"[OverridesLLM] Unexpected error: {e}")
        return '{"render_mode": "text_only", "summary": "An error occurred."}\nAn unexpected error occurred. Please try again.'
```

### Clarification Flow (Missing `claim_id`)

```
User: "What are my prior authorizations?"
    │
    ▼
extended_intent_agent_node
    intent = "pa_summary"
    entities = {}                          ← claim_id not in query
    required_entities_list = ["claim_id"]
    missing_slots = ["claim_id"]
    │
    ▼
confidence_check_router → "clarification"
    │
    ▼
clarification_node generates:
    "To look up your Prior Authorization records, I need your claim number.
     Could you please provide it?"
    │
    ▼
User: "My claim number is 12345"
    │
    ▼
Loops back → extended_intent_agent_node
    entities = {"claim_id": "12345"}
    missing_slots = []
    │
    ▼
Proceeds to call_overrides_tool_node
```

---

## 9. Open Questions and Risks

### Risk 1: `override_domain` API Endpoint Mismatch

**Observation:** `intents_mapping.py` `DOMAIN_REGISTRY` lists `api_endpoint: "/myclaims/overrides/v1/pa"` for `override_domain`. The actual Overrides API client (`override_api_client.py`) uses `/pss/myclaims/override/exp/v1/priorauth/search`.

**Impact:** Medium. If any routing logic reads `api_endpoint` from `DOMAIN_REGISTRY` directly (rather than from `INTENT_API_ROUTING`), it would construct the wrong URL.

**Mitigation:** `api_routing_config.py` is the authoritative source for `api_endpoint` (via `get_api_config(intent)`). Verify that no code path reads `api_endpoint` from `DOMAIN_REGISTRY.api_endpoint` for actual HTTP calls. Update `DOMAIN_REGISTRY["override_domain"]["api_endpoint"]` to match the real URL for consistency.

---

### Risk 2: `domain_mapping` Setting in `extended_intent_agent_node`

**Observation:** `domain_mapping` is set by the intent agent node from `api_routing_config.get("domain")`. But if `get_api_config(intent)` returns `None` (unregistered intent), `domain_mapping` would be `None`, and `is_override_domain_query()` would fall back to the `domain` field check. This is safe but means the routing has a two-signal dependency.

**Mitigation:** The triple-check in `is_override_domain_query()` (domain_mapping → domain → intent set lookup) provides defense in depth. Ensure all 16 PA intents are registered in `api_routing_config.py` before deployment.

---

### Risk 3: CAGM Extraction Reliability Across Environments

**Observation:** `_extract_cagm()` in `override_api_client.py` tries multiple nested paths for the member block. The `Claims_search_api/api_utils.py` also has two paths (`claimList[0].primary.beneficiary` and `claims[0].member`). These two functions are not fully aligned — `_extract_cagm` only checks `claims[0]` paths (not `claimList[0]`).

**Impact:** Medium. In environments where the claim list API returns `claimList` format (old), `_extract_cagm()` would return `None` and PA lookup would fail.

**Mitigation:** In `Overrides_api/api_client.py`, extend `_extract_cagm()` to also check the `claimList[0].primary.beneficiary` path, aligning it with `_extract_member_id_from_list()` in `api_utils.py`.

---

### Risk 4: `response_agent.py` Routing Logic Location

**Observation:** The exact location of the domain-routing `if/elif` block in `response_agent.py` was not fully readable (file is 340KB). The CHS branch (`build_claim_history_prompt`) must already exist. The override domain branch must be inserted before the default `cap_api`/`claims` branch.

**Action Required:** Before implementation, read `response_agent.py` lines 1–200 to locate the domain-dispatch block and verify the insertion point. Do not blindly append — incorrect ordering could cause CHS queries to be routed to the overrides LLM responder if `domain_mapping` is not set cleanly.

---

### Risk 5: Semantic Cache Key Collision

**Observation:** The semantic cache (`nodes/cache.py`) caches full LLM responses keyed by query + session context. If two PA intents ask semantically similar questions with the same claim_id (e.g., `pa_summary` and `pa_effective_dates`), the semantic cache might return one intent's answer for the other.

**Impact:** Low in practice (PA intents are domain-specific), but possible.

**Mitigation:** Verify that the semantic cache key includes `intent` or `domain_mapping`. If not, add `domain_mapping` to the cache key computation in `nodes/cache.py`.

---

### Risk 6: PII in PA Response

**Observation:** PA records may contain `memberId`, `carrierId`, and `agentCode`. The `response_safety_pii_precheck_node` runs before the LLM call and the `response_safety_pii_postcheck_node` runs after. The PA response trimmer (`response_trimmer.py`) should exclude `memberId` from the LLM context since it is PII.

**Mitigation:** Add `memberId` to the exclusion list in `trim_pa_response()`. The CAGM identifiers are used only for the API call and should not flow into the LLM prompt.

---

### Risk 7: Empty PA Records Response

**Observation:** If a member has no PA records, the Overrides API may return `{"priorAuthorizations": []}`. The LLM should handle this gracefully.

**Mitigation:** In `prepare_pa_data()`, detect empty records and return a sentinel:

```python
if not records:
    return []  # empty list

# In build_pa_prompt, add to user template:
if not slim_records:
    pa_records_json = '[]  # No PA records found for this member.'
```

The LLM should then respond: *"No Prior Authorization records were found for this claim."*

---

### Open Questions

1. **Personcode in CAGM payload:** The `_build_overrides_payload()` does not include `personCode` in the POST body, but `_extract_cagm()` extracts it. Should `personCode` be included in the Overrides API payload? Verify with API documentation.

2. **Multi-claim PA lookup:** The current design resolves one CAGM per `claim_id`. If a member has multiple claim IDs, should the agent aggregate PA records across them? The v1 approach (one claim_id → one CAGM → one PA lookup) is simpler and correct for the stated requirement.

3. **PA record volume cap:** `response_trimmer.py` caps at 20 PA records. Is 20 the right number? PA records can accumulate over years. Verify with product team.

4. **`overrides_api_base_url` per environment:** The base URL `https://internal-sit1-apix.cvshealth.com` is the SIT1 environment. Confirm the prod/staging URL and ensure `settings.overrides_api_base_url` is environment-configurable via `.env`.

5. **Confusion pair weights:** Adding confusion pairs to `pipeline.py` affects the ensemble scoring for all domains. Run classifier evaluation on the full test set after adding the pairs to ensure no regression in CHS or CAP intent accuracy.

---

## Implementation Checklist

### Phase 1 — Infrastructure (No Behavioral Change)
- [ ] Create `Overrides_api/` directory
- [ ] Create `Overrides_api/__init__.py`
- [ ] Create `Overrides_api/api_client.py` (adapted from `override_api_client.py`)
- [ ] Create `config/overriders.json` (fallback data)
- [ ] Add Settings fields to `config/config.py`
- [ ] Add `domain_mapping: Optional[str]` to `state/schema.py`
- [ ] Add `domain_mapping: None` to `create_initial_state()`

### Phase 2 — Routing
- [ ] Add 16 PA intents to `config/api_routing_config.py`
- [ ] Add `is_override_domain_intent()` and `get_domain_for_intent()` helpers
- [ ] Add `is_override_domain_query()` to `Claims_search_api/intent_router.py`
- [ ] Update `extended_intent_agent_node.py` to set `domain_mapping`

### Phase 3 — Tool Node
- [ ] Create `Overrides_api/response_trimmer.py`
- [ ] Create `Overrides_api/overrides_node.py`
- [ ] Register node in `langgraph_agent.py`
- [ ] Add routing branch in `_route_after_build_context()`
- [ ] Wire edges to/from `call_overrides_tool`

### Phase 4 — LLM Integration
- [ ] Create `Overrides_api/llm_query_responder.py`
- [ ] Create `agents/post_processing/overrides_rendering_config.py`
- [ ] Register in `agents/post_processing/domain_configs.py`
- [ ] Add `_get_overrides_domain_prompt()` to `response_agent.py`
- [ ] Add overrides domain routing in `response_agent.py`

### Phase 5 — Testing and Validation
- [ ] Unit test `_extract_cagm()` against both API response shapes
- [ ] Integration test: happy path with `claim_id` → PA records → HTML table
- [ ] Test clarification flow: query without `claim_id`
- [ ] Test API fallback: mock Step 2 5xx → verify `overriders.json` data returned
- [ ] Test LLM fallback: mock Gemini error → verify error string returned
- [ ] Test empty PA records: `{"priorAuthorizations": []}` → verify graceful message
- [ ] Run classifier evaluation to confirm no intent regression
- [ ] Verify Redis cache hit/miss logging in `call_overrides_tool_node`
