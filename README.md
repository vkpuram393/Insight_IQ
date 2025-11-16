# PBM AI Assist – LangGraph Multi‑Agent Starter

> A pragmatic starter project for building a pharmacy benefit (PBM) conversational assistant using **LangGraph**. This is NOT production code – it’s an accelerator with **mock LLMs** and **mock APIs** so you can explore flow, state, routing, memory, and clarification logic without burning real tokens or hitting real backend systems.

---
## 1. What This Is / What It Isn’t
**Is:**
- A working multi‑agent + multi‑node LangGraph flow
- Intent → routing → clarification/tool → response
- In‑memory cache + short/long term conversation memory
- Checkpointing scaffold (SQLite async saver)
- Clean Python 3.11 project structure

**Is NOT:**
- Final architecture for production
- Secure / audited / load‑tested
- Real LLM integration (uses `MockLLM`)
- Real claims API integration (uses a dummy tool)

You’re expected to **replace the mocks** with your enterprise logic, LLM provider(s), observability, guardrails, and compliance controls.

---
## 2. Tech Stack & Versions
| Component | Version (pinned) | Notes |
|-----------|------------------|-------|
| Python | 3.11.x (tested on 3.11.9) | Required – 3.12 not yet verified |
| FastAPI | 0.115.0 | REST layer |
| Uvicorn | 0.24.0 | ASGI server |
| LangGraph | 0.2.45 | Graph orchestration |
| LangChain Core | 0.3.18 | Prompt/message abstractions |
| LangChain OpenAI | 0.2.8 | Only used if you flip off mocks |
| Pydantic | 2.9.2 | Settings + schemas |
| pydantic-settings | 2.5.2 | Environment configuration |
| redis / pymongo | Present in requirements | Future expansion; not used yet |
| aiosqlite | Pulled transitively | Needed for async SQLite checkpointing |

Check the exact list in `requirements.txt` for drift.

> If you try another Python version and something breaks, start by reverting to 3.11.

---
## 3. Clone & Install
```bash
# macOS / Linux
git clone https://github.com/cvs-health-source-code/PBM-AI-Assist.git
cd PBM-AI-Assist
python -V  # Expect Python 3.11.x
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
```powershell
# Windows (PowerShell) – recommend PowerShell 7+, or Git Bash for UNIX tools
git clone https://github.com/cvs-health-source-code/PBM-AI-Assist.git
cd PBM-AI-Assist
py -3.11 -V          # Verify Python 3.11
py -3.11 -m venv .venv
# Activate in PowerShell
.\.venv\Scripts\Activate.ps1
# (If execution policy blocks activation: Run PowerShell as admin)
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```
```cmd
REM Windows (CMD classic shell)
git clone https://github.com/cvs-health-source-code/PBM-AI-Assist.git
cd PBM-AI-Assist
py -3.11 -m venv .venv
\.venv\Scripts\activate
pip install -r requirements.txt
```
Optional upgrade (any platform):
```bash
pip install --upgrade pip
```
> Windows tip: If you lack `curl` and `jq`, install Git Bash or use PowerShell `Invoke-RestMethod` (examples below).

### 3.1 Windows-Specific Notes
| Topic | Windows Guidance |
|-------|------------------|
| Python Install | Use official 3.11 installer. Check "Add Python to PATH". Disable "PATH length limit" if prompted. |
| Virtual Env Activation | PowerShell requires `RemoteSigned` policy; CMD uses `\.venv\Scripts\activate`. |
| Line Endings | Git should auto-handle CRLF; keep `core.autocrlf=true`. |
| Performance | Long paths may need enabling (`git config core.longpaths true`). |
| WSL Option | For closer parity with Linux tools (make, bash, etc.), you can develop inside WSL and still use PyCharm Remote Interpreter. |
| SSL / Corporate Proxy | Configure `pip.ini` at `%APPDATA%\pip\pip.ini` with proxy/cert if needed. |

Sample PowerShell request (no jq):
```powershell
$body = @{ text = 'why was my claim rejected'; session_id = 'ps-1' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/api/v1/chat' -ContentType 'application/json' -Body $body
```
Sample Git Bash request (has curl/jq):
```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"text":"why was my claim rejected","session_id":"bash-1"}' | jq
```

---
## 4. Configuration
All runtime settings live in `core/config.py` via `Settings`. A `.env` file is optional.
Key flags:
- `use_mock_llm = True` → Uses `MockLLM` (no external calls)
- `enable_checkpointing` → Currently `False` for simplicity (async saver is wired but can be toggled)
- `confidence_threshold` → Router decision for clarification vs tool path
- `enable_semantic_cache` → In‑memory cache on or off

Set environment variables in `.env` if you later integrate a real LLM:
```
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=sk-...
```
Those are presently ignored when mocks are active.

---
## 5. Run the Server
```bash
# macOS / Linux
python main.py
```
```powershell
# Windows PowerShell
py -3.11 main.py
# or
python main.py
```
If you want uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
On Windows CMD/PowerShell this is the same; just ensure the virtual env is activated.

Health check (cross‑platform):
```bash
curl -s http://localhost:8000/health
```
PowerShell alternative:
```powershell
Invoke-RestMethod http://localhost:8000/health
```

---
## 6. Request Lifecycle (Mental Model)
```
User POST /chat
  → safety_precheck
  → check_cache (hit? return early)
  → build_context (pull memory)
  → intent_agent (MockLLM classifies + entities)
  → confidence_check_router
      ├─ clarification (missing data) → update_memory → cache_response → END
      └─ call_claims_tool (mock API) → response_agent → safety_postcheck → update_memory → cache_response → END
```
Nodes live in `nodes/`; agents in `agents/`; tool(s) in `tools/`.

---
## 7. Agents & Mocks
### Intent Agent (`agents/intent_agent.py`)
- Keyword heuristic classification in `MockLLM`
- Extracts `claim_number` via regex (4–10 digits preceded by the word "claim")
- Emits: `intent`, `confidence`, `entities`

### Response Agent (`agents/response_agent.py`)
- Uses `MockLLM` to select canned text based on intent
- TODO: Replace `#12345` hardcoded piece with real `tool_results['claim_id']`

### Claims Tool (`tools/claims_api.py`)
- Fakes latency (`asyncio.sleep(0.2)`) and returns deterministic JSON
- Replace this with real backend integration (REST / GraphQL / SOAP / gRPC) later

---
## 8. Memory & Cache
- Short‑term memory: last 10 messages stored in `_short_term` (in‑memory, per `session_id`)
- Long‑term facts: naïve claim mentions stored in `_long_term`
- Cache: Keyed by MD5 of lowercased text; stores response + intent + confidence
- Clarification now persists to memory (edge changed: `clarification → update_memory → cache_response`)

Production considerations:
- Move memory & cache to Redis or a vector store
- Add eviction, TTL, and multi‑tenant isolation
- Persist conversation state with LangGraph’s checkpointing (currently scaffolded)

---
## 9. Clarification Strategy
We chose **entity completeness** (missing `claim_number`) over artificial confidence reduction. If intent = `claim_rejection_reason` and no claim number → ask user: “Which claim are you asking about?”

To tune:
- Adjust regex for claim number formats (`CLM-1234`, etc.)
- Add slot-filling or structured entity extraction
- Add multi-turn clarifications (e.g., need member ID, date, etc.)

---
## 10. Debugging in PyCharm (Step‑Through Guide)
1. **Interpreter**:
   - macOS/Linux: Select `./.venv/bin/python`
   - Windows: Select `./.venv/Scripts/python.exe`
2. **Environment Variables**: On Windows, set in Run Configuration → Environment (instead of shell export).
3. **Line Breakpoints**: Same UI; watch for path separators in evaluated expressions.
4. **Async Debugging**: Enable "Asyncio" in Preferences (Windows: `File > Settings`).
5. **Port Conflicts**: Windows sometimes leaves a process bound after debug stop; use Task Manager or `Get-Process -Id <PID> | Stop-Process`. On UNIX: `lsof -ti :8000 | xargs kill`.
6. **Reload Issues**: If `--reload` causes thread errors on Windows, disable and rely on manual restarts.
7. **Curl vs PowerShell**: If `curl` is missing, use `Invoke-RestMethod` to test endpoints.
8. **Path Separators**: Code uses forward slashes but Python handles them; no change needed unless doing manual file ops.
9. **Clipboard JSON**: PowerShell may escape quotes; prefer single quotes around JSON and escape inner quotes.
10. **Watch Variables**: Add `state`, `intent`, `entities` in debug panel; evaluating dict keys identical across platforms.

Conditional breakpoint example (cross‑platform): set at the first line inside `confidence_check_router` (file: `nodes/confidence.py`) with condition:
```
state.get('intent') == 'claim_rejection_reason' and not state.get('entities', {}).get('claim_number')
```

---
## 11. Replacing Mocks with Real Code
| Area | Current | Replace With |
|------|---------|--------------|
| LLM (intent) | `MockLLM` keyword heuristics | Real provider (OpenAI, Azure, internal model endpoint) + robust prompt + output schema validation |
| LLM (response) | Canned responses | Retrieval‑augmented generation + guardrails (toxicity, PHI filters) |
| Claims tool | Hardcoded JSON | Secure internal API client (auth, retries, circuit breaker) |
| Memory | In‑memory dict | Redis / Postgres / vector DB embedding store |
| Checkpointing | Async SQLite | Managed durable storage (cloud DB, encrypted volume) |
| Safety | Simple pass | Policy engine (PII scrubbing, abuse detection, jailbreak prevention) |

---
## 12. Extending Intents
1. Add new intent phrase detection to `MockLLM` or real classifier.
2. Update system prompt examples.
3. Add new branch in `confidence_check_router` if special handling needed.
4. Add corresponding tool node or agent if required.
5. Write tests (later) for classification + routing.

---
## 13. Git Workflow
- `main` – Stable baseline
- `develop` – Integration branch for features
- Feature branches: `feature/<short-desc>` → PR into `develop`
- Periodic merges from `develop` → `main` after review & testing

Example:
```bash
git checkout -b feature/real-llm-integration
git commit -m "feat: integrate real LLM for intent"
git push origin feature/real-llm-integration
# Open PR → review → merge into develop
```

---
## 14. API Examples (Postman / curl)
**Postman (Windows & macOS)**:
1. New Request → POST `http://localhost:8000/api/v1/chat`
2. Headers: `Content-Type: application/json`
3. Body (raw JSON):
```json
{
  "text": "why was my claim rejected",
  "session_id": "postman-1"
}
```
4. Send → Expect `needs_clarification: true`
5. Follow-up body:
```json
{
  "text": "Claim 987654 was rejected. Why?",
  "session_id": "postman-1"
}
```

*Tip:* Use Postman "Examples" or "Collections" to share typical flows with teammates.

Existing bash examples remain valid on macOS/Linux; substitute PowerShell commands where needed for Windows.

---
## 15. Observability & Logging

### Current Logging
- **Stdout logging**: Simple timestamps and messages to console
- **SQLite audit logs**: All state transitions, decisions, and API calls logged to `data/telemetry.db`
- **SQLite exceptions**: All exceptions logged to `data/telemetry.db` with full stack traces

### Database Schema

The application uses SQLite (`data/telemetry.db`) with three main tables:

1. **`logs`** - Audit trail of all operations
   - `log_id`, `session_id`, `request_id`, `node_name`, `event_type`, `data`, `timestamp`, `user_id`

2. **`exceptions`** - All exceptions and errors
   - `exception_id`, `error_code`, `category`, `severity`, `message`, `user_message`, `session_id`, `request_id`, `node_name`, `stacktrace`, `metadata`, `timestamp`, `user_id`

3. **`requests`** - Complete request-response cycles
   - `request_id`, `session_id`, `user_id`, `user_text`, `intent`, `confidence`, `response`, `metadata`, `duration_ms`, `timestamp`

### Querying Logs

#### View All Recent Logs
```bash
sqlite3 data/telemetry.db "SELECT * FROM logs ORDER BY timestamp DESC LIMIT 20;"
```

#### View Logs for a Specific Request (by UUID)
```bash
# Replace 'your-uuid-here' with actual request UUID
sqlite3 data/telemetry.db "SELECT * FROM logs WHERE request_id = 'your-uuid-here' ORDER BY timestamp;"
```

#### View Logs for a Specific Session
```bash
sqlite3 data/telemetry.db "SELECT node_name, event_type, timestamp, SUBSTR(data, 1, 100) as data_preview FROM logs WHERE session_id = 'your-session-id' ORDER BY timestamp;"
```

#### View Logs by Node
```bash
sqlite3 data/telemetry.db "SELECT node_name, event_type, COUNT(*) as count FROM logs GROUP BY node_name, event_type ORDER BY count DESC;"
```

#### View Confidence Check Decisions
```bash
sqlite3 data/telemetry.db "SELECT node_name, timestamp, json_extract(data, '$.decision') as decision, json_extract(data, '$.confidence') as confidence FROM logs WHERE event_type = 'confidence_check_decision' ORDER BY timestamp DESC LIMIT 10;"
```

#### View Context Builder Input/Output
```bash
# View context builder inputs
sqlite3 data/telemetry.db "SELECT timestamp, json_extract(data, '$.intent') as intent, json_extract(data, '$.domain') as domain FROM logs WHERE event_type = 'context_builder_input' ORDER BY timestamp DESC LIMIT 10;"

# View context builder outputs
sqlite3 data/telemetry.db "SELECT timestamp, json_extract(data, '$.history_length') as history_length, json_extract(data, '$.facts_count') as facts_count FROM logs WHERE event_type = 'context_builder_output' ORDER BY timestamp DESC LIMIT 10;"
```

### Querying Exceptions

#### View All Recent Exceptions
```bash
sqlite3 data/telemetry.db "SELECT node_name, error_code, category, severity, SUBSTR(message, 1, 80) as message_preview, timestamp FROM exceptions ORDER BY timestamp DESC LIMIT 20;"
```

#### View Exceptions by Node
```bash
sqlite3 data/telemetry.db "SELECT node_name, COUNT(*) as count, MAX(timestamp) as latest FROM exceptions GROUP BY node_name ORDER BY count DESC;"
```

#### View Exceptions by Severity
```bash
sqlite3 data/telemetry.db "SELECT severity, COUNT(*) as count FROM exceptions GROUP BY severity;"
```

#### View Exceptions for a Specific Request (by UUID)
```bash
sqlite3 data/telemetry.db "SELECT node_name, error_code, category, severity, message, timestamp FROM exceptions WHERE request_id = 'your-uuid-here' ORDER BY timestamp;"
```

#### View Exception Details with Stack Trace
```bash
# View full exception details including stack trace
sqlite3 data/telemetry.db "SELECT exception_id, node_name, error_code, message, stacktrace FROM exceptions WHERE node_name = 'safety_precheck' ORDER BY timestamp DESC LIMIT 1;" | python -m json.tool
```

#### View Exceptions by Error Code
```bash
sqlite3 data/telemetry.db "SELECT error_code, COUNT(*) as count, category FROM exceptions GROUP BY error_code, category ORDER BY count DESC;"
```

### Advanced Queries

#### Find All Logs and Exceptions for a Request (Full Audit Trail)
```bash
# Get all logs for a request UUID
sqlite3 data/telemetry.db "
SELECT 'LOG' as type, node_name, event_type, timestamp, data 
FROM logs 
WHERE request_id = 'your-uuid-here'
UNION ALL
SELECT 'EXCEPTION' as type, node_name, error_code, timestamp, message 
FROM exceptions 
WHERE request_id = 'your-uuid-here'
ORDER BY timestamp;
"
```

#### Count Logs and Exceptions by Node (Health Check)
```bash
sqlite3 data/telemetry.db "
SELECT 
    node_name,
    (SELECT COUNT(*) FROM logs WHERE logs.node_name = nodes.node_name) as log_count,
    (SELECT COUNT(*) FROM exceptions WHERE exceptions.node_name = nodes.node_name) as exception_count
FROM (
    SELECT DISTINCT node_name FROM logs
    UNION
    SELECT DISTINCT node_name FROM exceptions
) as nodes
ORDER BY exception_count DESC, log_count DESC;
"
```

#### View Exception Rate Over Time
```bash
sqlite3 data/telemetry.db "
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as exception_count,
    COUNT(DISTINCT node_name) as affected_nodes
FROM exceptions 
GROUP BY DATE(timestamp)
ORDER BY date DESC
LIMIT 7;
"
```

### Using SQLite Browser Tools

For a GUI experience, you can use:
- **DB Browser for SQLite** (cross-platform): https://sqlitebrowser.org/
- **VS Code Extension**: SQLite Viewer
- **DBeaver** (cross-platform database tool)

Open `data/telemetry.db` in any of these tools to browse tables visually.

### Example: Complete Request Audit Trail

To see everything that happened for a specific request:

```bash
REQUEST_UUID="your-request-uuid-here"

echo "=== LOGS ==="
sqlite3 data/telemetry.db "SELECT node_name, event_type, timestamp, data FROM logs WHERE request_id = '$REQUEST_UUID' ORDER BY timestamp;"

echo -e "\n=== EXCEPTIONS ==="
sqlite3 data/telemetry.db "SELECT node_name, error_code, severity, message FROM exceptions WHERE request_id = '$REQUEST_UUID' ORDER BY timestamp;"

echo -e "\n=== REQUEST SUMMARY ==="
sqlite3 data/telemetry.db "SELECT user_text, intent, confidence, response, duration_ms FROM requests WHERE request_id = '$REQUEST_UUID';"
```

### Quick Reference: Most Common Queries

```bash
# View recent exceptions
sqlite3 data/telemetry.db "SELECT node_name, error_code, severity, timestamp FROM exceptions ORDER BY timestamp DESC LIMIT 10;"

# View recent logs
sqlite3 data/telemetry.db "SELECT node_name, event_type, timestamp FROM logs ORDER BY timestamp DESC LIMIT 10;"

# Count exceptions by node
sqlite3 data/telemetry.db "SELECT node_name, COUNT(*) FROM exceptions GROUP BY node_name;"

# View all logs for a session
sqlite3 data/telemetry.db "SELECT * FROM logs WHERE session_id = 'your-session-id' ORDER BY timestamp;"

# View all exceptions for a session
sqlite3 data/telemetry.db "SELECT * FROM exceptions WHERE session_id = 'your-session-id' ORDER BY timestamp;"
```

### Production Considerations
Ideas to upgrade:
- Add correlation IDs (already have `session_id` and `request_id`/`uuid` concept)
- Ship logs to ELK / Splunk / OpenTelemetry
- Add latency metrics per node (LangGraph hooks or decorators)
- Migrate from SQLite to MongoDB/Firestore for production scale
- Add log retention policies and archival

---
## 16. Security & Compliance Placeholders
This starter does not implement:
- HIPAA / PHI redaction
- AuthN/AuthZ
- Rate limiting / abuse prevention
- Data encryption at rest/in transit beyond defaults

Before production: Engage security review, threat modeling, privacy compliance.

---
## 17. Troubleshooting Cheat Sheet
Extra Windows entries:
| Issue | Windows Quick Fix |
|-------|--------------------|
| Activation script blocked | Run PowerShell as admin: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Port appears busy after stop | `Get-NetTCPConnection -LocalPort 8000` then kill owning process |
| Missing curl/jq | Install Git for Windows or use PowerShell alternatives |
| Unicode log chars garbled | Switch to newer Windows Terminal or enable UTF-8 (`chcp 65001`) |

| Issue | Quick Action |
|-------|--------------|
| 404 Not Found | Confirm router prefix `/api/v1` + POST method |
| 500 KeyError | Check prompt templates for stray `{}` braces |
| Cache not working | Ensure `enable_semantic_cache=True` |
| Memory not updating | Verify flow edge reaches `update_memory` |
| Infinite reload errors | Run without `--reload` while debugging |
| Claim number not detected | Expand regex / ensure word "claim" present |

Use `logger.debug` prints or breakpoints; avoid silent failures.

---
## 18. Next Steps / TODO Roadmap
- [ ] Replace `MockLLM` with real intent classifier (could be a fine‑tuned model or embedding similarity)
- [ ] Implement dynamic response agent with retrieval (drug formulary, coverage rules)
- [ ] Real claims API client (retry, timeout, circuit breaker)
- [ ] Add unit tests (intent routing, clarification trigger, memory persistence)
- [ ] Integrate structured logging + tracing
- [ ] Add safety filters & redaction
- [ ] Persistent checkpoint store with migrations
- [ ] Add docker-compose for local dependencies (Redis, Postgres)
- [ ] CI pipeline (lint, type-check, tests) + PR gating

---
## 19. Minimal Code Tour
| File | Purpose |
|------|---------|
| `main.py` | FastAPI app wiring, startup/shutdown hooks |
| `api/routes.py` | `/chat` endpoint, response shaping & error logging |
| `langgraph_agent.py` | Graph building, node edges, init/close lifecycle |
| `nodes/*` | Pure functional pieces: safety, cache, context, clarification, memory |
| `agents/intent_agent.py` | Intent + entity extraction via mock LLM |
| `agents/response_agent.py` | Response synthesis via mock LLM |
| `tools/claims_api.py` | Mock external data source |
| `state/schema.py` | State shape & initial state factory |
| `core/config.py` | Settings & feature flags |
| `core/logger.py` | Basic logger factory |

Recommended reading order: `state/schema.py` → `langgraph_agent.py` → `nodes/confidence.py` → `agents/intent_agent.py`.

---
## 20. Contributing Etiquette (Internal Team)
1. Open small PRs – easier to review.
2. Include context in description (“Adds coverage tool node”).
3. Keep mock code clearly labeled to avoid accidental production use.
4. Tag reviewers early (domain expert + platform engineer).
5. Update README when architectural changes occur.

---
## 21. Disclaimer
This repository is a **starter / accelerator**. All mock logic, placeholder prompts, and simplified pathways must be **validated, hardened, and replaced** before serving real member data or connecting to protected health information systems.

Proceed thoughtfully – and have fun exploring LangGraph.

---
## 22. Quick Copy/Paste Command Bundle
macOS / Linux:
```bash
python main.py
curl -s -X POST http://localhost:8000/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"text":"why was my claim rejected","session_id":"demo"}' | jq
```
Windows (PowerShell):
```powershell
py -3.11 main.py
$body = @{ text = 'why was my claim rejected'; session_id = 'demo' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://localhost:8000/api/v1/chat' -Method Post -ContentType 'application/json' -Body $body
```
Follow-up (PowerShell):
```powershell
$body = @{ text = 'Claim 987654 was rejected. Why?'; session_id = 'demo' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://localhost:8000/api/v1/chat' -Method Post -ContentType 'application/json' -Body $body
```

---
**End of README – iterate as you evolve the system.**
