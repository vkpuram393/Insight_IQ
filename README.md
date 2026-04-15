# PBM AI Assist – LangGraph Multi‑Agent Starter

> A pragmatic starter project for building a pharmacy benefit (PBM) conversational assistant using **LangGraph**. This project uses **Google Gemini** for LLM integration and includes a configurable mock mode for development. It demonstrates flow, state, routing, memory, and clarification logic with real LLM integration and API calls.

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
- Fully production-hardened (some components still need scaling)

The system uses **real Google Gemini LLM** and **real API integrations** with configurable mock mode for development. You can extend with additional enterprise logic, observability, guardrails, and compliance controls as needed.

---
## 2. Tech Stack & Versions
| Component | Version (pinned) | Notes |
|-----------|------------------|-------|
| Python | 3.11.x (tested on 3.11.9) | Required – 3.12 not yet verified |
| FastAPI | 0.115.0 | REST layer |
| Uvicorn | 0.24.0 | ASGI server |
| LangGraph | 0.2.45 | Graph orchestration |
| LangChain Core | 0.3.18 | Prompt/message abstractions |
| google-genai | 1.0.0 | Google Gemini LLM client |
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
All runtime settings live in `config/config.py` via `Settings`. A `.env` file is optional.
Key flags:
- `use_mock_llm = False` → Uses real Gemini LLM (set to `True` for mock mode)
- `enable_checkpointing` → Currently `False` for simplicity (async saver is wired but can be toggled)
- `confidence_threshold` → Router decision for clarification vs tool path
- `enable_semantic_cache` → In‑memory cache on or off

Set environment variables in `.env` for real LLM integration:
```
GEMINI_API_KEY=your-gemini-api-key
USE_MOCK_LLM=False
```
When `USE_MOCK_LLM=True`, the system uses mock responses without external API calls.

---
## 5. Run the Server

### Option 1: Direct Python (Simple)
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

### Option 2: Using Makefile (Runs Tests First)
```bash
# Run tests then start server (recommended for local builds)
make build

# Or just run tests
make test

# Or just start server
make run
```

### Option 3: Using Build Script
```bash
# macOS / Linux
./build.sh

# This will:
# 1. Run all endpoint tests
# 2. If tests pass, start the server
# 3. If tests fail, exit without starting server
```

### Option 4: Uvicorn Directly
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 5: Auto-Run Tests on Startup (Development Only)
Set environment variable to automatically run tests when server starts:
```bash
# macOS / Linux
RUN_TESTS_ON_STARTUP=true python main.py

# Windows PowerShell
$env:RUN_TESTS_ON_STARTUP="true"; python main.py
```

**Note:** Tests run automatically on startup only if:
- `ENVIRONMENT=development` (in config or .env)
- `RUN_TESTS_ON_STARTUP=true` (environment variable)

On Windows CMD/PowerShell, ensure the virtual env is activated for all options.

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
  → orchestrator (entry point)
  → safety_precheck (PII masking + safety pattern checks + Gemini safety)
  → check_cache (currently always continues to intent_agent)
  → intent_agent (Gemini LLM classifies intent + extracts entities)
  → confidence_checker (evaluates confidence and missing slots)
  → confidence_check_router (three-way routing):
      ├─ clarification (missing required entities) 
      │     → update_memory → cache_response → END
      ├─ build_context (simple query with entities) 
      │     → call_claims_tool (real API call)
      │     → response_safety_pii_precheck (mask PII before LLM)
      │     → response_agent (Gemini LLM generates response)
      │     → response_safety_pii_postcheck (unmask PII for user)
      │     → update_memory → cache_response → END
      └─ response_agent (complex query, skip API)
            → response_safety_pii_precheck (mask PII before LLM)
            → response_agent (Gemini LLM generates response)
            → response_safety_pii_postcheck (unmask PII for user)
            → update_memory → cache_response → END
```
**Key Points:**
- **Orchestrator** is the entry point that initializes the flow
- **Safety checks** happen at input (precheck) and output (postcheck) with PII masking/unmasking
- **Cache** is checked but currently always continues (not fully implemented)
- **Three routing paths**: clarification (missing data), build_context→API (simple queries), or direct response_agent (complex queries)
- **PII protection** is applied before LLM calls and unmasked before returning to user
- Nodes live in `nodes/`; agents in `agents/`; tool(s) in `tools/`; classifiers in `classifiers/`; services in `services/`.

---
## 7. Agents & LLM Integration
### Intent Agent (`agents/intent_agent.py`)
- Uses **Google Gemini** for intent classification (or mock mode when `USE_MOCK_LLM=True`)
- Extracts `claim_number` and other entities from user input
- Emits: `intent`, `confidence`, `entities`, `slots`, `required_slots`, `missing_slots`
- Handles markdown code block stripping from Gemini responses

### Response Agent (`agents/response_agent.py`)
- Uses **Google Gemini** for natural language response generation (or mock mode)
- Incorporates `tool_results` from claims API into responses
- Generates contextually appropriate responses based on intent and API data

### Claims Tool (`tools/claims_api.py`)
- **Real API client** with retry logic and error handling
- Integrates with CVS Claims API (`claiminquiry-exp-qa.myclaims.pss-np.caremark.com`)
- Supports multiple API endpoints with routing based on intent
- Includes request validation, retry mechanisms, and comprehensive error handling

---
## 8. Memory & Cache
- **Short‑term memory**: Conversation history stored via `MemoryStore` facade (in‑memory by default, configurable to Redis/Memorystore)
- **Long‑term facts**: Session facts stored for context across conversations
- **Cache**: Keyed by MD5 hash of lowercased text; stores response + intent + confidence
  - **Note**: Cache check currently always continues to intent_agent (cache hit/miss routing not fully implemented in graph)
  - Cache is functional and stores/retrieves data, but graph routing always proceeds to intent_agent
- **Clarification** persists to memory: `clarification → update_memory → cache_response → END`

**Production considerations:**
- Switch memory store to Redis/Memorystore for production (configure via `MEMORY_STORE_TYPE`)
- Cache uses 1-hour TTL (configurable)
- Persist conversation state with LangGraph's checkpointing (currently disabled by default)

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
## 11. Production Readiness Status
| Area | Current | Production Status |
|------|---------|-------------------|
| LLM (intent) | **Google Gemini** (real) | ✅ Production-ready with configurable mock mode |
| LLM (response) | **Google Gemini** (real) | ✅ Production-ready with configurable mock mode |
| Claims tool | Real API client with retry logic | ✅ Production-ready |
| Memory | In‑memory dict (dev) / Redis (configurable) | ⚠️ Switch to Redis/Memorystore for production |
| Checkpointing | Async SQLite | ⚠️ Migrate to managed durable storage for production |
| Safety | PII protection + safety checks | ✅ Production-ready with Presidio |
| Logging | SQLite telemetry with state snapshots | ✅ Production-ready (migrate to Firestore for scale) |

---
## 12. Extending Intents
1. Add new intent examples to the intent classifier in `classifiers/` or update Gemini prompts in `agents/intent_agent.py`.
2. Update system prompt examples in `agents/intent_agent.py`.
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
- **State snapshots**: All nodes log complete state snapshots after successful execution via `log_state_snapshot()`
- **SQLite exceptions**: All exceptions logged to `data/telemetry.db` with full stack traces via `log_exception()`
- **WAL mode**: SQLite uses Write-Ahead Logging for better concurrency and performance
- **Test database isolation**: Separate test database (`data/telemetry_test.db`) for unit tests

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

### SQLite UI Viewers

Instead of using terminal commands, you can use GUI tools to view your SQLite database:

#### 1. **DB Browser for SQLite** (Recommended - Free, Cross-platform)
- **Download**: https://sqlitebrowser.org/
- **Features**: 
  - Visual table browser
  - SQL query editor
  - Data export/import
  - Schema visualization
- **Usage**: 
  1. Install DB Browser
  2. Open `data/telemetry.db`
  3. Browse tables, run queries, export data

#### 2. **TablePlus** (macOS/Windows - Free tier available)
- **Download**: https://tableplus.com/
- **Features**: Modern UI, multiple database support
- **Usage**: Connect to SQLite file, browse visually

#### 3. **DBeaver** (Free, Cross-platform)
- **Download**: https://dbeaver.io/
- **Features**: 
  - Universal database tool
  - SQL editor with syntax highlighting
  - Data visualization
- **Usage - Step by Step**:
  1. Open DBeaver
  2. Click **"New Database Connection"** (plug icon) or go to `Database` → `New Database Connection`
  3. In the connection wizard, select **"SQLite"** from the list
  4. Click **Next**
  5. In the **"Path"** field, click the folder icon and navigate to your project directory
  6. Select `data/telemetry.db` file
  7. Click **Test Connection** to verify it works
  8. Click **Finish**
  9. The database will appear in the Database Navigator panel on the left
  10. Expand it to see tables: `logs`, `exceptions`, `requests`
  11. Right-click any table → **"View Data"** to browse records
  12. Right-click any table → **"SQL Editor"** → **"New SQL Script"** to run queries

**Troubleshooting: SQLite JDBC Driver Error**

If you get an error like "can't load driver class 'org.sqlite.JDBC'":

**Solution 1: Download Driver Automatically (Easiest)**
1. When creating the connection, if you see a driver download prompt, click **"Download"**
2. DBeaver will automatically download the SQLite JDBC driver

**Solution 2: Configure Driver Manually**
1. Go to `Database` → `Driver Manager` (or `Window` → `Driver Manager`)
2. Find **"SQLite"** in the list and select it
3. Click **"Edit"** button
4. Go to **"Libraries"** tab
5. Click **"Download/Update"** button
6. Wait for download to complete
7. Click **"OK"** to save
8. Try creating the connection again

**Solution 3: Manual Driver Download**
1. Download SQLite JDBC driver from: https://github.com/xerial/sqlite-jdbc/releases
2. Download the latest `sqlite-jdbc-X.X.X.jar` file (e.g., `sqlite-jdbc-3.44.1.0.jar`)
3. In DBeaver: `Database` → `Driver Manager` → Select "SQLite" → `Edit` → `Libraries` tab
4. Click **"Add File"** and select the downloaded `.jar` file
5. Click **"OK"** to save
6. Try creating the connection again

**Solution 4: Use Native SQLite (Alternative)**
If JDBC continues to cause issues, you can use DBeaver's native SQLite connection:
1. When creating connection, look for **"SQLite (Native)"** option instead of "SQLite"
2. This uses the system's SQLite installation instead of JDBC driver

#### 4. **VS Code Extension** (Recommended for VS Code users)

**Option A: SQLite Extension by Alex Covizzi** (Most Popular - Recommended)
- **Extension**: **"SQLite"** by Alex Covizzi (alexcvzz)
- **Installation**:
  1. Open VS Code
  2. Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac) to open Extensions
  3. Search for "SQLite"
  4. Click **Install** on "SQLite" by Alex Covizzi
- **Usage**:
  1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) to open Command Palette
  2. Type "SQLite: Open Database" and select it
  3. Navigate to `data/telemetry.db` and select it
  4. The database will open in the sidebar showing all tables
- **Note**: Requires SQLite command-line tool installed on your system
  - **macOS**: Usually pre-installed. Verify with `sqlite3 --version` in terminal
  - **Windows**: Download from https://www.sqlite.org/download.html and add to PATH
  - **Linux**: `sudo apt install sqlite3` (Debian/Ubuntu)

**Option B: SQLite Viewer by qwtel** (Alternative)
- **Extension**: **"SQLite Viewer"** by Florian Klampfer (qwtel)
- **Installation**: Same as above, search for "SQLite Viewer"
- **Usage**:
  1. Right-click on `data/telemetry.db` in VS Code's file explorer
  2. Look for "Open with SQLite Viewer" or similar option
  3. Or use Command Palette: `Ctrl+Shift+P` → type "SQLite Viewer"

**Option C: DevDB** (Full Database GUI Client)
- **Extension**: **"DevDB"** - Full database GUI client
- Supports SQLite, MySQL, PostgreSQL
- Provides visual database browser with query editor
- Search for "DevDB" in Extensions marketplace

**Troubleshooting**:
- If "Open Database" command doesn't appear:
  1. **Restart VS Code** after installing extension
  2. **Check SQLite installation**: Run `sqlite3 --version` in terminal
  3. **Try right-clicking** the `.db` file in VS Code file explorer
  4. **Check extension settings**: `Ctrl+,` (or `Cmd+,` on Mac) → search "sqlite" → configure SQLite path if needed
  5. **Alternative**: Use DB Browser for SQLite (Option 1 above) - it's more reliable

#### Quick Query Examples for UI Tools

**View all logs for a request (by UUID):**
```sql
SELECT * FROM logs 
WHERE request_id = 'your-uuid-here' 
ORDER BY timestamp;
```

**View all exceptions:**
```sql
SELECT node_name, error_code, severity, message, timestamp 
FROM exceptions 
ORDER BY timestamp DESC;
```

**View logs by node:**
```sql
SELECT node_name, event_type, COUNT(*) as count 
FROM logs 
GROUP BY node_name, event_type 
ORDER BY count DESC;
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

### Node Input/Output Documentation

Each node in the LangGraph receives the full `AgentState` and returns a partial update dictionary. Here's what each node reads from state and what it writes back:

#### Graph Flow Overview
```
START
  ↓
orchestrator → safety_precheck → check_cache → intent_agent
  ↓
confidence_checker → confidence_check_router (three-way)
  ↓                    ↓                    ↓
clarification    build_context      response_safety_pii_precheck
  ↓                    ↓                    ↓
update_memory    call_claims_tool   response_agent
  ↓                    ↓                    ↓
cache_response   response_safety    response_safety_pii_postcheck
  ↓              _pii_precheck            ↓
END                    ↓              update_memory
                response_agent            ↓
                      ↓              cache_response
                response_safety           ↓
                _pii_postcheck         END
                      ↓
                update_memory
                      ↓
                cache_response
                      ↓
                    END
```

#### 1. **orchestrator_node** (`nodes/orchestrator.py`)
- **Input (reads from state)**:
  - `text`: User's input message
  - `session_id`: Session identifier
  - `uuid`: Request UUID (for logging)
  - `user_info`: User metadata (for logging)
- **Output (writes to state)**:
  - `metadata`: Updated with orchestrator processing info
  - `error`: `Optional[str]` - Error message if exception occurred
- **Next Node**: Always goes to `safety_precheck`

#### 2. **safety_precheck_node** (`nodes/safety.py`)
- **Input (reads from state)**:
  - `text`: User's input message
  - `session_id`: Session identifier
  - `uuid`: Request UUID (for logging)
  - `user_info`: User metadata (for logging)
- **Output (writes to state)**:
  - `safety_precheck_passed`: `bool` - Whether input passed safety checks
  - `safety_block_reason`: `Optional[str]` - Reason if blocked
  - `response`: `str` - Error message if blocked
  - `error`: `Optional[str]` - Error message if exception occurred
- **Next Node**: Routes to `check_cache` if passed, or `END` if blocked

#### 3. **check_cache_node** (`nodes/cache.py`)
- **Input (reads from state)**:
  - `text`: User's input message (hashed for cache key)
  - `session_id`, `uuid`, `user_info`: For logging
- **Output (writes to state)**:
  - `cache_hit`: `bool` - Whether cache was found
  - `response`: `str` - Cached response (if cache hit)
  - `intent`: `Optional[str]` - Cached intent (if cache hit)
  - `confidence`: `Optional[float]` - Cached confidence (if cache hit)
  - `metadata`: Updated with cache status
  - `error`: `Optional[str]` - Error message if exception occurred
- **Next Node**: Always routes to `intent_agent` (cache hit/miss routing not fully implemented in graph)

#### 4. **intent_agent_node** (`agents/intent_agent.py`)
- **Input (reads from state)**:
  - `text`: User's input message
  - `conversation_history`: Recent conversation context
  - `session_id`, `uuid`, `user_info`: For logging
- **Output (writes to state)**:
  - `intent`: `str` - Detected intent (e.g., "claim_status", "claim_rejection_reason")
  - `confidence`: `float` - Confidence score (0.0 to 1.0)
  - `entities`: `Dict[str, Any]` - Extracted entities (e.g., `{"claim_number": "12345"}`)
  - `slots`: `Dict[str, Any]` - API parameters (from intent classifier)
  - `required_slots`: `List[str]` - Required slots for this intent
  - `missing_slots`: `List[str]` - Required slots that are missing
  - `error`: `Optional[str]` - Error message if exception occurred
- **Next Node**: Always goes to `confidence_checker`

#### 5. **build_context_node** (`nodes/context.py`)
- **Input (reads from state)**:
  - `session_id`: To fetch conversation history
  - `slots`: Current slots from intent classifier
  - `required_slots`: Required slots for intent
  - `missing_slots`: Missing required slots
  - `domain`: Domain context
  - `uuid`: Request UUID
  - `user_info`: User information
- **Output (writes to state)**:
  - `conversation_history`: `List[Dict[str, str]]` - Last N messages (configurable)
  - `relevant_facts`: `List[Dict[str, Any]]` - Important facts from session
  - `extracted_slots`: `Dict[str, Any]` - Slots extracted from conversation history
  - `planner_context`: `Dict[str, Any]` - Complete context object for planner/executor
  - `error`: `Optional[str]` - Error message if exception occurred
- **Next Node**: Always goes to `call_claims_tool`

#### 6. **confidence_checker_node** (`nodes/confidence.py`)
- **Input (reads from state)**:
  - `intent`: Detected intent
  - `confidence`: Confidence score from intent agent
  - `missing_slots`: Missing required slots
- **Output (writes to state)**:
  - `needs_clarification`: `bool` - Whether clarification is needed
  - `confidence_check_passed`: `bool` - Whether confidence check passed
  - `metadata`: Updated with confidence check results
- **Next Node**: Always goes to `confidence_check_router`

#### 7. **confidence_check_router** (`nodes/confidence.py`)
- **Input (reads from state)**:
  - `confidence`: Confidence score from intent agent
  - `missing_slots`: Missing required slots
  - `needs_clarification`: Whether clarification is needed
- **Output**: Returns routing decision (not state update)
  - `"clarification"` if missing slots exist OR low confidence
  - `"build_context"` if high confidence AND has entities (simple query → API)
  - `"response_agent"` if high confidence but complex query (skip API)
- **Next Node**: Routes to `clarification`, `build_context`, or `response_safety_pii_precheck`

#### 8. **clarification_node** (`nodes/clarification.py`)
- **Input (reads from state)**:
  - `intent`: Detected intent
  - `needs_clarification`: Whether clarification is needed
  - `session_id`, `uuid`, `user_info`: For logging
- **Output (writes to state)**:
  - `needs_clarification`: `bool` - Set to `True`
  - `clarifying_question`: `str` - The question to ask user
  - `response`: `str` - Same as clarifying_question (for UI)
  - `metadata`: Updated with clarification flag
  - `error`: `Optional[str]` - Error message if exception occurred
- **Next Node**: Goes to `update_memory` (then END)

#### 9. **call_claims_tool_node** (`tools/claims_api.py`)
- **Input (reads from state)**:
  - `intent`: What data to fetch
  - `entities`: Parameters (e.g., `{"claim_number": "12345"}`)
  - `session_id`, `uuid`, `user_info`: For logging
- **Output (writes to state)**:
  - `tool_results`: `Dict[str, Any]` - API response data
    - Example: `{"claim_id": "12345", "status": "processing", "submitted_date": "2025-01-10"}`
  - `error`: `Optional[str]` - Error message if exception occurred
- **Next Node**: Always goes to `response_agent`

#### 10. **response_safety_pii_precheck_node** (`nodes/safety.py`)
- **Input (reads from state)**:
  - `tool_results`: Data from API calls (if available)
  - `conversation_history`: Context for response generation
  - `session_id`, `uuid`, `user_info`: For logging
- **Output (writes to state)**:
  - `tool_results`: Masked PII/PHI in tool results before LLM call
  - `conversation_history`: Masked PII/PHI in conversation history
- **Next Node**: Always goes to `response_agent`

#### 11. **response_agent_node** (`agents/response_agent.py`)
- **Input (reads from state)**:
  - `intent`: Detected intent
  - `tool_results`: Data from API calls (with PII masked)
  - `conversation_history`: Context for response generation (with PII masked)
  - `entities`: Extracted entities
  - `session_id`, `uuid`, `user_info`: For logging
- **Output (writes to state)**:
  - `response`: `str` - The final natural language response to user (with PII still masked)
  - `error`: `Optional[str]` - Error message if exception occurred
- **Next Node**: Always goes to `response_safety_pii_postcheck`

#### 12. **response_safety_pii_postcheck_node** (`nodes/safety.py`)
- **Input (reads from state)**:
  - `response`: Generated response (with PII masked)
  - `session_id`, `uuid`, `user_info`: For logging
- **Output (writes to state)**:
  - `response`: `str` - Unmasked response ready for user (PII restored)
  - `error`: `Optional[str]` - Error message if exception occurred
- **Next Node**: Always goes to `update_memory`

#### 13. **update_memory_node** (`nodes/context.py`)
- **Input (reads from state)**:
  - `text`: User's input message
  - `response`: Assistant's response
  - `session_id`: To store in memory
  - `uuid`, `user_info`: For logging
- **Output (writes to state)**:
  - `conversation_history`: `List[Dict[str, str]]` - Updated history after adding user/assistant messages
  - `relevant_facts`: `List[Dict[str, Any]]` - Updated facts after extracting important information
  - `metadata`: Updated with `memory_updated: True`
  - `error`: `Optional[str]` - Error message if exception occurred
- **Next Node**: Always goes to `cache_response`

#### 14. **cache_response_node** (`nodes/cache.py`)
- **Input (reads from state)**:
  - `text`: User's input message (hashed for cache key)
  - `response`: Generated response to cache
  - `intent`: Detected intent to cache
  - `confidence`: Confidence score to cache
  - `session_id`, `uuid`, `user_info`: For logging
- **Output (writes to state)**:
  - `metadata`: Updated with `cached: bool` flag
  - `error`: `Optional[str]` - Error message if exception occurred
- **Next Node**: Always goes to `END`

### State Flow Example

**Initial State:**
```python
{
    "text": "What's my claim status?",
    "session_id": "session-123",
    "intent": None,
    "confidence": None,
    "response": ""
}
```

**After intent_agent_node:**
```python
{
    "text": "What's my claim status?",
    "session_id": "session-123",
    "intent": "claim_status",
    "confidence": 0.92,
    "entities": {"claim_number": "12345"},
    "response": ""
}
```

**After response_agent_node:**
```python
{
    "text": "What's my claim status?",
    "session_id": "session-123",
    "intent": "claim_status",
    "confidence": 0.92,
    "entities": {"claim_number": "12345"},
    "response": "Your claim #12345 is currently being processed..."
}
```

### Production Considerations
Ideas to upgrade:
- Add correlation IDs (already have `session_id` and `request_id`/`uuid` concept)
- Ship logs to ELK / Splunk / OpenTelemetry
- Add latency metrics per node (LangGraph hooks or decorators)
- Migrate from SQLite to MongoDB/Firestore for production scale
- Add log retention policies and archival

---
## 16. Security & Compliance
**Implemented:**
- ✅ **PII/PHI Protection**: Uses Presidio analyzer and anonymizer for masking/unmasking
- ✅ **Safety Checks**: Input and output safety validation with Gemini safety API
- ✅ **PII Masking**: Automatic PII masking before LLM calls, unmasking before user responses

**Not Yet Implemented:**
- ⚠️ AuthN/AuthZ (authentication/authorization)
- ⚠️ Rate limiting / abuse prevention
- ⚠️ Data encryption at rest/in transit beyond defaults

**Before production:** Engage security review, threat modeling, privacy compliance, and implement missing security controls.

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
- [x] Replace `MockLLM` with real Gemini LLM integration
- [x] Implement comprehensive logging with state snapshots
- [x] Add PII protection with Presidio
- [x] Real claims API client with retry logic
- [x] Reorganize codebase structure (classifiers, services, utils)
- [ ] Add unit tests (intent routing, clarification trigger, memory persistence)
- [ ] Migrate memory store to Redis/Memorystore for production
- [ ] Migrate telemetry to Firestore/BigQuery for production scale
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
| `agents/intent_agent.py` | Intent + entity extraction via Gemini LLM |
| `agents/response_agent.py` | Response synthesis via Gemini LLM |
| `tools/claims_api.py` | Claims API integration with retry logic |
| `state/schema.py` | State shape & initial state factory |
| `config/config.py` | Settings & feature flags |
| `core/logger.py` | Basic logger factory |
| `services/llm_connection.py` | Gemini LLM client wrapper |
| `services/pii_protection.py` | PII masking/unmasking utilities |
| `classifiers/` | Intent classification modules (keyword, embedded, unified) |
| `core/errors/` | Error handling, exceptions, and error models |
| `utils/` | Utility functions (entity extraction, retry, serialization) |

**Project Structure:**
```
pss-myclaims-ai-agent/
├── agents/              # AI agents (intent, response)
├── nodes/               # Graph nodes (cache, safety, context, etc.)
├── tools/               # External tools (claims API)
├── classifiers/         # Intent classification modules
├── services/            # External service integrations (LLM, PII, embeddings)
├── memory/              # Memory store facade
├── persistence/         # Persistence store facade
├── utils/               # Utility functions and test endpoints
├── api/                 # FastAPI routes
├── config/              # Configuration (settings, routing, domain)
├── core/                # Core functionality (logging, errors, telemetry)
├── state/               # State schema
├── scripts/             # Utility scripts (embedding generation)
├── data/                # SQLite databases
└── certs/               # SSL certificates
```

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
