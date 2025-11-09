# Developer Guide

Complete guide for setting up, developing, and testing the PSS MyClaims AI Agent.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation & Setup](#installation--setup)
3. [CVS Certificate Setup](#cvs-certificate-setup)
4. [Storage Architecture](#storage-architecture)
5. [Development Workflow](#development-workflow)
6. [Debugging Guide](#debugging-guide)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Navigate to project
cd /Users/c882025/PycharmProjects/pss-myclaims-ai-agent

# 2. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Setup CVS certificate (for proxy)
./setup_cert.sh

# 4. Install dependencies
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# 5. Run the application
python main.py

# 6. Test it works
curl http://localhost:8000/health
```

---

## Installation & Setup

### Prerequisites

- **Python 3.11.9** (verified working version)
- **macOS** (or Linux/Windows with adjustments)
- **CVS Network Access** (for certificate download)

### Step-by-Step Installation

#### 1. Verify Python Version
```bash
python --version
# Should show: Python 3.11.9
```

If you need to install Python 3.11:
```bash
brew install python@3.11  # macOS
```

#### 2. Create Virtual Environment
```bash
cd /Users/c882025/PycharmProjects/pss-myclaims-ai-agent
python -m venv .venv
source .venv/bin/activate
```

To verify the virtual environment:
```bash
which python
# Should show: .../pss-myclaims-ai-agent/.venv/bin/python
```

#### 3. Setup CVS Root Certificate

**See [CVS Certificate Setup](#cvs-certificate-setup) section below.**

#### 4. Install Dependencies

```bash
# With CVS proxy (recommended)
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Or with certificate (if configured)
pip install --cert certs/CVSHealthRoot.cer -r requirements.txt
```

#### 5. Configure Environment

The `.env` file is already configured with defaults. Key settings:

```bash
# LLM
USE_MOCK_LLM=True                    # Use mock LLM (no API keys needed)

# Memory Store
MEMORY_STORE_TYPE=inmemory           # Use in-memory cache (no Redis needed)

# Persistence Store  
PERSISTENCE_STORE_TYPE=sqlite        # Use SQLite (no Firestore needed)
ENABLE_TELEMETRY=True                # Enable telemetry logging

# SSL Certificates
SSL_CERT_FILE=certs/CVSHealthRoot.cer
REQUESTS_CA_BUNDLE=certs/CVSHealthRoot.cer
```

#### 6. Verify Installation

```bash
# Run the application
python main.py

# In another terminal, test the health endpoint
curl http://localhost:8000/health

# Expected: {"status":"healthy"}
```

---

## CVS Certificate Setup

### Why This Is Needed

The CVS Health Root Certificate is required for SSL/TLS verification when making outbound connections through the CVS proxy/Zscaler.

### Automated Setup (Recommended)

```bash
./setup_cert.sh
```

This script will:
- Create the `certs/` directory
- Download the CVS root certificate from `http://crl.cvshealth.com/CVSHealthRoot.cer`
- Verify the certificate is valid
- Display certificate details

**Expected Output:**
```
================================
CVS Root Certificate Setup
================================
Attempting to download CVS root certificate...
✓ Certificate downloaded successfully to certs/CVSHealthRoot.cer
  File size: 2.0K
✓ Certificate appears to be valid

Certificate details:
subject=C=US, O=CVS Health Corporation, CN=CVSHealthRoot
issuer=C=US, O=CVS Health Corporation, CN=CVSHealthRoot
notBefore=Apr  9 00:00:00 2024 GMT
notAfter=Apr  8 23:59:59 2044 GMT
================================
```

### Manual Setup (If Script Fails)

If the automated script fails due to proxy blocking:

**Option 1: Download via Browser**
1. Open: http://crl.cvshealth.com/CVSHealthRoot.cer
2. Save file as: `certs/CVSHealthRoot.cer`

**Option 2: Copy from System (macOS)**
```bash
# Check if already in keychain
security find-certificate -a -c "CVS" -p > certs/CVSHealthRoot.cer

# Or search common locations
find /etc /usr/local -name "*CVS*" 2>/dev/null
```

**Option 3: Contact IT Support**
- Request the certificate from CVS IT/DevSecOps
- They may have it pre-installed on corporate machines

### Environment Configuration

The certificate paths are already configured in `.env`:

```bash
SSL_CERT_FILE=certs/CVSHealthRoot.cer
REQUESTS_CA_BUNDLE=certs/CVSHealthRoot.cer
```

These environment variables tell Python libraries (requests, httpx, urllib3) to use the CVS root certificate.

### Verification

Test that the certificate works:

```bash
source .venv/bin/activate

# Test with Python
python -c "import requests; print(requests.get('https://www.google.com').status_code)"
# Should print: 200

# Test pip with certificate
pip install --cert certs/CVSHealthRoot.cer certifi
```

### Docker Configuration

The `Dockerfile` is pre-configured to include the certificate:

```dockerfile
# CVS Health Root CA certificate for Zscaler proxy
COPY certs/CVSHealthRoot.cer /usr/local/share/ca-certificates/CVSHealthRoot.crt
RUN update-ca-certificates

# Set environment variables for Python SSL certificate handling
ENV SSL_CERT_FILE=/usr/local/share/ca-certificates/CVSHealthRoot.crt
ENV REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/CVSHealthRoot.crt
```

To build:
```bash
# Make sure certificate exists first!
docker build -t pss-myclaims-ai-agent .
```

### Troubleshooting Certificates

**Certificate Not Found:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'certs/CVSHealthRoot.cer'
```
→ Run `./setup_cert.sh` or manually download

**SSL Verification Failed:**
```
SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED]'))
```
→ Solutions:
1. Verify file exists: `ls -lh certs/CVSHealthRoot.cer`
2. Check environment: `echo $SSL_CERT_FILE`
3. Try absolute path in `.env`
4. Convert format if needed:
   ```bash
   openssl x509 -in certs/CVSHealthRoot.cer -inform DER -out certs/CVSHealthRoot.pem
   ```

**pip Install Fails:**
```bash
# Use trusted hosts to bypass SSL for PyPI
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

---

## Storage Architecture

### Overview

The project uses a **facade pattern** for storage layers, allowing easy switching between development and production implementations.

```
Application Code (nodes/agents)
        ↓
   Facades (interfaces)
        ↓
  Implementations
        ↓
Storage (SQLite, Memory)
```

### Two Storage Layers

#### 1. Memory Store (Cache & Session Memory)

**Location:** `memory/` folder

**Purpose:** Fast temporary storage for caching and conversation context

**Current Implementation:** InMemoryStore (Python dictionaries)
- ✅ Fast, no dependencies
- ✅ Perfect for development
- ⚠️ Data lost on restart

**Future Implementations:**
- Redis (staging/production)
- GCP Memorystore (production at scale)

**Configuration:**
```bash
MEMORY_STORE_TYPE=inmemory    # Change to: redis, memorystore
```

**Operations:**
```python
from memory import MemoryStoreFactory

memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)

# Cache operations
await memory_store.set("key", {"data": "value"}, ttl_seconds=3600)
value = await memory_store.get("key")

# Session history
await memory_store.append_to_session("session_id", "user", "Hello")
history = await memory_store.get_session_history("session_id")

# Facts storage
await memory_store.add_session_fact("session_id", "claim_mention", {...})
facts = await memory_store.get_session_facts("session_id")
```

#### 2. Persistence Store (Telemetry & Analytics)

**Location:** `persistence/` folder

**Purpose:** Long-term storage for telemetry, analytics, and debugging

**Current Implementation:** SQLitePersistenceStore
- ✅ No server required
- ✅ Great for development/testing
- ✅ Suitable for small-medium production

**Future Implementations:**
- Firestore (production)
- BigQuery (analytics at scale)

**Configuration:**
```bash
PERSISTENCE_STORE_TYPE=sqlite    # Change to: firestore, bigquery
TELEMETRY_DB_PATH=data/telemetry.db
ENABLE_TELEMETRY=True
```

**Operations:**
```python
from core.telemetry import log_event, log_request_response
from persistence import EventType

# Log an event
await log_event(
    EventType.CACHE_HIT,
    session_id="session_123",
    data={"key": "some_cache_key"}
)

# Log complete request
await log_request_response(
    session_id="session_123",
    user_text="Why was my claim rejected?",
    intent="claim_status",
    confidence=0.95,
    response="Your claim was rejected because...",
    metadata={"duration_ms": 234}
)

# Get analytics
from core.telemetry import get_analytics
analytics = await get_analytics()
```

### Database Schema (SQLite)

**Events Table:**
```sql
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id TEXT,
    data JSON NOT NULL,
    timestamp TEXT NOT NULL
);
```

**Requests Table:**
```sql
CREATE TABLE requests (
    request_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT,
    user_text TEXT NOT NULL,
    intent TEXT,
    confidence REAL,
    response TEXT NOT NULL,
    metadata JSON,
    duration_ms INTEGER,
    timestamp TEXT NOT NULL
);
```

### Migration Path

**Current (Phase 1):**
```
✅ InMemoryStore - Cache & sessions
✅ SQLite - Telemetry & analytics
✅ No external dependencies
```

**Future (Phase 2 - When GCP Access Available):**
```
⏳ Redis/Memorystore - Cache & sessions
⏳ Firestore - Telemetry & analytics
⏳ Update .env only - NO code changes!
```

Just update configuration:
```bash
MEMORY_STORE_TYPE=memorystore
MEMORY_STORE_HOST=10.x.x.x
PERSISTENCE_STORE_TYPE=firestore
```

### Key Benefits

1. **No Blockers** - Works immediately with no GCP access
2. **Zero Code Changes** - Switch implementations via config only
3. **Production Ready** - SQLite handles millions of rows
4. **Observable** - Built-in analytics and monitoring
5. **Testable** - Mock implementations for unit tests

---

## Development Workflow

### Project Structure

```
pss-myclaims-ai-agent/
├── agents/              # AI agents (intent, response)
├── nodes/               # Graph nodes (cache, safety, etc.)
├── tools/               # External tools (claims API)
├── memory/              # Memory store facade
├── persistence/         # Persistence store facade
├── utils/               # Test endpoints
├── api/                 # FastAPI routes
├── core/                # Config, logging, telemetry
├── state/               # State schema
├── data/                # SQLite databases
├── certs/               # SSL certificates
└── .env                 # Configuration
```

### Component Testing

Each component can be tested independently using the utils endpoints.

**See [Testing](#testing) section and `TEMP_ENDPOINTS.md` for details.**

### Adding New Features

#### Example: Adding a New Node

1. Create node file in `nodes/`:
```python
# nodes/my_new_node.py
async def my_new_node(state: AgentState) -> Dict[str, Any]:
    logger.info("🔧 Node: My New Node")
    # Your logic here
    return {"new_field": "value"}
```

2. Export from `nodes/__init__.py`:
```python
from nodes.my_new_node import my_new_node
__all__ = [..., "my_new_node"]
```

3. Add to graph in `langgraph_agent.py`:
```python
builder.add_node("my_new_node", my_new_node)
```

4. Create test endpoint in `utils/test_endpoints.py`:
```python
@router.post("/test-my-new-node")
async def test_my_new_node(request: TestRequest):
    state = AgentState(...)
    result = await my_new_node(state)
    return result
```

5. Test independently:
```bash
curl -X POST http://localhost:8000/utils/test-my-new-node \
  -H 'Content-Type: application/json' \
  -d '{"test":"data"}'
```

### Code Style

- Use async/await for all I/O operations
- Add type hints to function signatures
- Include docstrings with 🎓 CONCEPT explanations
- Log important steps with emoji prefixes
- Handle exceptions gracefully

---

## Debugging Guide

### PyCharm Setup

#### 1. Configure Python Interpreter

**Important:** Make sure PyCharm uses the correct virtual environment!

1. Open PyCharm Settings: `Cmd + ,`
2. Navigate to: `Project: pss-myclaims-ai-agent → Python Interpreter`
3. Click gear icon (⚙️) → `Add Interpreter` → `Add Local Interpreter`
4. Choose `Existing environment`
5. Browse to: `/Users/c882025/PycharmProjects/pss-myclaims-ai-agent/.venv/bin/python`
6. Click OK

#### 2. Run Configuration

A pre-configured run configuration is available at:
```
.idea/runConfigurations/Debug_Main_py.xml
```

Or create manually:
- **Script path:** `main.py`
- **Python interpreter:** Project virtual environment
- **Environment variables:**
  ```
  SSL_CERT_FILE=certs/CVSHealthRoot.cer
  REQUESTS_CA_BUNDLE=certs/CVSHealthRoot.cer
  ```

#### 3. Key Breakpoint Locations

**Application Bootstrap** (`main.py`):
- Line 53: `async def startup_event()` - App initialization
- Line 79: `async def root()` - Root endpoint
- Line 89: `async def health()` - Health check

**API Routes** (`api/routes.py`):
- Line 35: `async def chat()` - Chat endpoint entry

**Agent Logic**:
- `agents/intent_classifier.py` - Intent classification
- `agents/intent_agent.py` - Intent agent processing
- `agents/response_agent.py` - Response generation

**Nodes**:
- `nodes/safety.py` - Safety checks
- `nodes/cache.py` - Cache operations
- `nodes/context.py` - Context building
- `nodes/clarification.py` - Clarification logic

### VS Code Setup

Use the pre-configured launch configuration:

**`.vscode/launch.json`:**
```json
{
  "name": "Python: Main.py Direct",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/main.py",
  "console": "integratedTerminal",
  "env": {
    "SSL_CERT_FILE": "certs/CVSHealthRoot.cer",
    "REQUESTS_CA_BUNDLE": "certs/CVSHealthRoot.cer"
  }
}
```

**Usage:**
1. Open Debug panel: `Cmd+Shift+D`
2. Select "Python: Main.py Direct"
3. Press `F5` to start
4. Set breakpoints by clicking left of line numbers

### Common Debugging Scenarios

#### Debugging a Request

1. Set breakpoint at `api/routes.py` line 35 (chat endpoint)
2. Send request:
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"text":"test","session_id":"debug"}'
```
3. Step through execution to see flow

#### Debugging Intent Classification

1. Set breakpoint in `agents/intent_classifier.py`
2. Use test endpoint:
```bash
curl -X POST http://localhost:8000/utils/test-intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"test intent"}'
```
3. Inspect variables in debugger

#### Debugging Cache Issues

1. Set breakpoint in `nodes/cache.py` at `check_cache_node`
2. Send same request twice to test caching
3. Inspect `_memory_store` operations

### Logging

**Log Levels:**
- DEBUG - Detailed information
- INFO - General information (default)
- WARNING - Warning messages
- ERROR - Error messages

**View Logs:**
```bash
# Console output shows all logs
python main.py

# Or check specific log files if configured
tail -f logs/app.log
```

**Add Custom Logging:**
```python
from core.logger import get_logger
logger = get_logger(__name__)

logger.debug("Detailed debug info")
logger.info("General info")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)
```

### Telemetry/Analytics for Debugging

Query SQLite database directly:

```bash
sqlite3 data/telemetry.db

# View recent requests
SELECT session_id, user_text, intent, confidence, timestamp 
FROM requests 
ORDER BY timestamp DESC 
LIMIT 10;

# View events by type
SELECT event_type, COUNT(*) as count 
FROM events 
GROUP BY event_type;

# Search for errors
SELECT * FROM events 
WHERE event_type = 'error_occurred' 
ORDER BY timestamp DESC;

# Exit
.exit
```

Or use the analytics endpoint:
```bash
curl http://localhost:8000/api/v1/analytics | jq
```

---

## Testing

### Test Endpoints

Comprehensive test endpoints are available for each component.

**Full documentation:** See `TEMP_ENDPOINTS.md`

**Quick Reference:**

```bash
# Health check
curl http://localhost:8000/utils/health

# Test intent classifier
curl -X POST http://localhost:8000/utils/test-intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"why was my claim rejected"}'

# Test cache
curl -X POST http://localhost:8000/utils/test-cache \
  -H 'Content-Type: application/json' \
  -d '{"key":"test","value":{"data":"value"}}'

# Test session history
curl -X POST http://localhost:8000/utils/test-session-history \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"test","role":"user","content":"hello"}'

# Test persistence
curl -X POST http://localhost:8000/utils/test-persistence \
  -H 'Content-Type: application/json' \
  -d '{"event_type":"CACHE_HIT","session_id":"test","data":{}}'
```

### API Documentation

Interactive API documentation is available when the app is running:

**Swagger UI:** http://localhost:8000/docs

**ReDoc:** http://localhost:8000/redoc

All endpoints are documented with:
- Request/response schemas
- Example values
- Try-it-out functionality

### Unit Testing

Create tests in `tests/` directory:

```python
# tests/test_intent_classifier.py
import pytest
from agents.intent_classifier import classify_intent

@pytest.mark.asyncio
async def test_claim_status_intent():
    result = await classify_intent("claim 12345 status", {})
    assert result["intent"] == "claim_status"
    assert result["confidence"] > 0.8
```

Run tests:
```bash
pytest tests/
```

### Integration Testing Workflows

See `TEMP_ENDPOINTS.md` for complete testing workflows including:
- Intent → Cache → Response flow
- Session memory management
- Telemetry logging
- Component-specific tests

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'langgraph'
```

**Solution:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Verify interpreter
which python
# Should show: .../pss-myclaims-ai-agent/.venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. PyCharm Using Wrong Interpreter

**Problem:**
```
[BOOT] executable = /Users/.../PBMAssist/venv/bin/python
```

**Solution:** See [PyCharm Setup](#1-configure-python-interpreter) above

#### 3. SSL Certificate Errors

**Problem:**
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solution:**
```bash
# Run certificate setup
./setup_cert.sh

# Or use trusted hosts for pip
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

#### 4. Port Already in Use

**Problem:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn main:app --port 8001
```

#### 5. Database Locked

**Problem:**
```
sqlite3.OperationalError: database is locked
```

**Solution:**
```bash
# Close all connections to database
# Restart application
# Or delete and recreate: rm data/telemetry.db
```

#### 6. Cache Not Working

**Problem:** Second request not faster (no cache hit)

**Solution:**
```bash
# Check configuration
grep ENABLE_SEMANTIC_CACHE .env
# Should be: ENABLE_SEMANTIC_CACHE=True

# Test cache directly
curl -X POST http://localhost:8000/utils/test-cache \
  -H 'Content-Type: application/json' \
  -d '{"key":"test","value":"data"}'
```

#### 7. Telemetry Not Logging

**Problem:** No data in `data/telemetry.db`

**Solution:**
```bash
# Check configuration
grep ENABLE_TELEMETRY .env
# Should be: ENABLE_TELEMETRY=True

# Check database exists
ls -lh data/telemetry.db

# Test telemetry directly
curl -X POST http://localhost:8000/utils/test-persistence \
  -H 'Content-Type: application/json' \
  -d '{"event_type":"CACHE_HIT","session_id":"test","data":{}}'
```

### Getting Help

1. **Check Logs** - Console output has detailed information
2. **Test Endpoints** - Use utils endpoints to isolate issues
3. **Query Database** - Check telemetry data directly
4. **Review Documentation** - Check relevant sections above
5. **Debug Mode** - Run in PyCharm/VS Code debugger

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Utils health
curl http://localhost:8000/utils/health

# Analytics (check telemetry working)
curl http://localhost:8000/api/v1/analytics
```

---

## Additional Resources

- **Main README:** `README.md` - Project overview
- **Test Endpoints:** `TEMP_ENDPOINTS.md` - Complete testing guide
- **Intent Specs:** `INTENT_CLASSIFIER_REQUIREMENTS.md` - Requirements
- **Code Walkthrough:** `walkthrough.md` - Architecture details
- **LangGraph Studio:** `docs/LANGGRAPH_STUDIO.md` - External tool

---

**Happy Developing! 🚀**

