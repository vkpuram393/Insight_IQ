# 🧪 Temporary Test Endpoints

## Purpose

Individual test endpoints for each component, allowing developers to work and test independently without running the entire application pipeline. Perfect for parallel development and unit testing.

---

## Quick Start

```bash
# Start the application
cd /path/to/your/project/pss-myclaims-ai-agent
source .venv/bin/activate
python main.py

# Test any endpoint below
curl -X POST http://localhost:8000/utils/test-intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"why was my claim rejected"}'
```

---

## 📋 Table of Contents

1. [Intent Classification](#1-intent-classification)
2. [Cache Operations](#2-cache-operations)
3. [Persistence/Telemetry](#3-persistencetelemetry)
4. [Session Memory](#4-session-memory)
5. [Context Building](#5-context-building)
6. [Safety Checks](#6-safety-checks)
7. [Clarification Logic](#7-clarification-logic)
8. [Claims API](#8-claims-api)
9. [Response Generation](#9-response-generation)
10. [Health Check](#10-health-check)

---

## 1. Intent Classification

### Test Intent Classifier (Simple)
**Endpoint**: `POST /utils/test-intent`

**What it tests**: `agents/intent_classifier.py`

**Purpose**: Test the basic intent classification logic

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_intent_classifier(request: IntentTestRequest):`
2. `agents/intent_classifier.py` - `async def classify_intent(text: str, user_info: Dict)`
3. `agents/intent_classifier.py` - `return {"intent": intent, "confidence": confidence}`

```bash
curl -X POST http://localhost:8000/utils/test-intent \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "why was my claim rejected"
  }'
```

**Expected Response**:
```json
{
  "text": "why was my claim rejected",
  "intent": "claim_status",
  "confidence": 0.92,
  "reasoning": "User asking about claim rejection reason",
  "timestamp": "2025-01-09T..."
}
```

### Test Intent Agent Node (Full)
**Endpoint**: `POST /utils/test-intent-agent`

**What it tests**: `agents/intent_agent.py`

**Purpose**: Test the complete intent agent with entity extraction

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_intent_agent(request: IntentTestRequest):`
2. `agents/intent_agent.py` - `async def intent_agent_node(state: AgentState)`
3. `agents/intent_agent.py` - `result = await classify_intent(...)`
4. `agents/intent_agent.py` - `return {"intent": intent, "confidence": confidence, "entities": entities}`

```bash
curl -X POST http://localhost:8000/utils/test-intent-agent \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Claim 12345 was rejected, why?",
    "user_info": {"user_id": "test_user"}
  }'
```

**Expected Response**:
```json
{
  "input": "Claim 12345 was rejected, why?",
  "intent": "claim_status",
  "confidence": 0.95,
  "entities": {
    "claim_number": "12345"
  },
  "timestamp": "2025-01-09T..."
}
```

---

## 2. Cache Operations

### Set Cache Value
**Endpoint**: `POST /utils/test-cache`

**What it tests**: `nodes/cache.py`, `memory/inmemory_store.py`

**Purpose**: Test storing data in cache

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_cache_operations(request: CacheTestRequest):`
2. `utils/test_endpoints.py` - `success = await memory_store.set(...)`
3. `memory/inmemory_store.py` - `async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None)`
4. `memory/inmemory_store.py` - `self._cache[key] = {...}`

```bash
curl -X POST http://localhost:8000/utils/test-cache \
  -H 'Content-Type: application/json' \
  -d '{
    "key": "test_key_123",
    "value": {"data": "test value", "number": 42},
    "ttl_seconds": 3600
  }'
```

**Expected Response**:
```json
{
  "operation": "set",
  "key": "test_key_123",
  "value": {"data": "test value", "number": 42},
  "success": true,
  "timestamp": "2025-01-09T..."
}
```

### Get Cache Value
**Endpoint**: `POST /utils/test-cache` (without value) or `GET /utils/test-cache/{key}`

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_cache_get(key: str):` (for GET)
2. `memory/inmemory_store.py` - `async def get(self, key: str)`
3. `memory/inmemory_store.py` - `return entry["value"]`

```bash
curl -X POST http://localhost:8000/utils/test-cache \
  -H 'Content-Type: application/json' \
  -d '{
    "key": "test_key_123"
  }'
```

**Or use GET**:
```bash
curl http://localhost:8000/utils/test-cache/test_key_123
```

### Delete Cache Value
**Endpoint**: `DELETE /utils/test-cache/{key}`

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_cache_delete(key: str):`
2. `memory/inmemory_store.py` - `async def delete(self, key: str)`

```bash
curl -X DELETE http://localhost:8000/utils/test-cache/test_key_123
```

---

## 3. Persistence/Telemetry

### Log Event to SQLite
**Endpoint**: `POST /utils/test-persistence`

**What it tests**: `persistence/sqlite_store.py`

**Purpose**: Test logging events to SQLite database

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_persistence_logging(request: PersistenceTestRequest):`
2. `core/telemetry.py` - `async def log_event(...)`
3. `persistence/sqlite_store.py` - `async def log_event(...)`
4. `persistence/sqlite_store.py` - `await db.execute("INSERT INTO events...")`

```bash
curl -X POST http://localhost:8000/utils/test-persistence \
  -H 'Content-Type: application/json' \
  -d '{
    "event_type": "CACHE_HIT",
    "session_id": "test_session_456",
    "data": {
      "key": "some_cache_key",
      "hit_count": 5
    }
  }'
```

**Event Types**: 
- `REQUEST_RECEIVED`
- `INTENT_CLASSIFIED`
- `RESPONSE_GENERATED`
- `CACHE_HIT`
- `CACHE_MISS`
- `TOOL_CALLED`
- `ERROR_OCCURRED`
- `SAFETY_BLOCKED`
- `CLARIFICATION_NEEDED`

**Expected Response**:
```json
{
  "event_id": "uuid-here",
  "event_type": "CACHE_HIT",
  "session_id": "test_session_456",
  "success": true,
  "timestamp": "2025-01-09T..."
}
```

### Get Session Events
**Endpoint**: `GET /utils/test-persistence/events/{session_id}`

```bash
curl http://localhost:8000/utils/test-persistence/events/test_session_456
```

**Expected Response**:
```json
{
  "session_id": "test_session_456",
  "event_count": 3,
  "events": [
    {
      "event_id": "...",
      "event_type": "CACHE_HIT",
      "data": {...},
      "timestamp": "..."
    }
  ],
  "timestamp": "2025-01-09T..."
}
```

---

## 4. Session Memory

### Add Message to Session
**Endpoint**: `POST /utils/test-session-history`

**What it tests**: `nodes/context.py`, `memory/inmemory_store.py`

**Purpose**: Test session conversation history

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_session_history(request: SessionTestRequest):`
2. `utils/test_endpoints.py` - `await memory_store.append_to_session(...)`
3. `memory/inmemory_store.py` - `async def append_to_session(...)`
4. `memory/inmemory_store.py` - `self._session_history[session_id].append(...)`

```bash
curl -X POST http://localhost:8000/utils/test-session-history \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "user_session_789",
    "role": "user",
    "content": "Hello, my claim number is 12345"
  }'
```

**Expected Response**:
```json
{
  "session_id": "user_session_789",
  "message_added": {
    "role": "user",
    "content": "Hello, my claim number is 12345"
  },
  "total_messages": 1,
  "history": [
    {
      "role": "user",
      "content": "Hello, my claim number is 12345",
      "timestamp": "..."
    }
  ],
  "timestamp": "2025-01-09T..."
}
```

### Get Session History
**Endpoint**: `GET /utils/test-session-history/{session_id}`

```bash
curl http://localhost:8000/utils/test-session-history/user_session_789
```

**Expected Response**:
```json
{
  "session_id": "user_session_789",
  "message_count": 2,
  "history": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ],
  "facts": [],
  "timestamp": "2025-01-09T..."
}
```

---

## 5. Context Building

### Test Context Retrieval
**Endpoint**: `POST /utils/test-context-building`

**What it tests**: `nodes/context.py`

**Purpose**: Test retrieving conversation context and facts

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_context_building(request: ContextTestRequest):`
2. `nodes/context.py` - `async def build_context_node(state: AgentState)`
3. `memory/inmemory_store.py` - `async def get_session_history(session_id)`
4. `memory/inmemory_store.py` - `async def get_session_facts(session_id)`

```bash
curl -X POST http://localhost:8000/utils/test-context-building \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "user_session_789",
    "text": "What was my claim number again?"
  }'
```

**Expected Response**:
```json
{
  "session_id": "user_session_789",
  "conversation_history": [
    {"role": "user", "content": "Hello, my claim number is 12345"}
  ],
  "relevant_facts": [
    {"type": "claim_mention", "data": {...}}
  ],
  "timestamp": "2025-01-09T..."
}
```

---

## 6. Safety Checks

### Test Safety Precheck
**Endpoint**: `POST /utils/test-safety-precheck`

**What it tests**: `nodes/safety.py`

**Purpose**: Test input safety validation

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_safety_precheck(request: SafetyTestRequest):`
2. `nodes/safety.py` - `async def safety_precheck_node(state: AgentState)`
3. `nodes/safety.py` - `return {"safety_precheck_passed": passed, ...}`

```bash
# Safe input
curl -X POST http://localhost:8000/utils/test-safety-precheck \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "What is my claim status?",
    "session_id": "safe_test"
  }'
```

**Expected Response**:
```json
{
  "text": "What is my claim status?",
  "safety_precheck_passed": true,
  "blocked_reason": null,
  "safety_score": 0.95,
  "timestamp": "2025-01-09T..."
}
```

### Test Safety Postcheck
**Endpoint**: `POST /utils/test-safety-postcheck`

**Purpose**: Test response safety validation

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_safety_postcheck(request: SafetyTestRequest):`
2. `nodes/safety.py` - `async def safety_postcheck_node(state: AgentState)`
3. `nodes/safety.py` - `return {"safety_postcheck_passed": passed, ...}`

```bash
curl -X POST http://localhost:8000/utils/test-safety-postcheck \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Generated response to check",
    "session_id": "safety_test"
  }'
```

---

## 7. Clarification Logic

### Test Clarification Detection
**Endpoint**: `POST /utils/test-clarification`

**What it tests**: `nodes/clarification.py`

**Purpose**: Test if user input needs clarification

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_clarification(request: IntentTestRequest):`
2. `nodes/clarification.py` - `async def clarification_node(state: AgentState)`
3. `nodes/clarification.py` - `return {"needs_clarification": needs_clarification, ...}`

```bash
# Missing claim number
curl -X POST http://localhost:8000/utils/test-clarification \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "why was my claim rejected"
  }'
```

**Expected Response**:
```json
{
  "text": "why was my claim rejected",
  "needs_clarification": true,
  "clarifying_question": "Could you please provide your claim number?",
  "missing_info": ["claim_number"],
  "timestamp": "2025-01-09T..."
}
```

```bash
# With claim number
curl -X POST http://localhost:8000/utils/test-clarification \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "claim 12345 was rejected, why?"
  }'
```

**Expected Response**:
```json
{
  "text": "claim 12345 was rejected, why?",
  "needs_clarification": false,
  "clarifying_question": null,
  "missing_info": [],
  "timestamp": "2025-01-09T..."
}
```

---

## 7.5. Confidence Checker

### Test Confidence Checker
**Endpoint**: `POST /utils/test-confidence-checker`

**What it tests**: `nodes/confidence.py` - `confidence_checker_node`

**Purpose**: Test confidence checking and routing logic. Takes intent classifier output, checks confidence against threshold from config, and either returns clarification or calls context builder.

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_confidence_checker(request: ConfidenceCheckRequest):`
2. `nodes/confidence.py` - `async def confidence_checker_node(state: AgentState)`
3. `nodes/confidence.py` - Confidence check decision logic
4. `nodes/context.py` - `async def build_context_node(state: AgentState)` (if high confidence)

**Request Payload (Low Confidence Example)**:
```bash
curl -X POST http://localhost:8000/utils/test-confidence-checker \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "why was my claim rejected",
    "intent": "claim_rejection_reason",
    "confidence": 0.45,
    "entities": {},
    "session_id": "test-session-123",
    "uuid": "req-uuid-456",
    "domain": "claims",
    "user_info": {"user_id": "member_222"}
  }'
```

**Expected Response (Low Confidence)**:
```json
{
  "decision": "clarification",
  "needs_clarification": true,
  "clarifying_question": "I'\''m not quite sure what you'\''re asking. Could you rephrase your question?",
  "response": "I'\''m not quite sure what you'\''re asking. Could you rephrase your question?",
  "metadata": {
    "clarification": true,
    "clarification_reason": "low_confidence",
    "missing_entities": [],
    "confidence": 0.45,
    "threshold": 0.7
  },
  "timestamp": "2025-01-09T..."
}
```

**Request Payload (High Confidence Example)**:
```bash
curl -X POST http://localhost:8000/utils/test-confidence-checker \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "what is the status of claim 12345678",
    "intent": "claim_status",
    "confidence": 0.92,
    "entities": {"claim_number": "12345678"},
    "session_id": "test-session-123",
    "uuid": "req-uuid-456",
    "domain": "claims",
    "user_info": {"user_id": "member_222"}
  }'
```

**Expected Response (High Confidence)**:
```json
{
  "decision": "proceed",
  "confidence_check_passed": true,
  "context_builder_input": {
    "intent": "claim_status",
    "confidence": 0.92,
    "entities": {"claim_number": "12345678"},
    "domain": "claims",
    "uuid": "req-uuid-456",
    "user_profile": {"user_id": "member_222"},
    "chat_history": []
  },
  "context_builder_output": {
    "conversation_history": [],
    "relevant_facts": []
  },
  "timestamp": "2025-01-09T..."
}
```

**Notes**:
- Confidence threshold is loaded from `config/domain_config.json`
- If confidence < threshold OR missing required entities → returns clarification
- If confidence >= threshold → calls context builder, logs to SQLite, returns context builder output
- All decisions and context builder input/output are logged to `logs` table in SQLite
- UUID ties all related logs together for audit trail

---

## 8. Claims API

### Test Claims API Mock
**Endpoint**: `POST /utils/test-claims-api`

**What it tests**: `tools/claims_api.py`

**Purpose**: Test the mock claims API integration

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_claims_api(request: Dict[str, Any]):`
2. `tools/claims_api.py` - `async def call_claims_tool_node(state: AgentState)`
3. `tools/claims_api.py` - `return {"tool_result": mock_data}`

```bash
curl -X POST http://localhost:8000/utils/test-claims-api \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Get status for claim 12345",
    "intent": "claim_status",
    "entities": {
      "claim_number": "12345"
    }
  }'
```

**Expected Response**:
```json
{
  "tool_called": true,
  "tool_result": {
    "claim_number": "12345",
    "status": "approved",
    "amount": "$500",
    "date": "2025-01-01"
  },
  "timestamp": "2025-01-09T..."
}
```

---

## 9. Response Generation

### Test Response Agent
**Endpoint**: `POST /utils/test-response-agent`

**What it tests**: `agents/response_agent.py`

**Purpose**: Test AI response generation

**🔴 Breakpoints to set:**
1. `utils/test_endpoints.py` - `async def test_response_agent(request: IntentTestRequest):`
2. `agents/response_agent.py` - `async def response_agent_node(state: AgentState)`
3. `agents/response_agent.py` - `response = await llm.ainvoke(prompt)`
4. `agents/response_agent.py` - `return {"response": response}`

```bash
curl -X POST http://localhost:8000/utils/test-response-agent \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Why was my claim rejected?"
  }'
```

**Expected Response**:
```json
{
  "input": "Why was my claim rejected?",
  "response": "Your claim was processed and the status is...",
  "intent": "claim_status",
  "timestamp": "2025-01-09T..."
}
```

---

## 10. Health Check

### Check Utils Endpoints Health
**Endpoint**: `GET /utils/health`

```bash
curl http://localhost:8000/utils/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "endpoints": [
    "/utils/test-intent",
    "/utils/test-intent-agent",
    "/utils/test-cache",
    ...
  ],
  "timestamp": "2025-01-09T..."
}
```

---

## 🔄 Complete Testing Workflow

### Scenario 1: Test Intent → Cache → Response

```bash
# 1. Test intent classification
curl -X POST http://localhost:8000/utils/test-intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"claim 99999 status"}' | jq

# 2. Cache the result
curl -X POST http://localhost:8000/utils/test-cache \
  -H 'Content-Type: application/json' \
  -d '{
    "key": "intent:99999",
    "value": {"intent":"claim_status","confidence":0.95},
    "ttl_seconds":3600
  }' | jq

# 3. Retrieve from cache
curl http://localhost:8000/utils/test-cache/intent:99999 | jq

# 4. Generate response
curl -X POST http://localhost:8000/utils/test-response-agent \
  -H 'Content-Type: application/json' \
  -d '{"text":"claim 99999 status"}' | jq
```

### Scenario 2: Test Session Memory Flow

```bash
SESSION="dev_test_session"

# 1. Add user message
curl -X POST http://localhost:8000/utils/test-session-history \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"$SESSION\",
    \"role\": \"user\",
    \"content\": \"Hello, my claim is 55555\"
  }" | jq

# 2. Add assistant response
curl -X POST http://localhost:8000/utils/test-session-history \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"$SESSION\",
    \"role\": \"assistant\",
    \"content\": \"I can help with claim 55555\"
  }" | jq

# 3. Add another user message
curl -X POST http://localhost:8000/utils/test-session-history \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"$SESSION\",
    \"role\": \"user\",
    \"content\": \"What was my claim number?\"
  }" | jq

# 4. Retrieve full history
curl "http://localhost:8000/utils/test-session-history/$SESSION" | jq

# 5. Build context from history
curl -X POST http://localhost:8000/utils/test-context-building \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"$SESSION\",
    \"text\": \"What was my claim number?\"
  }" | jq
```

### Scenario 3: Test Telemetry Logging

```bash
SESSION="telemetry_test"

# 1. Log a request event
curl -X POST http://localhost:8000/utils/test-persistence \
  -H 'Content-Type: application/json' \
  -d "{
    \"event_type\": \"REQUEST_RECEIVED\",
    \"session_id\": \"$SESSION\",
    \"data\": {\"text\": \"test query\"}
  }" | jq

# 2. Log a cache hit
curl -X POST http://localhost:8000/utils/test-persistence \
  -H 'Content-Type: application/json' \
  -d "{
    \"event_type\": \"CACHE_HIT\",
    \"session_id\": \"$SESSION\",
    \"data\": {\"key\": \"test_key\"}
  }" | jq

# 3. Log a response generated
curl -X POST http://localhost:8000/utils/test-persistence \
  -H 'Content-Type: application/json' \
  -d "{
    \"event_type\": \"RESPONSE_GENERATED\",
    \"session_id\": \"$SESSION\",
    \"data\": {\"response\": \"test response\"}
  }" | jq

# 4. Retrieve all events for session
curl "http://localhost:8000/utils/test-persistence/events/$SESSION" | jq

# 5. View overall analytics
curl http://localhost:8000/api/v1/analytics | jq
```

---

## 🧑‍💻 Developer Workflows

### Intent Classifier Team
```bash
# Test your classifier changes
curl -X POST http://localhost:8000/utils/test-intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"YOUR_TEST_TEXT_HERE"}'

# Test with different intents
curl -X POST http://localhost:8000/utils/test-intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"I want to appeal my claim"}'

curl -X POST http://localhost:8000/utils/test-intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"what drugs are covered"}'
```

### Cache Team
```bash
# Test cache set/get/delete
curl -X POST http://localhost:8000/utils/test-cache \
  -H 'Content-Type: application/json' \
  -d '{"key":"test","value":{"data":"value"}}'

curl http://localhost:8000/utils/test-cache/test

curl -X DELETE http://localhost:8000/utils/test-cache/test
```

### Persistence/Analytics Team
```bash
# Test event logging
curl -X POST http://localhost:8000/utils/test-persistence \
  -H 'Content-Type: application/json' \
  -d '{"event_type":"CACHE_HIT","session_id":"test","data":{}}'

# Query SQLite directly
sqlite3 data/telemetry.db "SELECT * FROM events LIMIT 10"
```

### Session Memory Team
```bash
# Test conversation history
curl -X POST http://localhost:8000/utils/test-session-history \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"test","role":"user","content":"test message"}'

curl http://localhost:8000/utils/test-session-history/test
```

---

## 📊 Monitoring & Debugging

### Check if all components are working:
```bash
# 1. Utils health check
curl http://localhost:8000/utils/health

# 2. Main app health
curl http://localhost:8000/health

# 3. API health
curl http://localhost:8000/api/v1/analytics
```

### View Swagger Docs:
```
http://localhost:8000/docs
```

All utils endpoints will be visible under the "Testing Utils" section.

---

## 🎯 Benefits

1. **Parallel Development**: Teams can work on different components independently
2. **Fast Iteration**: Test changes without running full pipeline
3. **Easy Debugging**: Isolate and test specific functionality
4. **Integration Testing**: Build up from unit tests to integration tests
5. **Documentation**: Each endpoint is self-documenting
6. **No Dependencies**: Test components without waiting for upstream/downstream work

---

## 🔧 Troubleshooting

### Endpoint not found?
- Make sure `python main.py` is running
- Check console for "[BOOT] Utils test endpoints loaded"

### Import errors?
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check virtual environment is activated

### Database errors?
- Ensure `data/` directory exists
- Check `.env` has `ENABLE_TELEMETRY=True`

### Cache not working?
- Check `.env` has `MEMORY_STORE_TYPE=inmemory`
- Check `.env` has `ENABLE_SEMANTIC_CACHE=True`

---

## 📝 Notes

- All endpoints use `POST` unless specified otherwise
- Responses are JSON format
- Times are in ISO 8601 format
- UUIDs are auto-generated where needed
- Mock data is used for claims API
- SQLite database at `data/telemetry.db`

---

**Happy Testing! 🚀**

