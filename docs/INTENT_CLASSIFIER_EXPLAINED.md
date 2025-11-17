# Intent Classifier Explained - For 18-Year-Olds with ML Knowledge

## 🎯 What is Intent Classification?

Think of intent classification like **reading someone's mind** - but with AI!

When a user types:
- "What's the status of my claim #12345?"
- "Why was my claim rejected?"
- "Find me a pharmacy near me"

The intent classifier figures out **what they want**:
- `claim_status` - They want to check claim status
- `claim_rejection_reason` - They want to know why it was rejected
- `find_pharmacy` - They want to find a pharmacy

It's like a **smart router** that understands natural language and categorizes it.

---

## 🤖 How It Works: The Big Picture

```
User Input: "What's my claim status?"
    ↓
Intent Classifier (LLM)
    ↓
Output: {
    "intent": "claim_status",
    "confidence": 0.95,
    "entities": {"claim_number": "12345"}
}
```

**In simple terms:**
1. User sends a message
2. We send it to an LLM (Large Language Model) like GPT-4 or Gemini
3. LLM analyzes the message and figures out what the user wants
4. LLM returns structured data (intent, confidence, entities)
5. We use this to decide what to do next

---

## 📝 Step-by-Step: How the Code Works

### Step 1: The Function Gets Called

```python
# agents/intent_agent.py - Line 74

async def intent_agent_node(state: AgentState) -> Dict[str, Any]:
    """Classify user intent and extract entities."""
```

**What happens:**
- LangGraph automatically calls this function
- It passes the full `AgentState` (which contains the user's message)
- The function must return a dictionary with `intent`, `confidence`, and `entities`

**Think of it like:** A function that receives a question and returns an answer about what the question means.

---

### Step 2: Extract Input from State

```python
# agents/intent_agent.py - Lines 84-94

text = state["text"]  # User's message: "What's my claim status?"
history = state.get("conversation_history", [])  # Previous messages

# Convert history to readable string
if isinstance(history, list):
    if history and isinstance(history[0], dict):
        history_str = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in history)
    else:
        history_str = "\n".join(str(h) for h in history)
else:
    history_str = str(history)
```

**What this does:**
- Gets the user's current message from `state["text"]`
- Gets conversation history (previous messages in the chat)
- Converts history into a readable string format

**Example:**
```python
# Input state:
state = {
    "text": "What's my claim status?",
    "conversation_history": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"}
    ]
}

# After extraction:
text = "What's my claim status?"
history_str = "user: Hello\nassistant: Hi! How can I help?"
```

**Why conversation history?** 
Context matters! If the user said "What about claim #12345?" after asking about claims, the history helps the LLM understand they're still talking about claims.

---

### Step 3: Choose LLM (Real or Mock)

```python
# agents/intent_agent.py - Lines 96-104

# Select LLM (mock vs real)
if settings.use_mock_llm:
    llm = MockLLM()  # Fake LLM for testing
else:
    llm = ChatOpenAI(
        model=settings.llm_model,  # e.g., "gpt-4" or "gemini-2.5-flash"
        temperature=settings.llm_temperature,  # e.g., 0.7
        openai_api_key=settings.openai_api_key,
    )
```

**What this does:**
- Checks if we're using a mock LLM (for testing without API keys)
- If real: Creates a connection to OpenAI's API
- If mock: Uses a fake LLM that just does keyword matching

**Why two options?**
- **Mock LLM**: Fast, free, works offline - perfect for development
- **Real LLM**: Actually understands language, more accurate - for production

**Temperature explained:**
- `temperature = 0.0`: Very deterministic, always same answer
- `temperature = 0.7`: Balanced creativity and consistency (recommended)
- `temperature = 1.0`: Very creative, might give different answers

---

### Step 4: Build the Prompt (The Instructions)

This is the **MOST IMPORTANT** part! The prompt tells the LLM what to do.

```python
# agents/intent_agent.py - Lines 106-121

# Build raw system prompt
raw_system_prompt = (
    "You are an intent classification agent for a pharmacy benefits system.\n\n"
    "Your job: Classify the user's intent and extract entities.\n\n"
    "Available intents:\n"
    "- claim_status: User wants to check claim status\n"
    "- claim_rejection_reason: User wants to know why claim was rejected\n"
    "- find_pharmacy: User wants to find a pharmacy\n"
    "- check_coverage: User wants to check medication coverage\n"
    "- unknown: Cannot determine intent\n\n"
    "Respond ONLY with JSON like:\n"
    '{"intent": "claim_status", "confidence": 0.95, "entities": {"claim_number": "12345"}}\n\n'
    "Be conservative with confidence; if unsure use lower confidence."
)
# Escape braces for Python .format safety
system_prompt = raw_system_prompt.replace('{', '{{').replace('}', '}}')
```

**Breaking down the prompt:**

1. **Role Definition**: "You are an intent classification agent..."
   - Tells the LLM its job/role

2. **Task Description**: "Your job: Classify the user's intent..."
   - Explains what it needs to do

3. **Available Intents**: Lists all possible intents
   - Like a multiple-choice question with options

4. **Output Format**: "Respond ONLY with JSON like..."
   - Forces the LLM to return structured data (JSON)
   - This is **critical** - we need parseable output!

5. **Confidence Guidance**: "Be conservative with confidence..."
   - Tells the LLM to be honest about uncertainty

**Why escape braces?**
Python's `.format()` uses `{}` for placeholders. Since our prompt contains JSON with `{}`, we need to escape them as `{{` and `}}` so Python doesn't try to format them.

---

### Step 5: Create the Full Prompt Template

```python
# agents/intent_agent.py - Lines 123-130

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),  # Instructions for the LLM
    ("user", (
        "User message: {user_text}\n\n"
        "Conversation history:\n{conversation_history}\n\n"
        "Classify this."
    )),
])
```

**What this creates:**

A prompt template with two parts:

1. **System Message** (instructions):
   ```
   You are an intent classification agent...
   Available intents:
   - claim_status: ...
   - claim_rejection_reason: ...
   ...
   ```

2. **User Message** (the actual question):
   ```
   User message: What's my claim status?
   
   Conversation history:
   user: Hello
   assistant: Hi! How can I help?
   
   Classify this.
   ```

**Think of it like:**
- **System message** = Instructions for a student (the LLM)
- **User message** = The actual homework question

---

### Step 6: Format the Messages

```python
# agents/intent_agent.py - Lines 132-136

# Format messages
messages = prompt.format_messages(
    user_text=text,  # "What's my claim status?"
    conversation_history=history_str if history_str.strip() else "(none)"
)
```

**What this does:**
- Takes the template and fills in the placeholders
- Creates the final messages to send to the LLM

**Result:**
```python
messages = [
    {
        "role": "system",
        "content": "You are an intent classification agent..."
    },
    {
        "role": "user",
        "content": "User message: What's my claim status?\n\nConversation history:\n(none)\n\nClassify this."
    }
]
```

---

### Step 7: Call the LLM

```python
# agents/intent_agent.py - Line 139

# Invoke LLM
response = await llm.ainvoke(messages)
```

**What happens:**
- Sends the messages to the LLM (OpenAI API or MockLLM)
- Waits for response (async/await)
- Gets back a response object

**For MockLLM (testing):**
```python
# agents/intent_agent.py - Lines 28-68

class MockLLM:
    """Fake LLM for development"""
    
    async def ainvoke(self, messages: List[Any]):
        await asyncio.sleep(0.05)  # Simulate API delay
        
        # Extract user message
        user_msg = ""
        for m in messages:
            role = getattr(m, "role", getattr(m, "type", ""))
            if role in ("user", "human"):
                user_msg = m.content
        text = user_msg.lower()
        
        # Simple keyword matching (not real ML!)
        if "status" in text and "claim" in text:
            intent = "claim_status"
            confidence = 0.92
        elif "reject" in text or "denied" in text:
            intent = "claim_rejection_reason"
            confidence = 0.88
        elif any(greet in text for greet in ["hello", "hi", "hey"]):
            intent = "greeting"
            confidence = 0.45
        else:
            intent = "unknown"
            confidence = 0.30
        
        # Extract entities using regex
        entities: Dict[str, Any] = {}
        claim_match = re.search(r"\b\d{4,10}\b", text)  # Find 4-10 digit numbers
        if claim_match and "claim" in text:
            entities["claim_number"] = claim_match.group(0)
        
        # Return JSON string
        class Response:
            content = json.dumps({
                "intent": intent,
                "confidence": confidence,
                "entities": entities
            })
        return Response()
```

**MockLLM explanation:**
- **Not real ML!** Just keyword matching
- Looks for specific words like "status" + "claim" → `claim_status`
- Uses regex to find claim numbers (4-10 digits)
- Returns fake but realistic responses

**For Real LLM (production):**
- Actually understands language
- Can handle variations: "check my claim", "what's my claim status", "claim status please"
- More accurate but costs money and requires API key

---

### Step 8: Parse the Response

```python
# agents/intent_agent.py - Lines 141-145

# Parse JSON safely
try:
    result = json.loads(response.content)
except Exception:
    result = {"intent": "unknown", "confidence": 0.1, "entities": {}}
```

**What this does:**
- The LLM returns a JSON string like: `'{"intent": "claim_status", "confidence": 0.95, "entities": {"claim_number": "12345"}}'`
- We parse it into a Python dictionary
- If parsing fails (LLM returned invalid JSON), we use safe defaults

**Example:**
```python
# LLM response:
response.content = '{"intent": "claim_status", "confidence": 0.95, "entities": {"claim_number": "12345"}}'

# After parsing:
result = {
    "intent": "claim_status",
    "confidence": 0.95,
    "entities": {"claim_number": "12345"}
}
```

**Why try/except?**
LLMs sometimes mess up and return invalid JSON. We handle it gracefully instead of crashing.

---

### Step 9: Extract Values

```python
# agents/intent_agent.py - Lines 147-149

intent = result.get("intent", "unknown")
confidence = float(result.get("confidence", 0.1))
entities = result.get("entities") or {}
```

**What this does:**
- Extracts intent, confidence, and entities from the parsed result
- Uses `.get()` with defaults for safety
- Converts confidence to float

**Example:**
```python
result = {
    "intent": "claim_status",
    "confidence": 0.95,
    "entities": {"claim_number": "12345"}
}

# After extraction:
intent = "claim_status"
confidence = 0.95  # (as float)
entities = {"claim_number": "12345"}
```

---

### Step 10: Return the Result

```python
# agents/intent_agent.py - Lines 151-157

logger.info(f"🎯 Intent: {intent} ({confidence:.2f}) | Entities: {entities}")

return {
    "intent": intent,
    "confidence": confidence,
    "entities": entities,
}
```

**What this does:**
- Logs the result (for debugging)
- Returns a dictionary that LangGraph will merge into AgentState

**The return value:**
```python
{
    "intent": "claim_status",
    "confidence": 0.95,
    "entities": {"claim_number": "12345"}
}
```

**LangGraph automatically:**
- Takes this return value
- Merges it into the full AgentState
- Passes updated state to the next node

---

## 🎬 Complete Example: End-to-End

Let's trace through a real example:

### Input:
```python
state = {
    "text": "What's the status of claim #12345?",
    "session_id": "session_123",
    "conversation_history": []
}
```

### Step-by-Step Execution:

**1. Function called:**
```python
result = await intent_agent_node(state)
```

**2. Extract input:**
```python
text = "What's the status of claim #12345?"
history_str = "(none)"
```

**3. Choose LLM:**
```python
# Assuming use_mock_llm = False
llm = ChatOpenAI(model="gpt-4", temperature=0.7, ...)
```

**4. Build prompt:**
```python
messages = [
    {
        "role": "system",
        "content": "You are an intent classification agent..."
    },
    {
        "role": "user",
        "content": "User message: What's the status of claim #12345?\n\nConversation history:\n(none)\n\nClassify this."
    }
]
```

**5. Call LLM:**
```python
response = await llm.ainvoke(messages)
# LLM processes and returns:
response.content = '{"intent": "claim_status", "confidence": 0.95, "entities": {"claim_number": "12345"}}'
```

**6. Parse response:**
```python
result = json.loads(response.content)
# result = {
#     "intent": "claim_status",
#     "confidence": 0.95,
#     "entities": {"claim_number": "12345"}
# }
```

**7. Extract values:**
```python
intent = "claim_status"
confidence = 0.95
entities = {"claim_number": "12345"}
```

**8. Return:**
```python
return {
    "intent": "claim_status",
    "confidence": 0.95,
    "entities": {"claim_number": "12345"}
}
```

**9. LangGraph merges:**
```python
# Updated state:
state = {
    "text": "What's the status of claim #12345?",
    "session_id": "session_123",
    "intent": "claim_status",  # ← Added
    "confidence": 0.95,        # ← Added
    "entities": {"claim_number": "12345"}  # ← Added
}
```

---

## 🧠 ML Concepts Explained

### 1. **Large Language Models (LLMs)**

**What they are:**
- Neural networks trained on massive amounts of text
- Can understand and generate human-like text
- Examples: GPT-4, Gemini, Claude

**How they work (simplified):**
1. Trained on billions of text examples
2. Learn patterns in language
3. Can predict what comes next in a sequence
4. Can follow instructions (if trained properly)

**In our code:**
```python
llm = ChatOpenAI(model="gpt-4", ...)
response = await llm.ainvoke(messages)
```

We're using GPT-4 (or similar) to understand the user's message.

---

### 2. **Prompt Engineering**

**What it is:**
- The art/science of writing instructions for LLMs
- Small changes in prompts = big changes in output
- Critical for getting good results

**In our code:**
```python
raw_system_prompt = (
    "You are an intent classification agent...\n"
    "Available intents:\n"
    "- claim_status: ...\n"
    ...
)
```

**Why it matters:**
- Bad prompt: LLM might return random text
- Good prompt: LLM returns structured JSON exactly as we need

**Key techniques we use:**
1. **Role definition**: "You are an intent classification agent..."
2. **Clear instructions**: "Classify the user's intent..."
3. **Examples**: Shows LLM the format we want
4. **Constraints**: "Respond ONLY with JSON..."

---

### 3. **Structured Output**

**What it is:**
- Forcing LLM to return data in a specific format (JSON)
- Makes it easier to parse and use programmatically

**In our code:**
```python
# We tell LLM to return JSON:
'{"intent": "claim_status", "confidence": 0.95, "entities": {"claim_number": "12345"}}'

# Then parse it:
result = json.loads(response.content)
```

**Why it's important:**
- Without structure: LLM might return "The user wants to check claim status"
- With structure: LLM returns `{"intent": "claim_status"}` - easy to use!

---

### 4. **Entity Extraction**

**What it is:**
- Finding specific pieces of information in text
- Like finding names, dates, numbers, etc.

**In our code:**
```python
# MockLLM uses regex:
claim_match = re.search(r"\b\d{4,10}\b", text)  # Find 4-10 digit numbers
if claim_match and "claim" in text:
    entities["claim_number"] = claim_match.group(0)
```

**Real LLM:**
- Understands context better
- Can extract entities even if format varies
- Example: "claim 12345", "claim number 12345", "#12345" all work

---

### 5. **Confidence Scores**

**What it is:**
- A number (0.0 to 1.0) indicating how sure the model is
- 1.0 = 100% sure, 0.0 = completely unsure

**In our code:**
```python
confidence = float(result.get("confidence", 0.1))
```

**Why it matters:**
- High confidence (0.9+): Proceed with action
- Low confidence (0.5-): Ask for clarification
- Very low (<0.5): Return "unknown" intent

**Example:**
- "What's my claim status?" → confidence: 0.95 (very clear)
- "Help me" → confidence: 0.3 (unclear, could be anything)

---

## 🔍 Code Deep Dive: Key Sections

### The MockLLM (For Testing)

```python
# agents/intent_agent.py - Lines 28-68

class MockLLM:
    """Fake LLM for development"""
    
    async def ainvoke(self, messages: List[Any]):
        # Simulate API delay
        await asyncio.sleep(0.05)
        
        # Extract user message from messages
        user_msg = ""
        for m in messages:
            role = getattr(m, "role", getattr(m, "type", ""))
            if role in ("user", "human"):
                user_msg = m.content
        text = user_msg.lower()
        
        # Simple keyword matching (NOT real ML!)
        if "status" in text and "claim" in text:
            intent = "claim_status"
            confidence = 0.92
        elif "reject" in text or "denied" in text:
            intent = "claim_rejection_reason"
            confidence = 0.88
        # ... more patterns
        
        # Extract entities using regex
        entities: Dict[str, Any] = {}
        claim_match = re.search(r"\b\d{4,10}\b", text)
        if claim_match and "claim" in text:
            entities["claim_number"] = claim_match.group(0)
        
        # Return as JSON string
        class Response:
            content = json.dumps({
                "intent": intent,
                "confidence": confidence,
                "entities": entities
            })
        return Response()
```

**What this does:**
- **Not real ML!** Just pattern matching
- Looks for keywords: "status" + "claim" → `claim_status`
- Uses regex to find claim numbers
- Returns fake but realistic responses

**Why use it?**
- No API key needed
- Fast (no network calls)
- Free (no costs)
- Good for development and testing

---

### The Real LLM Call

```python
# agents/intent_agent.py - Lines 100-104

llm = ChatOpenAI(
    model=settings.llm_model,  # e.g., "gpt-4" or "gemini-2.5-flash"
    temperature=settings.llm_temperature,  # 0.7
    openai_api_key=settings.openai_api_key,
)

# Later...
response = await llm.ainvoke(messages)
```

**What happens:**
1. Creates connection to OpenAI API
2. Sends messages (system + user prompt)
3. Waits for response (async)
4. Gets back structured response

**Behind the scenes:**
- HTTP request to OpenAI API
- LLM processes the prompt
- Returns JSON string
- We parse it

---

### Error Handling

```python
# agents/intent_agent.py - Lines 159-202

except Exception as e:
    tb = traceback.format_exc()
    
    # Check if it's an LLM-related error
    if "openai" in str(e).lower() or "llm" in str(e).lower():
        error = create_llm_error(...)
    else:
        error = create_internal_error(...)
    
    # Log exception to database
    await persistence_store.log_exception(...)
    
    # Return safe defaults
    return {
        "error": error.user_message,
        "intent": "unknown",
        "confidence": 0.0,
        "entities": {},
    }
```

**What this does:**
- Catches any errors (network issues, API errors, parsing errors)
- Logs them to database for debugging
- Returns safe defaults so the system doesn't crash
- User gets a graceful error message

**Why it's important:**
- LLM APIs can fail (network issues, rate limits, etc.)
- We don't want the whole system to crash
- Better to return "unknown" intent than crash

---

## 📊 Real-World Examples

### Example 1: Clear Intent

**Input:**
```
"What's the status of my claim #12345?"
```

**Processing:**
1. LLM sees: "status" + "claim" + number
2. Matches pattern for `claim_status`
3. Extracts entity: `claim_number = "12345"`

**Output:**
```json
{
    "intent": "claim_status",
    "confidence": 0.95,
    "entities": {"claim_number": "12345"}
}
```

---

### Example 2: Unclear Intent

**Input:**
```
"Help me"
```

**Processing:**
1. LLM sees: Generic request, no specific intent
2. Can't match to any specific intent
3. Low confidence

**Output:**
```json
{
    "intent": "unknown",
    "confidence": 0.3,
    "entities": {}
}
```

**What happens next:**
- Confidence is low (< 0.7 threshold)
- System asks for clarification: "I'm not quite sure what you're asking. Could you rephrase your question?"

---

### Example 3: With Context

**Input:**
```
User: "What about claim #12345?"
(Previous: "I have a question about my claim")
```

**Processing:**
1. LLM sees current message: "What about claim #12345?"
2. LLM sees history: Previous message about claim
3. Understands context: Still talking about claim status
4. High confidence because of context

**Output:**
```json
{
    "intent": "claim_status",
    "confidence": 0.92,
    "entities": {"claim_number": "12345"}
}
```

**Why context matters:**
- Without context: "What about claim #12345?" is unclear
- With context: Clear it's about claim status

---

## 🎓 Key Takeaways

1. **Intent classification = Understanding what user wants**
   - Takes natural language input
   - Returns structured data (intent, confidence, entities)

2. **Uses LLM (Large Language Model)**
   - Real LLM: GPT-4, Gemini (understands language)
   - Mock LLM: Keyword matching (for testing)

3. **Prompt engineering is critical**
   - Good prompts = good results
   - Bad prompts = garbage output

4. **Structured output is essential**
   - Forces LLM to return JSON
   - Makes it easy to parse and use

5. **Error handling is important**
   - LLM APIs can fail
   - Always have fallbacks
   - Return safe defaults

6. **Context matters**
   - Conversation history helps
   - Makes classification more accurate

---

## 🔧 How to Improve It

### 1. **Better Prompts**
- Add more examples
- Be more specific about edge cases
- Use few-shot learning (show examples)

### 2. **Structured Output (Future)**
- Use OpenAI's structured output feature
- Guarantees valid JSON
- No parsing errors

### 3. **Fine-tuning**
- Train a custom model on your data
- More accurate for your specific domain
- Better entity extraction

### 4. **Hybrid Approach**
- Use keyword matching for simple cases (fast, free)
- Use LLM for complex cases (accurate, costs money)
- Best of both worlds

---

## 📚 Summary

The intent classifier is like a **smart translator**:
- **Input**: Natural language ("What's my claim status?")
- **Process**: LLM analyzes and understands
- **Output**: Structured data (`{"intent": "claim_status", ...}`)

It uses:
- **LLMs** to understand language
- **Prompt engineering** to get good results
- **Structured output** to make it usable
- **Error handling** to be robust

The code is well-structured, handles errors gracefully, and works with both mock (testing) and real (production) LLMs.

**That's how intent classification works in this codebase!** 🎉

